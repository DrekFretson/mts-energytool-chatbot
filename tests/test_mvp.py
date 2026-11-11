"""Запуск: python -m unittest discover -s tests -v"""

import csv
import io
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "core"), str(ROOT / "tests")]

import fake_llm  # noqa: E402

server = fake_llm.make_server()
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE_URL = f"http://127.0.0.1:{server.server_port}/v1"
os.environ.update(LLM_BASE_URL=BASE_URL, LLM_API_KEY="test", DEBUG_TOKEN="secret")
TMP = tempfile.TemporaryDirectory()
os.environ.update(DB_PATH=f"{TMP.name}/app.sqlite")

import kb  # noqa: E402
import pipeline  # noqa: E402
import router  # noqa: E402

TABLE = """# Заголовок

## Числа

  ------------------------------------------
  Показатель            Значение Оговорка
  -------------- --------------- -----------
  Порог доступа          750 кВт Ниже порога
  к рынку                        нельзя

  Экономия                 5--15% Не гарантия
  ------------------------------------------

Текст после таблицы.

## Простая

  Имя   Цена
  ----- ------
  А     100
  Б     200

Абзац.
"""

REAL_KB = ROOT / "data" / "knowledge-base.zip"


def make_zip(path):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a/doc.md", TABLE)
        archive.writestr("a/skip.bin", "x")


class Parsing(unittest.TestCase):
    def test_framed_table_keeps_caveat_with_number(self):
        text = dict(kb.split_sections("doc.md", TABLE))["Заголовок > Числа"]
        self.assertIn("Показатель: Порог доступа к рынку; Значение: 750 кВт; Оговорка: Ниже порога нельзя", text)
        self.assertIn("Текст после таблицы.", text)

    def test_simple_table(self):
        text = dict(kb.split_sections("doc.md", TABLE))["Заголовок > Простая"]
        self.assertIn("Имя: Б; Цена: 200", text)
        self.assertIn("Абзац.", text)

    @unittest.skipUnless(REAL_KB.exists(), "нет data/knowledge-base.zip")
    def test_real_kb_loses_no_numbers(self):
        chunks = kb.load_chunks(REAL_KB)
        with zipfile.ZipFile(REAL_KB) as archive:
            for name in archive.namelist():
                source = archive.read(name).decode("utf-8-sig")
                text = " ".join(c["crumb"] + " " + c["text"] for c in chunks if c["file"] == name)
                self.assertEqual(set(re.findall(r"\d+", source)) - set(re.findall(r"\d+", text)), set(), name)
        self.assertLessEqual(max(len(c["text"]) for c in chunks), 2 * 1500)

    @unittest.skipUnless(REAL_KB.exists(), "нет data/knowledge-base.zip")
    def test_real_kb_row_stays_together(self):
        joined = "\n".join(c["text"] for c in kb.load_chunks(REAL_KB))
        self.assertRegex(joined, r"Порог доступа к оптовому рынку; Значение: 750 кВт присоединённой мощности; Оговорка: Ниже порога")

    def test_escaped_heading_glued_to_paragraph_starts_section(self):
        text = "# Док\n\n## Результат\n\nТекст про результат. \\## Что нужно от клиента\n\nВыгрузка данных.\n"
        sections = dict(kb.split_sections("doc.md", text))
        self.assertEqual(sections["Док > Результат"], "Текст про результат.")
        self.assertEqual(sections["Док > Что нужно от клиента"], "Выгрузка данных.")

    def test_pandoc_dashes_become_real_dashes(self):
        text = "# Док\n\n## Раздел\n\nДиапазон --- 15--500 кВт для компании-посредника.\n\n- пункт списка\n\n-----\n"
        (crumb, content), = kb.split_sections("doc.md", text)
        self.assertEqual(content, "Диапазон — 15–500 кВт для компании-посредника.\n\n- пункт списка")

    def test_parse_dialog(self):
        history = pipeline.parse_dialog("Пользователь: А\nстрока\nАссистент: Б\nПользователь: В")
        self.assertEqual([t["role"] for t in history], ["user", "assistant", "user"])
        self.assertEqual(history[0]["content"], "А\nстрока")
        self.assertEqual(pipeline.parse_dialog("Просто вопрос"), [{"role": "user", "content": "Просто вопрос"}])


GUARDED = """# Продукт А

## Что это

Платформа учёта для складов.

## Цены

Подключение склада от 50 000 рублей.

## Факты, которые можно утверждать

Минимального числа приборов нет.

## Чего утверждать нельзя

Нельзя обещать экономию.
"""

OTHER = """# Продукт Б

## Что это

Солнечные станции на крышах.

## Чего утверждать нельзя

Нельзя обещать автономность.
"""


class Context(unittest.TestCase):
    def setUp(self):
        import config
        import retrieval

        self.config = config
        self.saved = config.CONTEXT_CHARS
        with tempfile.TemporaryDirectory() as work:
            path = Path(work, "kb.zip")
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("a.md", GUARDED)
                archive.writestr("b.md", OTHER)
            self.index = retrieval.Index(kb.load_chunks(path))

    def tearDown(self):
        self.config.CONTEXT_CHARS = self.saved

    def crumbs(self, query):
        return [s["crumb"] for s in self.index.context(self.index.search(query))]

    def test_guards_follow_their_document(self):
        crumbs = self.crumbs("цены подключение склада")
        self.assertIn("Продукт А > Цены", crumbs)
        self.assertIn("Продукт А > Чего утверждать нельзя", crumbs)
        self.assertIn("Продукт А > Факты, которые можно утверждать", crumbs)
        # Разделы одного документа идут подряд в его порядке.
        a = [c for c in crumbs if c.startswith("Продукт А")]
        self.assertEqual(crumbs[: len(a)], a)
        self.assertEqual(a, sorted(a, key=lambda crumb: GUARDED.index("## " + crumb.split(" > ")[-1])))

    def test_budget_is_respected_and_sections_are_whole(self):
        self.config.CONTEXT_CHARS = 60
        sections = self.index.context(self.index.search("цены подключение склада"))
        self.assertLessEqual(sum(len(s["text"]) for s in sections), 60)
        self.assertTrue(all(s["text"] in GUARDED + OTHER for s in sections))

    @unittest.skipUnless(REAL_KB.exists(), "нет data/knowledge-base.zip")
    def test_real_kb_long_section_is_whole(self):
        import retrieval

        index = retrieval.Index(kb.load_chunks(REAL_KB))
        sections = index.context(index.search("Сколько стоит солнечная станция и как считается эффект"))
        self.assertLessEqual(sum(len(s["text"]) for s in sections), self.config.CONTEXT_CHARS)
        effect = [s for s in sections if s["crumb"].endswith("Как считается эффект")]
        if effect:
            self.assertGreater(len(effect[0]["text"]), 1500)


class Routing(unittest.TestCase):
    def test_embeddings_use_own_url(self):
        import httpx
        import llm

        original = llm.EMB_BASE_URL
        llm.EMB_BASE_URL = "http://127.0.0.1:9/v1"
        try:
            with self.assertRaises(httpx.TransportError):
                llm.embed(["текст"])
            self.assertIn("fake-llm", llm.chat([{"role": "user", "content": "вопрос"}]))
        finally:
            llm.EMB_BASE_URL = original
        self.assertEqual(llm.embed(["текст"]).shape[0], 1)

    def test_reasoning_switches_go_to_any_endpoint_provider_only_to_openrouter(self):
        import llm

        messages = [{"role": "user", "content": "вопрос"}]
        platform = llm.chat_body(messages)
        # Какое поле понимает прокси платформы, неизвестно: шлём все, отвергнутое убирает _post_chat.
        effort = llm.config.REASONING_EFFORT
        self.assertEqual(platform["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(platform["reasoning"], {"effort": effort})
        self.assertEqual(platform["reasoning_effort"], effort)
        self.assertNotIn("provider", platform)

        saved = llm.BASE_URL, llm.LLM_PROVIDER
        llm.BASE_URL, llm.LLM_PROVIDER = "https://openrouter.ai/api/v1", "deepinfra/bf16"
        try:
            stand = llm.chat_body(messages)
        finally:
            llm.BASE_URL, llm.LLM_PROVIDER = saved
        self.assertEqual(stand["reasoning"], {"effort": effort})
        self.assertEqual(stand["provider"], {"order": ["deepinfra/bf16"], "allow_fallbacks": False})


class Submission(unittest.TestCase):
    def run_solution(self, code, extra_env=None, llm_url=BASE_URL):
        with tempfile.TemporaryDirectory() as work:
            data = Path(work, "data")
            data.mkdir()
            make_zip(data / "knowledge-base.zip")
            (data / "questions.csv").write_text(
                '﻿id,question\n1,"Вопрос, с запятой"\n2,"Пользователь: А\nАссистент: Б\nПользователь: В"\n',
                encoding="utf-8",
            )
            env = {**os.environ, "PYTHONPATH": str(ROOT / "core"), "DATA_DIR": "data", "LLM_BASE_URL": llm_url, **(extra_env or {})}
            done = subprocess.run([sys.executable, "-c", code], cwd=work, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr)
            content = Path(work, "submission.csv").read_text(encoding="utf-8")
            self.assertEqual(content.splitlines()[0], '"id","answer"')
            return list(csv.DictReader(io.StringIO(content)))

    def test_full_run(self):
        rows = self.run_solution("import solution; solution.main()")
        self.assertEqual([r["id"] for r in rows], ["1", "2"])
        self.assertTrue(all("fake-llm" in r["answer"] for r in rows))

    def test_llm_down_still_gives_full_csv(self):
        rows = self.run_solution(
            "import llm, solution; llm.time.sleep = lambda s: None; solution.main()",
            llm_url="http://127.0.0.1:9/v1",
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["answer"] for r in rows))

    def test_deadline_leaves_valid_csv(self):
        slow = fake_llm.make_server()
        original = fake_llm.CHAT_DELAY
        fake_llm.CHAT_DELAY = 5
        threading.Thread(target=slow.serve_forever, daemon=True).start()
        try:
            rows = self.run_solution(
                "import config; config.DEADLINE_SEC = 1\nimport solution; solution.main()",
                llm_url=f"http://127.0.0.1:{slow.server_port}/v1",
            )
        finally:
            fake_llm.CHAT_DELAY = original
            slow.shutdown()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all("fake-llm" not in r["answer"] and r["answer"] for r in rows))


class Web(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        sys.path.insert(0, str(ROOT))
        kb_path = Path(TMP.name, "kb.zip")
        make_zip(kb_path)
        os.environ["KB_PATH"] = str(kb_path)
        from web import app as web_app

        cls.client_context = TestClient(web_app.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def session(self):
        return self.client.post("/api/sessions").json()["id"]

    def test_scenario_question_to_lead_and_history(self):
        sid = self.session()
        texts = ["вопрос", "вопрос [ready]", "вопрос [ready]", "вопрос"]
        offers = [self.client.post(f"/api/sessions/{sid}/messages", json={"text": t}).json()["offer_lead"] for t in texts]
        self.assertEqual(offers, [False, True, False, False])

        lead = {"name": "Тест Тестов", "phone": "8 (900) 000-00-00", "company": "ООО Тест", "consent": True}
        created = self.client.post(f"/api/sessions/{sid}/lead", json=lead).json()
        self.assertEqual(created["status"], "created")
        self.assertTrue(created["summary"])
        self.assertEqual(self.client.post(f"/api/sessions/{sid}/lead", json=lead).json()["status"], "exists")

        detail = self.client.get(f"/api/admin/sessions/{sid}", headers={"X-Debug-Token": "secret"}).json()
        self.assertEqual(len(detail["messages"]), 8)
        self.assertEqual(detail["lead"]["phone"], "+79000000000")

        notes = []
        with mock.patch.object(pipeline, "answer", lambda index, history, note="": notes.append(note) or {"text": "ок", "route": "operator"}):
            after = self.client.post(f"/api/sessions/{sid}/messages", json={"text": "Позовите менеджера"}).json()
        self.assertEqual((after["open_form"], after["offer_lead"]), (False, False))
        self.assertIn("уже отправил заявку", notes[0])

    def test_operator_route_opens_lead_form(self):
        sid = self.session()
        decision = {"route": "operator", "reason": "просит менеджера", "ok": True, "ms": 1}
        with mock.patch.object(router, "classify", return_value=decision):
            reply = self.client.post(f"/api/sessions/{sid}/messages", json={"text": "Позовите менеджера"}).json()
        self.assertTrue(reply["open_form"])
        self.assertFalse(reply["offer_lead"])

    def test_lead_requires_consent_and_valid_phone(self):
        sid = self.session()
        lead = {"name": "Тест", "phone": "+79000000000", "company": "ООО", "consent": False}
        self.assertEqual(self.client.post(f"/api/sessions/{sid}/lead", json=lead).status_code, 422)
        lead.update(consent=True, phone="1234567890123")
        self.assertEqual(self.client.post(f"/api/sessions/{sid}/lead", json=lead).status_code, 422)

    def test_debug_and_admin_need_token(self):
        sid = self.session()
        plain = self.client.post(f"/api/sessions/{sid}/messages", json={"text": "вопрос"}).json()
        self.assertNotIn("debug", plain)
        wrong = self.client.post(f"/api/sessions/{sid}/messages", json={"text": "вопрос"}, headers={"X-Debug-Token": "nope"}).json()
        self.assertNotIn("debug", wrong)
        self.assertEqual(wrong["debug_error"], "Неверный токен отладки")
        good = self.client.post(f"/api/sessions/{sid}/messages", json={"text": "вопрос"}, headers={"X-Debug-Token": "secret"}).json()
        self.assertIn("total_ms", good["debug"]["timings"])
        self.assertTrue(good["debug"]["sources"])
        self.assertEqual(self.client.get("/api/admin/sessions").status_code, 403)

    def test_new_dialog_does_not_see_old_context(self):
        first, second = self.session(), self.session()
        self.client.post(f"/api/sessions/{first}/messages", json={"text": "секрет первой сессии"})
        self.client.post(f"/api/sessions/{second}/messages", json={"text": "другое"})
        detail = self.client.get(f"/api/admin/sessions/{second}", headers={"X-Debug-Token": "secret"}).json()
        self.assertNotIn("секрет", " ".join(m["content"] for m in detail["messages"]))

    def test_missing_kb_stops_with_clear_message(self):
        env = {**os.environ, "KB_PATH": str(Path(TMP.name, "missing.zip"))}
        done = subprocess.run([sys.executable, "-c", "import web.app"], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(done.returncode, 1)
        self.assertIn("data/knowledge-base.zip", done.stderr)
        self.assertNotIn("Traceback", done.stderr)

    def test_unknown_session(self):
        self.assertEqual(self.client.post("/api/sessions/none/messages", json={"text": "x"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()


class Materials(unittest.TestCase):
    def test_sections_grouped_by_document_with_product_title(self):
        hits = [
            {"file": "a.md", "crumb": "Платформа учёта > Что это", "text": "Учёт по объектам."},
            {"file": "a.md", "crumb": "Платформа учёта > Ограничения", "text": "Старая система не препятствие."},
            {"file": "b.md", "crumb": "Оптовый рынок > Аудит", "text": "Пригодность учёта решает аудит."},
            {"file": "c.md", "crumb": "Справка", "text": "Без подзаголовка."},
        ]
        text = pipeline.format_materials(hits)
        self.assertEqual(text.count("<document "), 3)
        self.assertEqual(text.count("</document>"), 3)
        platform, market = text.split('<document title="Оптовый рынок">')
        self.assertIn("Старая система не препятствие.", platform)
        self.assertNotIn("Старая система", market)
        self.assertIn("[3] Аудит", market)
        self.assertIn('<document title="Справка">\n[4] Справка', text)
