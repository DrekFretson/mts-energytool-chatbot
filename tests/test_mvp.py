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

    def test_parse_dialog(self):
        history = pipeline.parse_dialog("Пользователь: А\nстрока\nАссистент: Б\nПользователь: В")
        self.assertEqual([t["role"] for t in history], ["user", "assistant", "user"])
        self.assertEqual(history[0]["content"], "А\nстрока")
        self.assertEqual(pipeline.parse_dialog("Просто вопрос"), [{"role": "user", "content": "Просто вопрос"}])


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
        offers = [self.client.post(f"/api/sessions/{sid}/messages", json={"text": f"вопрос {i}"}).json()["offer_lead"] for i in range(4)]
        self.assertEqual(offers, [False, False, True, False])

        lead = {"name": "Тест Тестов", "phone": "8 (900) 000-00-00", "company": "ООО Тест", "consent": True}
        self.assertEqual(self.client.post(f"/api/sessions/{sid}/lead", json=lead).json()["status"], "created")
        self.assertEqual(self.client.post(f"/api/sessions/{sid}/lead", json=lead).json()["status"], "exists")

        detail = self.client.get(f"/api/admin/sessions/{sid}", headers={"X-Debug-Token": "secret"}).json()
        self.assertEqual(len(detail["messages"]), 8)
        self.assertEqual(detail["lead"]["phone"], "+79000000000")

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

    def test_unknown_session(self):
        self.assertEqual(self.client.post("/api/sessions/none/messages", json={"text": "x"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
