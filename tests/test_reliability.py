"""Отказоустойчивость прогона: разбор questions.csv, срок вопроса, повторы, второй проход, пауза эмбеддингов.
Запуск: python -m unittest discover -s tests -v
"""

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))


def load():
    # Импорт при запуске: test_mvp раньше задаёт адрес своей заглушки.
    global llm, retrieval, solution
    import llm  # noqa: F401
    import retrieval  # noqa: F401
    import solution  # noqa: F401


class Questions(unittest.TestCase):
    def setUp(self):
        load()

    def read(self, text):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "q.csv")
            path.write_text(text, encoding="utf-8")
            return solution.read_questions(path)

    def test_malformed_rows(self):
        huge = "x" * 300_000
        got = self.read(f'﻿id,question\n1,"{huge}"\n2,\n,без id\n3\n 4 ,норм\n')
        self.assertEqual(sorted(got), ["1", "2", "3", "4"])
        self.assertEqual(len(got["1"]), 300_000)
        self.assertEqual(got["2"], "")
        self.assertEqual(got["3"], "")

    def test_other_column_names_and_empty_file(self):
        self.assertEqual(self.read("qid,text\n7,вопрос\n"), {"7": "вопрос"})
        self.assertEqual(self.read(""), {})

    def test_schedule_projection_uses_throughput(self):
        until = solution.config.REASONING_UNTIL_SEC  # 840 s
        # 6 of 100 done 40 s after a 10 s start: 5 s per question -> 40 + 94*5 = 510 s, on time
        self.assertFalse(solution.behind_schedule(40, 10, 6, 100, 3))
        # the old formula (elapsed / done) said 40 + 94*6.7 = 670 s; with a 100 s start it said 2000+ s
        self.assertFalse(solution.behind_schedule(130, 100, 6, 100, 3))
        self.assertTrue(solution.behind_schedule(100, 10, 6, 600, 3))  # 15 s per question x 594 left
        self.assertFalse(solution.behind_schedule(100, 10, 5, 600, 3))  # too early to judge
        self.assertGreater(until, 0)

    def test_workers_garbage(self):
        for value, expected in (("abc", 3), ("0", 1), ("100", 8), ("", 3), ("5", 5)):
            os.environ["MAX_WORKERS"] = value
            try:
                self.assertEqual(solution.workers_from_env(), expected)
            finally:
                os.environ.pop("MAX_WORKERS", None)


class Budget(unittest.TestCase):
    def setUp(self):
        load()
        self.client, self.base, self.sleep = llm._client, llm.BASE_URL, llm.time.sleep
        self.pauses = []
        llm.time.sleep = self.pauses.append
        llm.BASE_URL = "http://proxy.test/v1"

    def tearDown(self):
        llm._client, llm.BASE_URL, llm.time.sleep = self.client, self.base, self.sleep
        llm.set_deadline(None)

    def mock(self, handler):
        llm._client = httpx.Client(transport=httpx.MockTransport(handler))

    def test_retry_after_is_honoured(self):
        calls = []

        def handler(request):
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "rate"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "Да."}, "finish_reason": "stop"}]})
        self.mock(handler)
        self.assertEqual(llm.chat([{"role": "user", "content": "q"}]), "Да.")
        self.assertEqual(self.pauses, [1.0])

    def test_deadline_stops_retries(self):
        self.mock(lambda request: httpx.Response(503, json={"error": "down"}))
        llm.set_deadline(2.0)  # backoff 1-3 s would not fit after the first failure
        started = time.monotonic()
        with self.assertRaises(httpx.HTTPStatusError):
            llm.chat([{"role": "user", "content": "q"}])
        self.assertLess(time.monotonic() - started, 1.0)
        llm.set_deadline(0)
        with self.assertRaises(llm.BudgetExceeded):
            llm.chat([{"role": "user", "content": "q"}])

    def test_low_max_tokens_cap_is_halved_not_fatal(self):
        seen = []

        def handler(request):
            body = json.loads(request.content)
            seen.append(body["max_tokens"])
            if body["max_tokens"] > 256:
                return httpx.Response(400, json={"error": "max_tokens must be <= 256"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "Да."}, "finish_reason": "stop"}]})
        self.mock(handler)
        llm._limits.clear()
        try:
            self.assertEqual(llm.chat([{"role": "user", "content": "q"}]), "Да.")
        finally:
            llm._limits.clear()
        self.assertEqual(seen, [400, 200])
        self.assertTrue(llm._reasoning)  # reasoning switches are kept

    def test_backoff_has_jitter(self):
        values = {round(llm._backoff(1, httpx.ConnectError("x")), 3) for _ in range(20)}
        self.assertGreater(len(values), 10)
        self.assertTrue(all(1.0 <= v <= 3.0 for v in values))


class DensePause(unittest.TestCase):
    def setUp(self):
        load()
        self.embed = llm.embed

    def tearDown(self):
        llm.embed = self.embed

    def test_failing_query_embeddings_pause_dense_search(self):
        chunks = [{"id": i, "section": i, "file": "a.md", "crumb": f"A > {i}", "text": f"текст {i}"} for i in range(5)]
        llm.embed = lambda texts, batch_size=32: np.eye(len(texts), 4, dtype=np.float32)
        index = retrieval.Index(chunks)
        calls = []

        def broken(texts, batch_size=32):
            calls.append(1)
            raise httpx.ConnectError("down")
        llm.embed = broken
        for _ in range(6):
            self.assertTrue(index.search("текст"))
        self.assertEqual(len(calls), retrieval.DENSE_PAUSE_AFTER)


class RerankPause(unittest.TestCase):
    def setUp(self):
        load()
        self.embed, self.rerank = llm.embed, llm.rerank

    def tearDown(self):
        llm.embed, llm.rerank = self.embed, self.rerank

    def test_failing_reranker_is_paused(self):
        chunks = [{"id": i, "section": i, "file": "a.md", "crumb": f"A > {i}", "text": f"текст {i}"} for i in range(5)]
        llm.embed = lambda texts, batch_size=32: np.eye(len(texts), 4, dtype=np.float32)
        index = retrieval.Index(chunks)
        calls = []

        def slow(query, documents):
            calls.append(1)
            raise httpx.ReadTimeout("slow")
        llm.rerank = slow
        saved, retrieval.config.USE_RERANK = retrieval.config.USE_RERANK, True
        try:
            for _ in range(6):
                self.assertTrue(index.search("текст"))
        finally:
            retrieval.config.USE_RERANK = saved
        self.assertEqual(len(calls), retrieval.DENSE_PAUSE_AFTER)


class FlakyChat(BaseHTTPRequestHandler):
    """Чат падает 500 на первые три запроса по вопросу с меткой, эмбеддинги работают."""
    failures = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return self.reply(200, {"data": [{"index": i, "embedding": [1.0, float(len(t) % 7)]} for i, t in enumerate(texts)]})
        if self.path.endswith("/rerank"):
            return self.reply(200, {"results": []})
        text = json.dumps(body["messages"], ensure_ascii=False)
        if "МЕТКА" in text:
            n = FlakyChat.failures.get("n", 0)
            FlakyChat.failures["n"] = n + 1
            if n < 3:
                return self.reply(500, {"error": "flaky"})
        self.reply(200, {"choices": [{"message": {"content": "Ответ."}, "finish_reason": "stop"}]})

    def reply(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class SecondPass(unittest.TestCase):
    def test_failed_question_is_retried_in_a_second_pass(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FlakyChat)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as work:
                data = Path(work, "data")
                data.mkdir()
                import zipfile
                with zipfile.ZipFile(data / "knowledge-base.zip", "w") as z:
                    z.writestr("a.md", "# Док\n\n## Раздел\n\nТекст раздела.\n")
                (data / "questions.csv").write_text("id,question\n1,обычный\n2,МЕТКА вопрос\n", encoding="utf-8")
                env = {**os.environ, "PYTHONPATH": str(ROOT / "core"), "DATA_DIR": "data",
                       "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1"}
                code = "import llm, solution; llm.time.sleep = lambda s: None; solution.main()"
                done = subprocess.run([sys.executable, "-c", code], cwd=work, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertIn("Повторный проход: 1", done.stderr)
                rows = list(csv.DictReader(io.StringIO(Path(work, "submission.csv").read_text(encoding="utf-8"))))
        finally:
            server.shutdown()
        self.assertEqual({r["id"]: r["answer"] for r in rows}, {"1": "Ответ.", "2": "Ответ."})


if __name__ == "__main__":
    unittest.main()


class WebGuards(unittest.TestCase):
    """Ограничения веба: размер запроса, одна обработка на сессию, лимит частоты, срок ответа, заявка один раз."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        sys.path.insert(0, str(ROOT))
        if "KB_PATH" not in os.environ:
            import zipfile
            kb_path = Path(tempfile.mkdtemp(), "kb.zip")
            with zipfile.ZipFile(kb_path, "w") as z:
                z.writestr("a.md", "# Док\n\n## Раздел\n\nТекст раздела.\n")
            os.environ["KB_PATH"] = str(kb_path)
        os.environ.setdefault("DB_PATH", str(Path(tempfile.mkdtemp(), "app.sqlite")))
        os.environ.setdefault("DEBUG_TOKEN", "secret")
        from web import app as web_app

        cls.web = web_app
        cls.client_context = TestClient(web_app.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def setUp(self):
        self.web._recent.clear()
        self.sid = self.client.post("/api/sessions").json()["id"]

    def post(self, text="вопрос"):
        return self.client.post(f"/api/sessions/{self.sid}/messages", json={"text": text})

    def test_body_too_large(self):
        r = self.client.post(f"/api/sessions/{self.sid}/messages", content=b'{"text": "' + b"x" * 100_000 + b'"}',
                             headers={"Content-Type": "application/json"})
        self.assertEqual(r.status_code, 413)

    def test_busy_session_rejected(self):
        self.web._busy_sessions.add(self.sid)
        try:
            self.assertEqual(self.post().status_code, 429)
        finally:
            self.web._busy_sessions.discard(self.sid)
        self.assertEqual(self.post().status_code, 200)

    def test_rate_limit_per_ip(self):
        saved, self.web.MESSAGES_PER_MINUTE = self.web.MESSAGES_PER_MINUTE, 2
        try:
            self.assertEqual([self.post().status_code for _ in range(3)], [200, 200, 429])
        finally:
            self.web.MESSAGES_PER_MINUTE = saved
            self.web._recent.clear()

    def test_slow_llm_gives_degraded_answer_within_budget(self):
        import llm
        hang = ThreadingHTTPServer(("127.0.0.1", 0), Hang)
        threading.Thread(target=hang.serve_forever, daemon=True).start()
        saved = llm.BASE_URL, llm.EMB_BASE_URL, self.web.ANSWER_BUDGET_SEC
        llm.BASE_URL = llm.EMB_BASE_URL = f"http://127.0.0.1:{hang.server_port}/v1"
        self.web.ANSWER_BUDGET_SEC = 1.5
        try:
            started = time.monotonic()
            r = self.client.post(f"/api/sessions/{self.sid}/messages", json={"text": "вопрос"}, headers={"X-Debug-Token": "secret"})
            elapsed = time.monotonic() - started
        finally:
            llm.BASE_URL, llm.EMB_BASE_URL, self.web.ANSWER_BUDGET_SEC = saved
            hang.shutdown()
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["debug"]["degraded"])
        self.assertLess(elapsed, 4)

    def test_concurrent_lead_submits_create_one_lead(self):
        lead = {"name": "Тест Тестов", "phone": "+79000000000", "company": "ООО", "consent": True}
        results = []

        def submit():
            r = self.client.post(f"/api/sessions/{self.sid}/lead", json=lead)
            results.append(r.json().get("status") if r.status_code == 200 else r.status_code)
        threads = [threading.Thread(target=submit) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(results.count("created"), 1, results)
        detail = self.client.get(f"/api/admin/sessions/{self.sid}", headers={"X-Debug-Token": "secret"}).json()
        self.assertIsNotNone(detail["lead"])


class Hang(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        time.sleep(10)
        self.send_response(500)
        self.end_headers()
