"""Маршрутизатор перед ответом: запасные пути и узел передачи менеджеру. Запуск: python -m unittest discover -s tests -v"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "core"), str(ROOT / "tests")]

import config  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402
import prompts  # noqa: E402
import router  # noqa: E402

HISTORY = [{"role": "user", "content": "Позовите менеджера"}]


class FakeIndex:
    def search(self, query, past=()):
        return [{"file": "a.md", "crumb": "A > B", "text": "Порог 750 кВт.", "score": 1.0}]

    def context(self, hits):
        return hits


def status_error(code):
    request = httpx.Request("POST", "http://x/v1/chat/completions")
    return httpx.HTTPStatusError("bad", request=request, response=httpx.Response(code, request=request, text="response_format"))


class ChatJsonTest(unittest.TestCase):
    def test_lenient_parse(self):
        self.assertEqual(llm._lenient_json('Вот: ```json\n{"route": "operator", "reason": "x"}\n```'),
                         {"route": "operator", "reason": "x"})
        self.assertIsNone(llm._lenient_json("без json"))

    def test_schema_rejected_falls_back_to_json_object(self):
        sent = []

        def fake_post(path, body, **kwargs):
            sent.append(body["response_format"]["type"])
            if body["response_format"]["type"] == "json_schema":
                raise status_error(400)
            return {"choices": [{"message": {"content": '{"route": "question", "reason": "r"}'}}]}

        with mock.patch.object(llm, "_post", fake_post):
            self.assertEqual(llm.chat_json([], {}), {"route": "question", "reason": "r"})
        self.assertEqual(sent, ["json_schema", "json_object"])

    def test_both_formats_rejected_gives_none(self):
        with mock.patch.object(llm, "_post", side_effect=status_error(422)):
            self.assertIsNone(llm.chat_json([], {}))


class ClassifyTest(unittest.TestCase):
    def check(self, returned, route, ok):
        with mock.patch.object(llm, "chat_json", **returned):
            result = router.classify(HISTORY)
        self.assertEqual((result["route"], result["ok"]), (route, ok))

    def test_valid_operator(self):
        self.check({"return_value": {"route": "operator", "reason": "просит менеджера"}}, "operator", True)

    def test_unknown_route_is_question(self):
        self.check({"return_value": {"route": "manager", "reason": "?"}}, "question", False)

    def test_no_json_is_question(self):
        self.check({"return_value": None}, "question", False)

    def test_timeout_is_question(self):
        self.check({"side_effect": httpx.ReadTimeout("slow")}, "question", False)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.saved = config.ROUTER

    def tearDown(self):
        config.ROUTER = self.saved

    def run_answer(self, mode, **classify):
        config.ROUTER = mode
        calls = []

        def fake_chat(messages, max_tokens=None):
            calls.append(messages[0]["content"])
            return "ответ"

        with mock.patch.object(router, "classify", **classify), mock.patch.object(llm, "chat", fake_chat):
            return pipeline.answer(FakeIndex(), HISTORY), calls

    def test_parallel_router_does_not_change_the_answer(self):
        decision = {"route": "operator", "reason": "просит менеджера", "ok": True, "ms": 1}
        result, calls = self.run_answer("parallel", return_value=decision)
        self.assertEqual((result["text"], result["route"]), ("ответ", "operator"))
        self.assertEqual(calls, [prompts.SYSTEM])
        self.assertTrue(result["sources"])

    def test_router_failure_keeps_answer(self):
        decision = {"route": "question", "reason": "ReadTimeout", "ok": False, "ms": 15000}
        result, _ = self.run_answer("parallel", return_value=decision)
        self.assertEqual((result["text"], result["route"]), ("ответ", "question"))

    def test_router_off_is_single_call(self):
        result, calls = self.run_answer("off", side_effect=AssertionError("router must not run"))
        self.assertEqual((result["route"], len(calls)), ("off", 1))

    def test_judge_default_is_off(self):
        # Веб меняет config.ROUTER при импорте, поэтому проверяем значение в самом модуле.
        spec = importlib.util.spec_from_file_location("fresh_config", ROOT / "core" / "config.py")
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)
        self.assertEqual(fresh.ROUTER, "off")


if __name__ == "__main__":
    unittest.main()
