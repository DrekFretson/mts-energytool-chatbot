"""Управление рассуждениями на разных прокси. Запуск: python -m unittest discover -s tests -v"""

import json
import sys
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

THOUGHT = "Okay, the user asks about panels. Let me check the materials. " * 3


def reply(content, finish="stop", status=200, **message):
    return status, {"choices": [{"message": {"content": content, **message}, "finish_reason": finish}]}


class Reasoning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Импорт здесь, а не в шапке: llm читает адрес прокси при импорте, а его задаёт test_mvp.
        global llm, FIELDS
        import llm

        FIELDS = dict(llm._reasoning)

    def setUp(self):
        self.script, self.seen = [], []
        self.client, self.base = llm._client, llm.BASE_URL
        llm._client = httpx.Client(transport=httpx.MockTransport(self.handle))
        llm.BASE_URL = "http://proxy.test/v1"

    def tearDown(self):
        llm._client, llm.BASE_URL = self.client, self.base
        llm._reasoning.clear()
        llm._reasoning.update(FIELDS)

    def handle(self, request):
        self.seen.append(json.loads(request.content))
        self.auth = request.headers.get("authorization")
        status, payload = self.script.pop(0)
        if status == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(status, json=payload)

    def ask(self):
        return llm.chat([{"role": "user", "content": "вопрос"}])

    def test_sends_every_switch(self):
        self.script.append(reply("Да."))
        self.assertEqual(self.ask(), "Да.")
        self.assertEqual({key: self.seen[0][key] for key in FIELDS}, FIELDS)

    def test_rejected_field_is_dropped_and_remembered(self):
        self.script += [(400, {"error": "Unrecognized request argument: reasoning_effort"}), reply("Да."), reply("Нет.")]
        self.assertEqual(self.ask(), "Да.")
        self.ask()
        for body in self.seen[1:]:
            self.assertNotIn("reasoning_effort", body)
            self.assertIn("chat_template_kwargs", body)

    def test_error_without_field_name_drops_fields_for_that_request_only(self):
        self.script += [(422, {"detail": "bad"}), (422, {"detail": "bad"}), reply("Да."), reply("Нет.")]
        self.assertEqual(self.ask(), "Да.")
        self.assertEqual([sum(key in body for key in FIELDS) for body in self.seen], [3, 2, 1])
        self.ask()
        self.assertEqual({key: self.seen[-1][key] for key in FIELDS}, FIELDS)

    def test_reasoning_never_returned_as_answer(self):
        self.script.append(reply(THOUGHT + "</think>\n\nДа, можно."))
        self.assertEqual(self.ask(), "Да, можно.")
        self.script += [reply(THOUGHT, finish="length")] * 3
        with self.assertRaises(ValueError):
            self.ask()
        self.assertEqual(self.seen[-1]["reasoning"], {"effort": "none"})

    def test_reasoning_timeout_is_answered_without_reasoning(self):
        self.script += [("timeout", None), reply("Да.")]
        self.assertEqual(self.ask(), "Да.")
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(self.seen[1]["reasoning"], {"effort": "none"})

    def test_sampling_goes_only_with_reasoning_and_survives_rejection(self):
        config = llm.config
        saved = config.TEMPERATURE, config.TOP_P
        config.TEMPERATURE, config.TOP_P = 1.0, 0.95
        try:
            self.script += [(400, {"error": "Unrecognized request argument: top_p"}), reply("Да."), ("timeout", None), reply("Да.")]
            self.ask()
            self.ask()
        finally:
            config.TEMPERATURE, config.TOP_P = saved
            llm._rejected.clear()
        first, retry, second, plain = self.seen
        self.assertEqual((first["temperature"], first["top_p"]), (1.0, 0.95))
        self.assertNotIn("top_p", retry)
        self.assertEqual({key: retry[key] for key in FIELDS}, FIELDS)
        self.assertNotIn("top_p", second)
        self.assertEqual(second["temperature"], 1.0)
        self.assertEqual(plain["temperature"], 0.0)

    def test_reranker_gets_its_own_key(self):
        saved, llm.RERANK_API_KEY = llm.RERANK_API_KEY, "rerank-key"
        try:
            self.script.append((200, {"results": [{"index": 1}, {"index": 0}]}))
            self.assertEqual(llm.rerank("вопрос", ["а", "б"]), [1, 0])
        finally:
            llm.RERANK_API_KEY = saved
        self.assertEqual(self.auth, "Bearer rerank-key")

    def test_reasoning_mode_raises_limits(self):
        config = llm.config
        saved = config.MAX_ANSWER_TOKENS, config.LLM_TIMEOUT, llm.MAX_TOKENS
        try:
            llm.reasoning_mode()
            self.assertGreaterEqual(config.MAX_ANSWER_TOKENS, config.REASONING_MAX_TOKENS)
            self.assertEqual(llm.MAX_TOKENS, config.MAX_ANSWER_TOKENS)
            self.assertEqual(config.LLM_TIMEOUT, config.REASONING_TIMEOUT)
        finally:
            config.MAX_ANSWER_TOKENS, config.LLM_TIMEOUT, llm.MAX_TOKENS = saved

    def test_cut_answer_is_retried_with_full_budget(self):
        self.script += [reply("", finish="length", reasoning=THOUGHT), reply("Да.", reasoning=THOUGHT)]
        self.assertEqual(self.ask(), "Да.")
        self.assertEqual(self.seen[1]["max_tokens"], llm.MAX_TOKENS)

    def test_probe(self):
        for payload, expected in (
            (reply("да", reasoning="hmm"), True),
            (reply("", finish="length"), True),
            (reply("да"), False),
        ):
            self.script.append(payload)
            self.assertEqual(llm.probe(), expected)


if __name__ == "__main__":
    unittest.main()
