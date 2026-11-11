"""Клиент OpenAI-совместимого прокси: чат, эмбеддинги, реранк. Адреса и ключи только из env."""

import os
import re
import time

import httpx
import numpy as np

import config


def _env(*names, default=""):
    for name in names:
        value = os.getenv(name, "").strip()
        # Пример организаторов считает значение, равное имени переменной, незаполненным.
        if value and value != name:
            return value
    return default


BASE_URL = _env("LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL").rstrip("/")
API_KEY = _env("LLM_API_KEY", "OPENAI_API_KEY", default="not-needed")
# На платформе один прокси на всё; на стенде генерация по API, а bge-m3 локально.
EMB_BASE_URL = _env("EMB_BASE_URL", default=BASE_URL).rstrip("/")
EMB_API_KEY = _env("EMB_API_KEY", "EMBEDDER_API_KEY", default=API_KEY)
MODEL = _env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL", default="qwen/qwen3.8-27b")
EMB_MODEL = _env("EMB_MODEL", default="baai/bge-m3")
RERANKER_MODEL = _env("RERANKER_MODEL", default="qwen/qwen3-reranker-8b")
MAX_TOKENS = int(_env("MAX_TOKENS", default="2048"))
# Только для стенда на OpenRouter: провайдеры отличаются квантизацией, фиксируем один, например deepinfra/bf16.
LLM_PROVIDER = _env("LLM_PROVIDER")

THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
# Шаблоны Qwen3.5+ кладут "<think>" в промпт: без парсера ответ приходит как "рассуждение</think>ответ".
THINK_TAIL = re.compile(r"^.*?</think>", re.S)
_client = httpx.Client(timeout=config.LLM_TIMEOUT)
# Прокси управляют рассуждениями разными полями (OpenRouter, OpenAI, шаблон Qwen); отвергнутое поле убираем.
_reasoning = {
    "reasoning": {"effort": config.REASONING_EFFORT},
    "reasoning_effort": config.REASONING_EFFORT,
    "chat_template_kwargs": {"enable_thinking": config.REASONING_EFFORT != "none"},
}


def _post(path, body, timeout=None, attempts=3, base_url=None, api_key=None):
    base_url = base_url or BASE_URL
    if not base_url:
        raise RuntimeError("Не задан LLM_BASE_URL")
    for attempt in range(attempts):
        try:
            response = _client.post(
                base_url + path,
                headers={"Authorization": f"Bearer {api_key or API_KEY}"},
                json=body,
                timeout=timeout or config.LLM_TIMEOUT,
            )
            if response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()
                return response.json()
            response.raise_for_status()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                exc.response.status_code == 429 or exc.response.status_code >= 500
            )
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(2**attempt)


def chat_body(messages, max_tokens=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": config.TEMPERATURE,
        "max_tokens": max_tokens or config.MAX_ANSWER_TOKENS,
        **_reasoning,
    }
    if "openrouter.ai" in BASE_URL and LLM_PROVIDER:
        body["provider"] = {"order": LLM_PROVIDER.split(","), "allow_fallbacks": False}
    return body


_limits = {}  # потолок max_tokens, который принял прокси (узнаём по отказу)
_OFF = {"reasoning": {"effort": "none"}, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}


def _post_chat(body):
    while True:
        if "max_tokens" in _limits:
            body["max_tokens"] = min(body["max_tokens"], _limits["max_tokens"])
        try:
            # Один повтор: зависший на рассуждениях запрос иначе держит вопрос три тайм-аута подряд.
            return _post("/chat/completions", body, attempts=2)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (400, 422) and "max_tokens" in exc.response.text and body["max_tokens"] > 512:
                # Прокси ограничивает длину ответа: уменьшаем лимит и запоминаем, а не теряем поля рассуждений.
                _limits["max_tokens"] = body["max_tokens"] // 2
                continue
            sent = [key for key in _reasoning if key in body]
            if exc.response.status_code not in (400, 422) or not sent:
                raise
            named = [key for key in sent if key in exc.response.text]
            for key in named or sent[:1]:
                _reasoning.pop(key, None)
                body.pop(key)


def _answer(result):
    choice = result["choices"][0]
    content = choice["message"].get("content") or ""
    return THINK_TAIL.sub("", THINK.sub("", content)).strip(), choice.get("finish_reason") == "length"


def thinks(result):
    """Признаки рассуждений в ответе прокси: отдельное поле, счётчик токенов или расход много больше видимого текста."""
    message, usage = result["choices"][0]["message"], result.get("usage") or {}
    content = message.get("content") or ""
    details = usage.get("completion_tokens_details") or {}
    return bool(
        not content.strip()
        or message.get("reasoning")
        or message.get("reasoning_content")
        or details.get("reasoning_tokens")
        or "</think>" in content
        or usage.get("completion_tokens", 0) > 2 * (len(content) / 3 + 20)
    )


def chat(messages, max_tokens=None):
    body = chat_body(messages, max_tokens)
    text, cut = _answer(_post_chat(body))
    if (not text or cut) and body["max_tokens"] < MAX_TOKENS:
        # Рассуждения или длина съели лимит: повторяем один раз с запасом.
        body["max_tokens"] = MAX_TOKENS
        text, cut = _answer(_post_chat(body))
    if (not text or cut) and any(body.get(key) != value for key, value in _OFF.items() if key in body):
        # Рассуждения не уложились в лимит: последний шанс - тот же запрос без рассуждений, а не заглушка.
        body.update({key: _OFF[key] for key in body if key in _OFF})
        body["max_tokens"] = config.MAX_ANSWER_TOKENS_OFF
        text, cut = _answer(_post_chat(body))
    if not text or cut:
        # Обрезанный текст - это рассуждение без тегов или половина фразы, а не ответ.
        raise ValueError("LLM вернула пустой или обрезанный ответ")
    return text


def reasoning_off():
    """Выключить рассуждения для оставшихся вопросов: страховка по времени прогона."""
    for key in _reasoning:
        _reasoning[key] = _OFF[key]


def probe():
    """Короткий запрос на старте: True, если модель рассуждает. Заодно отсеивает поля, которые прокси отвергает."""
    body = chat_body([{"role": "user", "content": "Ответь одним словом: да"}], max_tokens=64)
    return thinks(_post_chat(body))


def embed(texts, batch_size=32):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        data = _post("/embeddings", {"model": EMB_MODEL, "input": batch}, base_url=EMB_BASE_URL, api_key=EMB_API_KEY)["data"]
        ordered = sorted(enumerate(data), key=lambda pair: pair[1].get("index", pair[0]))
        vectors.extend(item["embedding"] for _, item in ordered)
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def rerank(query, documents):
    """Индексы documents от лучшего к худшему."""
    result = _post(
        "/rerank",
        {"model": RERANKER_MODEL, "query": query, "documents": documents, "top_n": len(documents)},
        timeout=config.RERANK_TIMEOUT,
        attempts=1,
    )
    return [int(item["index"]) for item in result.get("results", result.get("data", []))]
