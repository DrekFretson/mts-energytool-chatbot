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
MODEL = _env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL", default="qwen/qwen3.8-27b")
EMB_MODEL = _env("EMB_MODEL", default="baai/bge-m3")
RERANKER_MODEL = _env("RERANKER_MODEL", default="qwen/qwen3-reranker-8b")
MAX_TOKENS = int(_env("MAX_TOKENS", default="2048"))

THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
_client = httpx.Client(timeout=config.LLM_TIMEOUT)
_thinking_flag_accepted = True


def _post(path, body, timeout=None, attempts=3):
    if not BASE_URL:
        raise RuntimeError("Не задан LLM_BASE_URL")
    for attempt in range(attempts):
        try:
            response = _client.post(
                BASE_URL + path,
                headers={"Authorization": f"Bearer {API_KEY}"},
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


def chat(messages, max_tokens=None):
    global _thinking_flag_accepted
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": config.TEMPERATURE,
        "max_tokens": max_tokens or config.MAX_ANSWER_TOKENS,
    }
    if config.THINKING_OFF and _thinking_flag_accepted:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        result = _post("/chat/completions", body)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 400 or "chat_template_kwargs" not in body:
            raise
        _thinking_flag_accepted = False
        del body["chat_template_kwargs"]
        result = _post("/chat/completions", body)

    text = THINK.sub("", result["choices"][0]["message"].get("content") or "").strip()
    if not text and body["max_tokens"] < MAX_TOKENS:
        # Рассуждения съели весь лимит: повторяем один раз с запасом.
        body["max_tokens"] = MAX_TOKENS
        result = _post("/chat/completions", body)
        text = THINK.sub("", result["choices"][0]["message"].get("content") or "").strip()
    if not text:
        raise ValueError("LLM вернула пустой ответ")
    return text


def embed(texts, batch_size=32):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        data = _post("/embeddings", {"model": EMB_MODEL, "input": batch})["data"]
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
