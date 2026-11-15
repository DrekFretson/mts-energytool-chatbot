"""Клиент OpenAI-совместимого прокси: чат, эмбеддинги, реранк. Адреса и ключи только из env."""

import concurrent.futures as cf
import json
import os
import random
import re
import threading
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
# Платформа может выдать реранкеру отдельный ключ (README примера организаторов); без него реранк молча отваливается.
RERANK_API_KEY = _env("RERANK_API_KEY", "RERANKER_API_KEY", default=API_KEY)
MODEL = _env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL", default="qwen/qwen3.8-27b")
EMB_MODEL = _env("EMB_MODEL", default="baai/bge-m3")
RERANKER_MODEL = _env("RERANKER_MODEL", default="qwen/qwen3-reranker-8b")
try:
    MAX_TOKENS = max(128, min(8192, int(_env("MAX_TOKENS", default="2048"))))
except ValueError:
    MAX_TOKENS = 2048
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


_budget = threading.local()  # срок текущей задачи потока (вопрос, запрос веба, сборка индекса)


class BudgetExceeded(TimeoutError):
    pass


def set_deadline(seconds):
    """Все вызовы прокси в этом потоке укладываются в seconds от текущего момента; None снимает срок."""
    _budget.deadline = None if seconds is None else time.monotonic() + seconds


def _remaining():
    deadline = getattr(_budget, "deadline", None)
    return None if deadline is None else deadline - time.monotonic()


def _backoff(attempt, exc):
    retry_after = isinstance(exc, httpx.HTTPStatusError) and exc.response.headers.get("retry-after", "")
    if retry_after and retry_after.replace(".", "", 1).isdigit():
        return min(float(retry_after), 10.0)
    # Случайный разброс: иначе параллельные потоки бьют в лимит прокси одновременно.
    return 2**attempt * (0.5 + random.random())


wasted = []  # секунды неудачных попыток: по ним защита по времени отличает больной прокси от медленного


def _post(path, body, timeout=None, attempts=3, base_url=None, api_key=None, retry_timeout=True):
    base_url = base_url or BASE_URL
    if not base_url:
        raise RuntimeError("Не задан LLM_BASE_URL")
    for attempt in range(attempts):
        timeout_now = timeout or config.LLM_TIMEOUT
        left = _remaining()
        if left is not None:
            if left <= 0.5:
                raise BudgetExceeded(f"срок задачи истёк до запроса {path}")
            timeout_now = min(timeout_now, left)
        sent = time.monotonic()
        try:
            response = _client.post(
                base_url + path,
                headers={"Authorization": f"Bearer {api_key or API_KEY}"},
                json=body,
                timeout=timeout_now,
            )
            if response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()
                return response.json()
            response.raise_for_status()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            wasted.append(time.monotonic() - sent)
            retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                exc.response.status_code == 429 or exc.response.status_code >= 500
            )
            if isinstance(exc, httpx.TimeoutException) and not retry_timeout:
                retryable = False
            pause = _backoff(attempt, exc)
            left = _remaining()
            if not retryable or attempt == attempts - 1 or (left is not None and left < pause + 1):
                raise
            time.sleep(pause)


def chat_body(messages, max_tokens=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": config.TEMPERATURE,
        "max_tokens": max_tokens or config.MAX_ANSWER_TOKENS,
        **_reasoning,
    }
    if config.TOP_P is not None and "top_p" not in _rejected:
        body["top_p"] = config.TOP_P
    if not _reasons(body):
        _greedy(body)
    if "openrouter.ai" in BASE_URL and LLM_PROVIDER:
        body["provider"] = {"order": LLM_PROVIDER.split(","), "allow_fallbacks": False}
    return body


_limits = {}  # потолок max_tokens, который принял прокси (узнаём по отказу)
_OFF = {"reasoning": {"effort": "none"}, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
_rejected = set()  # необязательные поля выборки, которые прокси отверг: больше не отправляем


def _greedy(body):
    """Выборка задаётся для режима рассуждений; без рассуждений ответ детерминированный, как на всех замерах."""
    body["temperature"] = 0.0
    body.pop("top_p", None)


def _reasons(body):
    """В запросе рассуждения не выключены."""
    return any(body.get(key) != value for key, value in _OFF.items() if key in body)


def _post_chat(body):
    wanted = body["max_tokens"]
    limit = _limits.get("max_tokens")  # другой поток может убрать запись между проверкой и чтением
    if limit:
        # Потолок мог появиться из-за одного длинного промпта, а не правила прокси: каждый запрос пробует вдвое больше.
        body["max_tokens"] = min(wanted, limit * 2)
    while True:
        try:
            # Повторы ограничены сроком вопроса (set_deadline), а не числом попыток.
            result = _post("/chat/completions", body, attempts=3, retry_timeout=not _reasons(body))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (400, 422) and "max_tokens" in exc.response.text:
                if body["max_tokens"] <= 64:
                    raise
                # Прокси ограничивает длину ответа: уменьшаем лимит, а не теряем поля рассуждений.
                body["max_tokens"] //= 2
                continue
            sent = [key for key in ("top_p", *_OFF) if key in body]
            if exc.response.status_code not in (400, 422) or not sent:
                raise
            named = [key for key in sent if key in exc.response.text]
            for key in named or sent[:1]:
                # Запоминаем только поле, которое прокси назвал: безымянная ошибка одного запроса
                # (например, переполнение контекста) не должна менять остальные вопросы прогона.
                if named and key == "top_p":
                    _rejected.add(key)
                elif named:
                    _reasoning.pop(key, None)
                body.pop(key)
            continue
        if body["max_tokens"] < wanted:
            _limits["max_tokens"] = body["max_tokens"]
        else:
            _limits.pop("max_tokens", None)
        return result


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
    try:
        text, cut = _answer(_post_chat(body))
        if (not text or cut) and body["max_tokens"] < MAX_TOKENS and _limits.get("max_tokens") != body["max_tokens"]:
            # Рассуждения или длина съели лимит: повторяем один раз с запасом (если прокси сам не держит нас на этом лимите).
            body["max_tokens"] = MAX_TOKENS
            text, cut = _answer(_post_chat(body))
    except httpx.TimeoutException:
        if not _reasons(body):
            raise
        # Вызов с рассуждениями не уложился в тайм-аут: отвечаем без них, а не ждём его снова.
        text, cut = "", True
    if (not text or cut) and _reasons(body):
        # Рассуждения не уложились в лимит: последний шанс - тот же запрос без рассуждений, а не заглушка.
        body.update({key: _OFF[key] for key in body if key in _OFF})
        _greedy(body)
        body["max_tokens"] = config.MAX_ANSWER_TOKENS_OFF
        text, cut = _answer(_post_chat(body))
    if not text or cut:
        # Обрезанный текст - это рассуждение без тегов или половина фразы, а не ответ.
        raise ValueError("LLM вернула пустой или обрезанный ответ")
    return text


def _stream(body, stop=None):
    """Куски ответа из потока прокси: ('thinking', '') на первом куске рассуждений, затем ('text', кусок).

    stop - функция без аргументов: вернула True - читать дальше не нужно (клиент ушёл), поток закрывается.
    """
    left = _remaining()
    if left is not None and left <= 0.5:
        raise BudgetExceeded("срок задачи истёк до запроса /chat/completions")
    timeout = config.LLM_TIMEOUT if left is None else min(config.LLM_TIMEOUT, left)
    thinking = ended = False
    # Без парсера рассуждений прокси отдаёт их в content до "</think>": такой текст придерживаем, пока не станет ясно, где ответ.
    separate, held = not _reasons(body), ""
    with _client.stream("POST", BASE_URL + "/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"},
                        json=body, timeout=timeout) as response:
        response.raise_for_status()
        if "text/event-stream" not in response.headers.get("content-type", ""):
            # Прокси не умеет stream и ответил целиком: берём этот ответ, а не генерируем второй раз.
            response.read()
            text, cut = _answer(response.json())
            if cut:
                raise ValueError("LLM вернула обрезанный ответ")
            if text:
                yield "text", text
            return
        for line in response.iter_lines():
            if stop and stop():
                return
            # Тайм-аут httpx действует на каждое чтение: медленно капающий поток иначе пережил бы срок ответа.
            if (_remaining() or 1) <= 0:
                raise BudgetExceeded("срок задачи истёк во время потока")
            if not line.startswith("data:"):
                continue
            if line[5:].strip() == "[DONE]":
                ended = True
                continue
            chunk = json.loads(line[5:])
            if chunk.get("error"):
                raise ValueError("прокси прервал поток ошибкой")
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta") or {}
            if delta.get("reasoning_content") or delta.get("reasoning"):
                separate = True
                if not thinking:
                    thinking = True
                    yield "thinking", ""
            piece = delta.get("content") or ""
            if piece and separate:
                yield "text", piece
            elif piece:
                held += piece
                if "</think>" in held:
                    separate, piece = True, held.split("</think>", 1)[1].lstrip()
                    if piece:
                        yield "text", piece
            if choices[0].get("finish_reason") == "length":
                raise ValueError("LLM вернула обрезанный ответ")
            ended = ended or bool(choices[0].get("finish_reason"))
    if not ended:
        # Соединение закрылось без признака конца: текст оборван на полуслове, полным ответом его считать нельзя.
        raise ValueError("поток ответа оборвался")
    if held and "</think>" not in held and THINK.sub("", held).strip():
        yield "text", THINK.sub("", held).strip()


def chat_stream(messages, max_tokens=None, stop=None):
    """Ответ по мере генерации: ('thinking', '') пока модель рассуждает, затем ('text', кусок).

    Сбой до первого куска текста (прокси без stream, обрыв, пустой или обрезанный ответ) закрывает обычный chat()
    со всеми его повторами; после первого куска ошибка уходит вызывающему.
    """
    started = False
    try:
        for kind, piece in _stream({**chat_body(messages, max_tokens), "stream": True}, stop):
            started = started or kind == "text"
            yield kind, piece
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError):
        if started:
            raise
    if not started and not (stop and stop()):
        yield "text", chat(messages, max_tokens)


def reasoning_mode():
    """Запас токенов и времени под рассуждения: без него ответы обрезаются и уходят в запасной текст."""
    global MAX_TOKENS
    config.MAX_ANSWER_TOKENS = MAX_TOKENS = max(MAX_TOKENS, config.REASONING_MAX_TOKENS)
    config.LLM_TIMEOUT = config.REASONING_TIMEOUT


def reasoning_off():
    """Выключить рассуждения для оставшихся вопросов: страховка по времени прогона."""
    for key in list(_reasoning):
        _reasoning[key] = _OFF[key]


def probe():
    """Короткий запрос на старте: True, если модель рассуждает. Заодно отсеивает поля, которые прокси отвергает."""
    body = chat_body([{"role": "user", "content": "Ответь одним словом: да"}], max_tokens=64)
    return thinks(_post_chat(body))


embed_parallel = 1


def embed(texts, batch_size=32):
    """Пачки уходят по embed_parallel одновременно; состав пачек и порядок векторов от этого не зависят."""
    deadline = getattr(_budget, "deadline", None)  # срок задаёт set_deadline вызывающего потока

    def one(batch):
        _budget.deadline = deadline
        data = _post("/embeddings", {"model": EMB_MODEL, "input": batch}, timeout=config.EMB_TIMEOUT,
                     base_url=EMB_BASE_URL, api_key=EMB_API_KEY)["data"]
        if len(data) != len(batch):
            raise ValueError("Не все эмбеддинги получены")
        return [item["embedding"] for _, item in sorted(enumerate(data), key=lambda pair: pair[1].get("index", pair[0]))]

    batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
    if embed_parallel > 1 and len(batches) > 1:
        with cf.ThreadPoolExecutor(min(embed_parallel, len(batches))) as pool:
            vectors = [vector for part in pool.map(one, batches) for vector in part]
    else:
        vectors = [vector for batch in batches for vector in one(batch)]
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


_RERANK_SHAPES = ((None, None), (None, 30), (20, 20))  # (сколько документов, top_n): следующая форма после отказа 4xx


def rerank(query, documents):
    """Индексы documents от лучшего к худшему; при урезанном запросе - только тех, что вернул реранкер.

    Пример организаторов шлёт до 30 документов с top_n до 30. Если прокси отвергает наши 40, пробуем top_n=30,
    затем 20 первых документов: иначе реранк молча выключен на весь прогон. Принятая форма не запоминается:
    отказ мог вызвать один длинный запрос, а повторный отказ 4xx стоит доли секунды.
    """
    for count, top in _RERANK_SHAPES:
        sent = documents[:count] if count else documents
        try:
            result = _post(
                "/rerank",
                {"model": RERANKER_MODEL, "query": query, "documents": sent, "top_n": min(top or len(sent), len(sent))},
                timeout=config.RERANK_TIMEOUT,
                attempts=2,
                api_key=RERANK_API_KEY,
                retry_timeout=False,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (400, 413, 422) and (count, top) != _RERANK_SHAPES[-1]:
                continue
            raise
        items = result.get("results", result.get("data", []))
        scores = [item.get("relevance_score", item.get("score")) for item in items]
        if items and all(isinstance(score, (int, float)) for score in scores):
            # Порядок задаёт оценка: прокси может вернуть документы в исходном порядке. Сортировка устойчивая.
            items = [item for _, item in sorted(zip(scores, items), key=lambda pair: -pair[0])]
        return [int(item["index"]) for item in items]


def _lenient_json(text):
    """Первый JSON-объект в тексте: модели иногда оборачивают его в ``` или пишут фразу до него."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                try:
                    return json.loads(text[start : end + 1])
                except ValueError:
                    break
        start = text.find("{", start + 1)
    return None


def chat_json(messages, schema, max_tokens=200, timeout=None):
    """Структурный ответ: json_schema, при отказе прокси json_object, затем мягкий разбор. None, если JSON не получен."""
    formats = [{"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}}, {"type": "json_object"}]
    for response_format in formats:
        body = {**chat_body(messages, max_tokens), "response_format": response_format}
        # Только поля, которые прокси уже принял (лишние _post_chat отсеял на пробе): классификатору рассуждения не нужны.
        body.update({key: _OFF[key] for key in list(body) if key in _OFF})
        _greedy(body)
        try:
            result = _post("/chat/completions", body, timeout=timeout, attempts=1)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (400, 422):
                continue
            raise
        text, _ = _answer(result)
        try:
            return json.loads(text)
        except ValueError:
            return _lenient_json(text)
    return None
