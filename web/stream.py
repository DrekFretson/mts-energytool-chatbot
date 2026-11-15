"""Поток ответа для веб-чата поверх ядра, без правок в core/.

Ядро вызывает llm.chat один раз и ждёт весь ответ. Здесь llm.chat подменён диспетчером: если поток исполнения
объявил приёмник событий, тот же запрос уходит с "stream": true и текст отдаётся по мере генерации; без приёмника
(резюме заявки, сабмит) работает исходный llm.chat. Модуль импортирует только веб, в архив сабмита он не входит.
"""

import json
import threading
import time

import httpx

import config
import llm
import pipeline

_sink = threading.local()
_chat = llm.chat


def _reasons(body):
    """В запросе рассуждения не выключены."""
    return any(body.get(key) != value for key, value in llm._OFF.items() if key in body)


def _stream(body, stop):
    """Куски ответа из потока прокси: ('thinking', '') на первом куске рассуждений, затем ('text', кусок).

    stop - функция без аргументов: вернула True - читать дальше не нужно (клиент ушёл), поток закрывается.
    """
    left = llm._remaining()
    if left is not None and left <= 0.5:
        raise llm.BudgetExceeded("срок задачи истёк до запроса /chat/completions")
    timeout = config.LLM_TIMEOUT if left is None else min(config.LLM_TIMEOUT, left)
    thinking = ended = False
    # Без парсера рассуждений прокси отдаёт их в content до "</think>": такой текст придерживаем, пока не станет ясно, где ответ.
    separate, held = not _reasons(body), ""
    with llm._client.stream("POST", llm.BASE_URL + "/chat/completions", headers={"Authorization": f"Bearer {llm.API_KEY}"},
                            json=body, timeout=timeout) as response:
        response.raise_for_status()
        if "text/event-stream" not in response.headers.get("content-type", ""):
            # Прокси не умеет stream и ответил целиком: берём этот ответ, а не генерируем второй раз.
            response.read()
            text, cut = llm._answer(response.json())
            if cut:
                raise ValueError("LLM вернула обрезанный ответ")
            if text:
                yield "text", text
            return
        for line in response.iter_lines():
            if stop():
                return
            # Тайм-аут httpx действует на каждое чтение: медленно капающий поток иначе пережил бы срок ответа.
            if (llm._remaining() or 1) <= 0:
                raise llm.BudgetExceeded("срок задачи истёк во время потока")
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
    if held and "</think>" not in held and llm.THINK.sub("", held).strip():
        yield "text", llm.THINK.sub("", held).strip()


def _dispatch(messages, max_tokens=None):
    emit = getattr(_sink, "emit", None)
    if emit is None:
        return _chat(messages, max_tokens)
    emit("searched", None)
    pieces = []
    try:
        for kind, piece in _stream({**llm.chat_body(messages, max_tokens), "stream": True}, _sink.stop):
            if kind == "text":
                pieces.append(piece)
            emit(kind, piece or None)
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError):
        if pieces:
            raise
    if pieces or _sink.stop():
        return "".join(pieces)
    # Сбой до первого куска текста (прокси без stream, обрыв, пустой или обрезанный ответ): обычный вызов со всеми его повторами.
    text = _chat(messages, max_tokens)
    emit("text", text)
    return text


llm.chat = _dispatch


def answer(index, history, note, emit, stop):
    """pipeline.answer с событиями по ходу: emit('searched' | 'thinking', None), затем emit('text', кусок).

    Возвращает обычный результат pipeline.answer: текст в нём очищен целиком, им заменяют показанные куски.
    stop - функция без аргументов: вернула True - клиент ушёл, генерация останавливается.
    """
    started, first = time.perf_counter(), []

    def timed(kind, value):
        if kind == "text" and not first:
            first.append(time.perf_counter())
        emit(kind, value)

    _sink.emit, _sink.stop = timed, stop
    try:
        result = pipeline.answer(index, history, note)
    finally:
        _sink.emit = None
    result["timings"]["first_token_ms"] = round(((first[0] if first else time.perf_counter()) - started) * 1000)
    return result
