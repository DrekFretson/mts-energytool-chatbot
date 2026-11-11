"""Классификатор реплики: operator (клиент просит человека или заявку) или question (обычный ответ). Только LLM, без ключевых слов."""

import time

import config
import llm
import prompts

ROUTES = ("operator", "question")


def _transcript(history, turns=6):
    lines = [f"{'Клиент' if t['role'] == 'user' else 'Бот'}: {t['content']}" for t in history[-turns:]]
    return "Диалог:\n" + "\n".join(lines) + "\n\nМаршрут последней реплики клиента?"


def classify(history):
    """{route, reason, ms, ok}. Любой сбой, тайм-аут или неверный JSON -> question: обычный ответ безопаснее передачи менеджеру."""
    started = time.perf_counter()
    messages = [{"role": "system", "content": prompts.ROUTER}, {"role": "user", "content": _transcript(history)}]
    try:
        data = llm.chat_json(messages, prompts.ROUTER_SCHEMA, config.ROUTER_MAX_TOKENS, timeout=config.ROUTER_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - маршрутизатор не должен ронять ответ
        data, error = None, type(exc).__name__
    else:
        error = ""
    ok = isinstance(data, dict) and data.get("route") in ROUTES
    return {
        "route": data["route"] if ok else "question",
        "reason": str(data.get("reason", ""))[:200] if ok else error or "invalid",
        "ok": ok,
        "ms": round((time.perf_counter() - started) * 1000),
    }
