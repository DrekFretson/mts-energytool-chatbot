"""Единый контракт ядра для сабмита и веба: answer(index, history) -> {text, sources, timings, ...}."""

import concurrent.futures as cf
import re
import time

import config
import llm
import prompts
import router

MARKER = re.compile(r"(?m)^\s*(Пользователь|Клиент|Ассистент|Бот)\s*:\s*")
NUMBER = re.compile(r"\d+")


def parse_dialog(raw):
    """Строка вопроса платформы -> [{role, content}], последняя реплика пользователя в конце."""
    parts = MARKER.split(raw)
    if len(parts) == 1:
        return [{"role": "user", "content": raw.strip()}]
    history = [{"role": "user", "content": parts[0].strip()}] if parts[0].strip() else []
    for speaker, content in zip(parts[1::2], parts[2::2]):
        role = "assistant" if speaker in ("Ассистент", "Бот") else "user"
        history.append({"role": role, "content": content.strip()})
    return history


def _clean(text):
    return " ".join(re.sub(r"[*#`]+", "", text).split())


_pool = cf.ThreadPoolExecutor(max_workers=8)


def format_materials(hits):
    """Разделы группируются по документам с названием продукта в заголовке группы.

    В разных документах одно понятие действует с разными условиями; явная граница документа
    не даёт модели перенести правило одного продукта на другой.
    """
    groups, numbered = [], 0
    for hit in hits:
        title, _, section = hit["crumb"].partition(" > ")
        if not groups or groups[-1][0] != (hit["file"], title):
            groups.append(((hit["file"], title), []))
        numbered += 1
        groups[-1][1].append(f"[{numbered}] {section or title}\n{hit['text']}")
    return "\n\n".join(f'<document title="{title}">\n' + "\n\n".join(parts) + "\n</document>" for (_, title), parts in groups)


def answer(index, history, note=""):
    """note - дополнение к правилам от вызывающего канала (веб-чат сообщает про форму заявки)."""
    started = time.perf_counter()
    # Маршрут считается параллельно и ответ не ждёт: к концу генерации он уже готов.
    pending = _pool.submit(router.classify, history) if config.ROUTER == "parallel" else None
    question = history[-1]["content"]
    past = [turn["content"] for turn in history[:-1] if turn["role"] == "user"]

    recent = "\n".join(f"{'Клиент' if turn['role'] == 'user' else 'Консультант'}: {turn['content']}" for turn in history[-4:])
    hits = index.search(question, past, recent)
    hits = index.context(hits) if config.WHOLE_SECTIONS else hits[: config.TOP_K]
    searched = time.perf_counter()

    materials = format_materials(hits)
    recap = recent.rsplit("\n", 1)[0] if config.DIALOG_RECAP and len(history) > 1 else ""
    messages = history[:-1] + [{"role": "user", "content": prompts.user_turn(materials, question, recap)}]
    system = f"{prompts.SYSTEM}\n\n{note}" if note else prompts.SYSTEM
    text = _clean(llm.chat([{"role": "system", "content": system}] + messages))
    finished = time.perf_counter()

    route = pending.result() if pending else None

    known = set(NUMBER.findall(materials + " ".join(turn["content"] for turn in history)))
    return {
        "text": text,
        "route": route["route"] if route else "off",
        "router": route,
        "sources": [{"file": h["file"], "crumb": h["crumb"], "score": h["score"]} for h in hits],
        "timings": {
            "search_ms": round((searched - started) * 1000),
            "llm_ms": round((finished - searched) * 1000),
            "total_ms": round((finished - started) * 1000),
            "router_ms": route["ms"] if route else 0,
        },
        "unverified_numbers": sorted(set(NUMBER.findall(text)) - known),
    }


def summarize(history):
    transcript = "\n".join(
        f"{'Клиент' if turn['role'] == 'user' else 'Бот'}: {turn['content']}" for turn in history
    )
    return llm.chat(
        [{"role": "system", "content": prompts.SUMMARY}, {"role": "user", "content": transcript}],
        max_tokens=600,
    )
