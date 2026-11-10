"""Единый контракт ядра для сабмита и веба: answer(index, history) -> {text, sources, timings, ...}."""

import re
import time

import config
import llm
import prompts

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


def answer(index, history):
    started = time.perf_counter()
    question = history[-1]["content"]
    past = [turn["content"] for turn in history[:-1] if turn["role"] == "user"]

    hits = index.search(question, past)
    hits = index.context(hits) if config.WHOLE_SECTIONS else hits[: config.TOP_K]
    searched = time.perf_counter()

    materials = "\n\n".join(f"[{hit['crumb']}]\n{hit['text']}" for hit in hits)
    text = _clean(
        llm.chat(
            [{"role": "system", "content": prompts.SYSTEM}]
            + history[:-1]
            + [{"role": "user", "content": prompts.user_turn(materials, question)}]
        )
    )
    finished = time.perf_counter()

    known = set(NUMBER.findall(materials + " ".join(turn["content"] for turn in history)))
    return {
        "text": text,
        "sources": [{"file": h["file"], "crumb": h["crumb"], "score": h["score"]} for h in hits],
        "timings": {
            "search_ms": round((searched - started) * 1000),
            "llm_ms": round((finished - searched) * 1000),
            "total_ms": round((finished - started) * 1000),
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
