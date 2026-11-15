"""Ссылки на страницы сайта EnergyTool под ответом бота.

Страницы перечислены в site_links.json: добавить или убрать страницу можно без правки кода. Какая страница
подходит к реплике, решает LLM-классификатор со схемой из этого списка; адреса модель не видит и не пишет,
их подставляет сервер. Любой сбой означает ответ без ссылок.
"""

import json
from pathlib import Path

import config
import llm

PAGES = json.loads((Path(__file__).resolve().parent / "site_links.json").read_text(encoding="utf-8"))


def _options(pages, field):
    return "\n".join(f"{key}: {page[field]}" for key, page in pages.items())


PROMPT = f"""Ты подбираешь страницы сайта МТС EnergyTool, которые стоит показать клиенту под ответом консультанта. Смотри на ПОСЛЕДНЮЮ реплику клиента с учётом диалога.

product - страница продукта, о котором спрашивает клиент, или none:
{_options(PAGES["products"], "about")}

extra - одна дополнительная страница, если реплика точно про этот повод, иначе none:
{_options(PAGES["extra"], "when")}

Выбирай none, если реплика не про энергетику и продукты EnergyTool, если клиент просит менеджера, звонок или заявку, благодарит или прощается, если непонятно, о каком продукте речь, и если подходящей страницы в списке нет. Не выбирай страницу продукта, о котором клиент не спрашивал. Ответь JSON по схеме: product, extra."""

SCHEMA = {
    "type": "object",
    "properties": {
        "product": {"type": "string", "enum": [*PAGES["products"], "none"]},
        "extra": {"type": "string", "enum": [*PAGES["extra"], "none"]},
    },
    "required": ["product", "extra"],
    "additionalProperties": False,
}


def pick(history):
    """Не больше двух ссылок [{url, label}]: страница продукта и дополнительная."""
    lines = [f"{'Клиент' if turn['role'] == 'user' else 'Бот'}: {turn['content']}" for turn in history[-6:]]
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": "Диалог:\n" + "\n".join(lines) + "\n\nКакие страницы показать под ответом на последнюю реплику клиента?"},
    ]
    try:
        data = llm.chat_json(messages, SCHEMA, config.ROUTER_MAX_TOKENS, timeout=config.ROUTER_TIMEOUT)
    except Exception:  # noqa: BLE001 - ссылки не должны ронять ответ
        return []
    if not isinstance(data, dict):
        return []
    chosen = [PAGES["products"].get(data.get("product")), PAGES["extra"].get(data.get("extra"))]
    links = []
    for page in chosen:
        if page and all(page["url"] != link["url"] for link in links):
            links.append({"url": page["url"], "label": page["label"]})
    return links
