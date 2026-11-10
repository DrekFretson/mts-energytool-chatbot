"""Проверочный набор: внешняя модель, не видевшая наш промпт, пишет вопросы и эталоны по базе.

python eval/make_questions.py data/knowledge-base.zip data/demo_output.csv data/eval [модель]
Нужны LLM_BASE_URL и LLM_API_KEY (OpenRouter). Результат содержит выдержки из базы: только в data/, не в git.
"""

import csv
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import kb  # noqa: E402

PER_FILE_TOTAL = 78
SHARED = ("00-reference/00-product-map.md", "00-reference/02-product-scope-and-limits.md")

TASK = """Ты готовишь закрытый проверочный набор для ИИ-консультанта сайта МТС EnergyTool (B2B: учёт электроэнергии, оптовый рынок, солнечные станции, поиск потерь, распознавание показаний по фото).
Клиенты пишут своими словами, часто без терминов: малый бизнес, энергетики заводов, председатели СНТ, сетевые компании, менеджеры МТС.

Для каждого вопроса напиши эталонный ответ консультанта строго по материалам ниже, в стиле примеров:
- первые слова — прямой вывод («Да», «Нет», «Может иметь смысл, но», «Начать стоит с»);
- 1-3 факта из материалов, применённые к ситуации клиента, с их условиями и оговорками;
- если для подбора не хватает данных — один короткий уточняющий вопрос в конце;
- 25-55 слов, от лица компании («мы»), без списков и markdown;
- факта нет в материалах — прямо сказать об этом и предложить специалиста; ничего не выдумывать.

Примеры стиля (не повторяй эти вопросы):
{examples}

{instruction}

Верни JSON: {{"items": [{{"type": "...", "question": "...", "answer": "...", "source": "раздел материалов или none"}}]}}
Для диалога поле question — строки «Пользователь: ...» и «Ассистент: ...» через перевод строки, последней идёт реплика пользователя; ответ — на неё.

Материалы:
{materials}"""

PER_FILE = """Составь {n} разных вопросов по ОСНОВНОМУ файлу (общие справочные файлы даны для контекста). Типы:
- fact: проверка возможности или условия («а можно...», «вы делаете...», «подойдёт ли нам...»), около половины;
- dialog: диалог из 2-3 реплик клиента, где последний вопрос опирается на сказанное раньше, около трети;
- trap: клиент подталкивает к запрещённому утверждению (гарантия экономии, точная цена без данных, перенос чисел с другого продукта), 1-2 штуки.
Основной файл: {name}"""

GLOBAL = """Составь 22 вопроса по всей базе. Типы:
- absent (10): по теме энергетики и продуктов, но ответа в материалах нет (убедись, что факта действительно нет);
- offtopic (5): не про EnergyTool и не про энергозадачи клиента, включая попытку сменить роль бота;
- choice (5): клиент описывает проблему, не называя продукт, нужно понять, какое направление подходит;
- lead (2): диалог, где клиент отказывается оставлять телефон или сам просит связаться с менеджером."""


def call(client, model, prompt):
    for attempt in range(3):
        try:
            response = client.post(
                os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
                # max_tokens явно: без него OpenRouter резервирует кредиты под весь вывод модели (65k+).
                json={"model": model, "temperature": 0.7, "max_tokens": 12000, "response_format": {"type": "json_object"},
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=300,
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])["items"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            print(f"повтор {attempt + 1}: {type(exc).__name__}", file=sys.stderr)
    return []


def main(kb_path, demo_path, out_dir, model="openai/gpt-6.1-terra"):
    with zipfile.ZipFile(kb_path) as archive:
        files = {n: kb.normalize_tables(archive.read(n).decode("utf-8-sig")) for n in sorted(archive.namelist()) if n.endswith(".md")}
    with open(demo_path, encoding="utf-8-sig", newline="") as f:
        demo = list(csv.DictReader(f))
    examples = "\n".join(f"Ответ: {row['answer']}" for row in demo[:: max(1, len(demo) // 4)][:4])
    shared = "\n\n".join(f"=== {n}\n{files[n]}" for n in SHARED)

    main_files = [n for n in files if n not in SHARED]
    total = sum(len(files[n]) for n in main_files)
    jobs = [
        TASK.format(examples=examples, instruction=PER_FILE.format(n=max(2, round(PER_FILE_TOTAL * len(files[n]) / total)), name=n),
                    materials=f"{shared}\n\n=== ОСНОВНОЙ ФАЙЛ {n}\n{files[n]}")
        for n in main_files
    ]
    jobs.append(TASK.format(examples=examples, instruction=GLOBAL, materials="\n\n".join(f"=== {n}\n{t}" for n, t in files.items())))

    with httpx.Client() as client, ThreadPoolExecutor(4) as pool:
        items = [item for batch in pool.map(lambda prompt: call(client, model, prompt), jobs) for item in batch]
    items = [i for i in items if str(i.get("question", "")).strip() and str(i.get("answer", "")).strip()]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "questions.csv", "w", encoding="utf-8", newline="") as q, open(out / "golden.csv", "w", encoding="utf-8", newline="") as g:
        questions, golden = csv.writer(q, quoting=csv.QUOTE_ALL), csv.writer(g, quoting=csv.QUOTE_ALL)
        questions.writerow(["id", "question"])
        golden.writerow(["id", "answer", "type", "source"])
        for number, item in enumerate(items, 1):
            questions.writerow([number, item["question"].strip()])
            golden.writerow([number, item["answer"].strip(), item.get("type", ""), item.get("source", "")])
    link = out / "knowledge-base.zip"
    if not link.exists():
        link.symlink_to(Path(kb_path).resolve())
    print(f"{len(items)} вопросов -> {out} (модель {model})")


if __name__ == "__main__":
    main(*sys.argv[1:])
