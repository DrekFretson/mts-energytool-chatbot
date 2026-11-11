"""Где ломается решение: оценка ответов по эталону. Не судья платформы, а поиск провалов перед сабмитом.

python eval/grade.py submission.csv eval/set/questions.csv eval/set/golden.csv [grades.json]
Нужны LLM_BASE_URL, LLM_API_KEY, LLM_MODEL (например OpenRouter и qwen/qwen3.8-27b): один вызов на ответ, ~0.03 $ на 100 ответов.
Провал: оценка 2 и ниже (другой вывод, противоречие эталону, упущено главное, отказ не к месту).
"""

import concurrent.futures as cf
import csv
import json
import os
import sys

import httpx

SYSTEM = """Ты проверяешь ответ ИИ-консультанта МТС EnergyTool по эталонному ответу. Эталон считается верным.
Оцени только содержание, не стиль и не длину:
- verdict: совпадает ли вывод ответа с выводом эталона (да / нет / зависит / отказ / не по теме). same, different или none, если вывода нет в обоих.
- missing: ключевые факты, условия или оговорки эталона, которых в ответе нет (кратко, не больше трёх).
- contradictions: утверждения ответа, которые противоречат эталону (кратко). Дополнительные факты, не противоречащие эталону, сюда не относятся.
- scope: ok, если оба отвечают по существу или оба отказывают; wrong_refusal, если ответ отказывает или уходит от ответа, а эталон отвечает; wrong_answer, если ответ уверенно отвечает там, где эталон говорит, что данных нет или вопрос вне темы.
- question: ok, если уточняющий вопрос ответа по смыслу тот же, что в эталоне, или его нет в обоих; missing, если в эталоне есть, а в ответе нет; other, если вопрос о другом.
- grade: 4 - по содержанию равноценен эталону; 3 - вывод тот же, упущена одна деталь; 2 - вывод тот же, но упущено главное или есть неточность; 1 - вывод другой или серьёзная ошибка; 0 - противоречит эталону или отвечает не на тот вопрос."""
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "missing", "contradictions", "scope", "question", "grade"],
    "properties": {
        "verdict": {"type": "string", "enum": ["same", "different", "none"]},
        "missing": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "contradictions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "scope": {"type": "string", "enum": ["ok", "wrong_refusal", "wrong_answer"]},
        "question": {"type": "string", "enum": ["ok", "missing", "other"]},
        "grade": {"type": "integer", "minimum": 0, "maximum": 4},
    },
}


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["id"].strip(): row for row in csv.DictReader(f)}


def grade(client, question, reference, answer):
    body = {
        "model": os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b"),
        "temperature": 0,
        "max_tokens": 400,
        "reasoning": {"effort": "none"},
        "response_format": {"type": "json_schema", "json_schema": {"name": "grade", "schema": SCHEMA, "strict": True}},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Вопрос клиента:\n{question}\n\nЭталонный ответ:\n{reference}\n\nОтвет консультанта:\n{answer}"},
        ],
    }
    for _ in range(2):
        try:
            response = client.post("/chat/completions", json=body)
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            error = f"{type(exc).__name__}"
    return {"error": error}


def main(submission_path, questions_path, golden_path, out_path="grades.json"):
    answers, questions, golden = read(submission_path), read(questions_path), read(golden_path)
    ids = [qid for qid in questions if qid in answers and qid in golden]
    client = httpx.Client(
        base_url=os.environ["LLM_BASE_URL"].rstrip("/"),
        headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
        timeout=120,
    )
    with cf.ThreadPoolExecutor(6) as pool:
        grades = dict(zip(ids, pool.map(
            lambda qid: grade(client, questions[qid]["question"], golden[qid]["answer"], answers[qid]["answer"]), ids)))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(grades, f, ensure_ascii=False, indent=1)
    scored = {qid: g for qid, g in grades.items() if "grade" in g}
    failed = [qid for qid, g in scored.items() if g["grade"] <= 2]
    print(f"оценено {len(scored)} из {len(ids)}, средняя {sum(g['grade'] for g in scored.values()) / max(1, len(scored)):.2f}, провалов {len(failed)}")
    for qid in failed:
        g = scored[qid]
        kind = golden[qid].get("type", "")
        print(f"  id={qid} {kind} оценка {g['grade']}: {'; '.join(g['contradictions'] or g['missing'])[:200]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
