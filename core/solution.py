"""Точка входа платформы: questions.csv + knowledge-base.zip -> submission.csv."""

import concurrent.futures as cf
import csv
import os
import sys
import time
from pathlib import Path

import config
import kb
import llm
import pipeline
import retrieval

MAX_ANSWER_CHARS = 10_000


def find_file(*names):
    data_dir = Path(os.getenv("DATA_DIR", "").strip() or "data")
    for root in (data_dir, Path(__file__).resolve().parent / data_dir, Path.cwd()):
        for name in names:
            if (root / name).is_file():
                return root / name
    raise FileNotFoundError(f"Не найдены файлы: {names}")


def write_csv(answers):
    # Атомарно: при тайм-ауте платформа всегда увидит целый файл.
    with open("submission.csv.tmp", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        writer.writerows(answers.items())
    os.replace("submission.csv.tmp", "submission.csv")


def main():
    started = time.monotonic()
    with find_file("questions.csv", "train.csv").open(encoding="utf-8-sig", newline="") as f:
        questions = {row["id"].strip(): row["question"] for row in csv.DictReader(f)}
    answers = {qid: config.FALLBACK_ANSWER for qid in questions}
    write_csv(answers)

    index = retrieval.Index(kb.load_chunks(find_file("knowledge-base.zip", "knowledge_base.zip", "kb.zip")))
    print(f"Вопросов: {len(questions)}; фрагментов: {len(index.chunks)}; "
          f"индекс: {time.monotonic() - started:.1f} с", flush=True)

    def work(qid, question):
        try:
            return qid, pipeline.answer(index, pipeline.parse_dialog(question))["text"]
        except Exception as exc:
            print(f"Вопрос {qid}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return qid, ""

    workers = max(1, min(8, int(llm._env("MAX_WORKERS", default="3"))))
    pool = cf.ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(work, qid, question) for qid, question in questions.items()]
    done = 0
    try:
        for future in cf.as_completed(futures, timeout=max(1, config.DEADLINE_SEC - (time.monotonic() - started))):
            qid, text = future.result()
            if text:
                answers[qid] = text[:MAX_ANSWER_CHARS]
                done += 1
                write_csv(answers)
    except cf.TimeoutError:
        print("Дедлайн: оставшиеся ответы остаются запасными", file=sys.stderr, flush=True)

    write_csv(answers)
    print(f"Готово: {done} из {len(questions)} за {time.monotonic() - started:.0f} с", flush=True)
    # Потоки могут висеть в сетевых ретраях, а лимит платформы жёсткий.
    os._exit(0)


if __name__ == "__main__":
    main()
