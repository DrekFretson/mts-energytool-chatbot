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


def read_questions(path):
    """id -> вопрос. Терпит другие имена столбцов, пустые строки и огромные поля; строки без id пропускает."""
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    header = [name.strip().lower() for name in rows[0]]
    id_col = header.index("id") if "id" in header else 0
    q_col = header.index("question") if "question" in header else min(1, len(header) - 1)
    questions = {}
    for row in rows[1:]:
        qid = row[id_col].strip() if len(row) > id_col else ""
        if not qid:
            print(f"Строка без id пропущена: {row[:2]}", file=sys.stderr, flush=True)
            continue
        questions[qid] = row[q_col] if len(row) > q_col else ""
    return questions


def workers_from_env():
    try:
        return max(1, min(8, int(llm._env("MAX_WORKERS", default="3"))))
    except ValueError:
        return 3


def behind_schedule(elapsed, indexed, done, total, workers):
    """Прогноз по темпу уже готовых ответов (они идут параллельно, старт не считается) выходит за REASONING_UNTIL_SEC.
    Раньше двух волн по workers вопросов темп неточен, решения нет."""
    if done < 2 * workers:
        return False
    return elapsed + (total - done) * (elapsed - indexed) / done > config.REASONING_UNTIL_SEC


def main():
    started = time.monotonic()
    questions = read_questions(find_file("questions.csv", "train.csv"))
    answers = {qid: config.FALLBACK_ANSWER for qid in questions}
    write_csv(answers)

    llm.set_deadline(config.QUESTION_BUDGET_SEC)
    try:
        if config.REASONING_EFFORT != "none" or llm.probe():
            llm.reasoning_mode()
            print("Режим с рассуждениями: лимит токенов и тайм-аут увеличены", flush=True)
    except Exception as exc:
        print(f"Проба LLM не удалась: {type(exc).__name__}", file=sys.stderr, flush=True)
    llm.set_deadline(None)

    chunks = kb.load_chunks(find_file("knowledge-base.zip", "knowledge_base.zip", "kb.zip"))
    llm.set_deadline(config.STARTUP_EMBED_SEC)
    index = retrieval.Index(chunks)
    llm.set_deadline(None)
    print(f"Вопросов: {len(questions)}; фрагментов: {len(index.chunks)}; "
          f"индекс: {time.monotonic() - started:.1f} с", flush=True)

    finished = []
    budget = config.QUESTION_BUDGET_SEC * (1 if config.LLM_TIMEOUT < config.REASONING_TIMEOUT else 3)

    indexed = time.monotonic() - started
    workers = workers_from_env()

    def work(qid, question):
        elapsed = time.monotonic() - started
        if behind_schedule(elapsed, indexed, len(finished), len(questions), workers):
            # Прогноз по уже готовым ответам не укладывается в лимит: дальше без рассуждений и без реранка (~25% времени вопроса).
            llm.reasoning_off()
            config.USE_RERANK = False
        # Срок вопроса не дольше остатка до дедлайна: зависший вызов не держит поток после конца прогона.
        llm.set_deadline(min(budget, config.DEADLINE_SEC - (time.monotonic() - started)))
        try:
            return qid, pipeline.answer(index, pipeline.parse_dialog(question or ""))["text"]
        except Exception as exc:
            print(f"Вопрос {qid}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return qid, ""

    pool = cf.ThreadPoolExecutor(max_workers=workers)
    done, pending, flushed = 0, list(questions), time.monotonic()
    # Второй и следующие проходы: упавшие из-за сбоев прокси вопросы повторяются, пока есть время.
    while pending and config.DEADLINE_SEC - (time.monotonic() - started) > (config.RETRY_MARGIN_SEC if done else 0):
        futures = [pool.submit(work, qid, questions[qid]) for qid in pending]
        failed = []
        try:
            for future in cf.as_completed(futures, timeout=max(1, config.DEADLINE_SEC - (time.monotonic() - started))):
                qid, text = future.result()
                finished.append(qid)
                if not text:
                    failed.append(qid)
                    continue
                answers[qid] = text[:MAX_ANSWER_CHARS]
                done += 1
                if time.monotonic() - flushed > config.CSV_FLUSH_SEC:
                    write_csv(answers)
                    flushed = time.monotonic()
        except cf.TimeoutError:
            print("Дедлайн: оставшиеся ответы остаются запасными", file=sys.stderr, flush=True)
            break
        if failed and len(failed) < len(pending):
            print(f"Повторный проход: {len(failed)} вопросов", file=sys.stderr, flush=True)
        elif failed:
            break  # ни один не прошёл: сбой не временный, повтор только тратит время
        pending = failed

    write_csv(answers)
    print(f"Готово: {done} из {len(questions)} за {time.monotonic() - started:.0f} с", flush=True)
    # Потоки могут висеть в сетевых ретраях, а лимит платформы жёсткий.
    os._exit(0)


if __name__ == "__main__":
    main()
