"""Поведенческие метрики ответов против эталона. Проверка перед сабмитом, не замена судьи.

python eval/report.py submission.csv data/demo_output.csv
"""

import csv
import re
import statistics
import sys


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["id"].strip(): row["answer"] for row in csv.DictReader(f)}


def polarity(text):
    head = text.lower()[:40]
    for label, prefix in (("да", r"да\b"), ("нет", r"нет\b"), ("может", r"может иметь")):
        if re.match(prefix, head):
            return label
    return "другое"


def describe(answers):
    values = list(answers.values())
    share = lambda test: round(sum(map(test, values)) / len(values), 2)  # noqa: E731
    return {
        "n": len(values),
        "слов, медиана": statistics.median(len(a.split()) for a in values),
        "заканчивается вопросом": share(lambda a: a.rstrip().endswith("?")),
        "два вопроса и больше": share(lambda a: a.count("?") > 1),
        "разметка": share(lambda a: bool(re.search(r"[*#`]|^\s*- ", a))),
        "заглушка": share(lambda a: a.startswith("Точного ответа в материалах сейчас")),
    }


def main(submission_path, golden_path):
    submission, golden = read(submission_path), read(golden_path)
    common = sorted(submission.keys() & golden.keys(), key=int)
    print("ответы ", describe(submission))
    print("эталон ", describe(golden))
    matched, recalled, total = 0, 0, 0
    for qid in common:
        same = polarity(submission[qid]) == polarity(golden[qid])
        matched += same
        expected = set(re.findall(r"\d+", golden[qid]))
        found = expected & set(re.findall(r"\d+", submission[qid]))
        recalled += len(found)
        total += len(expected)
        if not same or expected - found:
            print(f"  id={qid}: вывод {polarity(submission[qid])}/{polarity(golden[qid])}, нет чисел {sorted(expected - found)}")
    print(f"совпал вывод: {matched}/{len(common)}; числа эталона: {recalled}/{total}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
