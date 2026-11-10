"""Гибридный поиск: bge-m3 + BM25, затем реранк с откатом. Любой внешний сбой деградирует, а не падает."""

import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi

import config
import llm

WORD = re.compile(r"[0-9a-zа-я]+")


def tokenize(text):
    # Обрезка до 6 символов заменяет стеммер: PyStemmer нет в разрешённых пакетах.
    return [word[:6] for word in WORD.findall(text.lower().replace("ё", "е"))]


def _document(chunk):
    return f"{chunk['crumb']}\n{chunk['text']}"


class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        documents = [_document(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi([tokenize(document) for document in documents])
        try:
            self.vectors = llm.embed(documents)
        except Exception as exc:
            print(f"Эмбеддинги недоступны, только BM25: {type(exc).__name__}", file=sys.stderr)
            self.vectors = None

    def _lexical(self, query, past):
        scores = self.bm25.get_scores(tokenize(query))
        if past:
            scores = scores + config.HISTORY_WEIGHT_BM25 * self.bm25.get_scores(tokenize(" ".join(past)))
        top = scores.max()
        return scores / top if top > 0 else scores

    def _dense(self, query, past):
        vectors = llm.embed([query] + past)
        query_vector = vectors[0] + config.HISTORY_WEIGHT_DENSE * vectors[1:].sum(axis=0)
        return self.vectors @ (query_vector / np.linalg.norm(query_vector))

    def _rerank(self, query, order):
        try:
            ranks = llm.rerank(query, [_document(self.chunks[i]) for i in order])
        except Exception as exc:
            print(f"Реранкер пропущен: {type(exc).__name__}", file=sys.stderr)
            return order
        if not ranks or any(not 0 <= r < len(order) for r in ranks):
            return order
        # RRF: сам реранкер путает первое место, слияние с первым этапом это гасит.
        fused = {i: 1 / (60 + position) for position, i in enumerate(order)}
        for position, r in enumerate(ranks):
            fused[order[r]] += 1 / (60 + position)
        return sorted(order, key=fused.get, reverse=True)

    def search(self, query, past=()):
        past = list(past)
        lexical = self._lexical(query, past)
        scores = lexical
        if self.vectors is not None:
            try:
                scores = config.DENSE_WEIGHT * self._dense(query, past) + (1 - config.DENSE_WEIGHT) * lexical
            except Exception as exc:
                print(f"Эмбеддинг запроса не получен: {type(exc).__name__}", file=sys.stderr)
        order = [int(i) for i in np.argsort(-scores)[: config.CANDIDATES]]
        if config.USE_RERANK:
            order = self._rerank(query, order)
        return [{**self.chunks[i], "score": round(float(scores[i]), 4)} for i in order]

    def context(self, hits):
        """Найденные куски -> разделы целиком по рангу, пока влезают в CONTEXT_CHARS.

        С первым разделом документа сразу берутся его защитные разделы: оговорки и запреты
        должны идти вместе с фактами, иначе модель отдаёт число без условия.
        """
        sections = {}
        for chunk in self.chunks:
            section = sections.setdefault(chunk["section"], {"file": chunk["file"], "crumb": chunk["crumb"], "parts": []})
            section["parts"].append(chunk["text"])
        guards = {}
        for number, section in sections.items():
            if section["crumb"].rsplit(" > ", 1)[-1] in config.GUARD_HEADINGS:
                guards.setdefault(section["file"], []).append(number)

        chosen, score, used = [], {}, 0

        def take(number, hit_score):
            nonlocal used
            if number in chosen:
                return
            size = sum(len(part) for part in sections[number]["parts"])
            if used + size <= config.CONTEXT_CHARS:
                chosen.append(number)
                score[number] = hit_score
                used += size

        for hit in hits:
            new_file = all(sections[n]["file"] != hit["file"] for n in chosen)
            take(hit["section"], hit["score"])
            if new_file and hit["section"] in chosen:
                for number in guards.get(hit["file"], []):
                    take(number, None)

        file_rank = {}
        for number in chosen:
            file_rank.setdefault(sections[number]["file"], len(file_rank))
        # В промпте разделы одного документа идут подряд и в его порядке, так легче читать связи.
        return [
            {"file": sections[n]["file"], "crumb": sections[n]["crumb"], "text": "\n".join(sections[n]["parts"]), "score": score[n]}
            for n in sorted(chosen, key=lambda n: (file_rank[sections[n]["file"]], n))
        ]
