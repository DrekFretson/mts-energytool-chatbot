"""Гибридный поиск: bge-m3 + BM25, затем реранк с откатом. Любой внешний сбой деградирует, а не падает."""

import re
import sys
import time

import httpx
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


DENSE_PAUSE_AFTER = 3  # подряд упавших эмбеддингов запроса, после которых поиск временно идёт только по BM25
DENSE_PAUSE_SEC = 60.0
RERANK_PAUSE_SEC = 60.0  # реранкер, упавший или не уложившийся в тайм-аут 3 раза подряд, не зовём минуту
STARTUP_EMBED_PAUSES = (0, 10, 20, 30)  # секунды перед повторами эмбеддингов базы на старте


class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self._dense_failures, self._dense_paused_until = 0, 0.0
        self._rerank_failures, self._rerank_paused_until = 0, 0.0
        self._queries = {}
        documents = [_document(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi([tokenize(document) for document in documents])
        try:
            vectors = self._embed_twice(documents + list(config.ASPECT_QUERIES))
            self.vectors = vectors[: len(documents)]
            self.aspects = self.vectors @ vectors[len(documents) :].T
        except Exception as exc:
            print(f"Эмбеддинги недоступны, только BM25: {type(exc).__name__}", file=sys.stderr)
            self.vectors = None

    @staticmethod
    def _embed_twice(texts):
        """Эмбеддинги базы на старте. Без них весь прогон идёт на одном BM25, поэтому одного отказа мало:
        после сбоя пачки идут по одной и мелкие (лимит прокси на параллельные запросы или на размер запроса),
        затем ещё три попытки с паузами - прокси платформы может быть не готов в первую минуту после запуска."""
        try:
            return llm.embed(texts)
        except llm.BudgetExceeded:
            raise
        except Exception as exc:
            error = exc
            print(f"Эмбеддинги базы: повтор после {type(exc).__name__}", file=sys.stderr)
        llm.embed_parallel = 1
        for pause in STARTUP_EMBED_PAUSES:
            left = llm._remaining()
            if left is not None and left < pause + 5:
                break
            time.sleep(pause)
            try:
                return llm.embed(texts, batch_size=8)
            except llm.BudgetExceeded:
                raise
            except Exception as exc:
                error = exc
                print(f"Эмбеддинги базы: повтор после {type(exc).__name__}", file=sys.stderr)
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else 0
                if 400 <= status < 500 and status != 429:
                    # Отказ по существу (нет такого адреса, не тот ключ): ожидание его не исправит.
                    break
        raise error

    def _lexical(self, query, past):
        scores = self.bm25.get_scores(tokenize(query))
        if past:
            scores = scores + config.HISTORY_WEIGHT_BM25 * self.bm25.get_scores(tokenize(" ".join(past)))
        top = scores.max()
        return scores / top if top > 0 else scores

    def prefetch(self, texts):
        """Эмбеддинги текстов запросов заранее, пачками; при сбое каждый вопрос получает свои сам, как без prefetch."""
        if self.vectors is None:
            return
        texts = list(dict.fromkeys(texts))
        try:
            self._queries = dict(zip(texts, llm.embed(texts)))
        except Exception as exc:
            print(f"Эмбеддинги запросов заранее не получены: {type(exc).__name__}", file=sys.stderr)

    def _dense(self, query, past):
        texts = [query] + past
        if all(text in self._queries for text in texts):
            vectors = np.stack([self._queries[text] for text in texts])
        else:
            vectors = llm.embed(texts)
        query_vector = vectors[0] + config.HISTORY_WEIGHT_DENSE * vectors[1:].sum(axis=0)
        return self.vectors @ (query_vector / np.linalg.norm(query_vector))

    def _rerank(self, query, order):
        if time.monotonic() < self._rerank_paused_until:
            return order
        try:
            ranks = llm.rerank(query, [_document(self.chunks[i]) for i in order])
            self._rerank_failures = 0
        except Exception as exc:
            print(f"Реранкер пропущен: {type(exc).__name__}", file=sys.stderr)
            # Медленный реранкер иначе съедает RERANK_TIMEOUT на каждом вопросе.
            self._rerank_failures += 1
            if self._rerank_failures >= DENSE_PAUSE_AFTER:
                self._rerank_paused_until = time.monotonic() + RERANK_PAUSE_SEC
            return order
        if not ranks or any(not 0 <= r < len(order) for r in ranks):
            return order
        # RRF: сам реранкер путает первое место, слияние с первым этапом это гасит.
        fused = {i: 1 / (60 + position) for position, i in enumerate(order)}
        for position, r in enumerate(ranks):
            fused[order[r]] += 1 / (60 + position)
        return sorted(order, key=fused.get, reverse=True)

    def search(self, query, past=(), dialog=""):
        """dialog - последние реплики беседы для реранкера: по одной реплике он не понимает, о чём уточняющий вопрос."""
        past = list(past)
        lexical = self._lexical(query, past)
        scores, dense = lexical, None
        if self.vectors is not None and time.monotonic() >= self._dense_paused_until:
            try:
                dense = self._dense(query, past)
                scores = config.DENSE_WEIGHT * dense + (1 - config.DENSE_WEIGHT) * lexical
                self._dense_failures = 0
            except Exception as exc:
                print(f"Эмбеддинг запроса не получен: {type(exc).__name__}", file=sys.stderr)
                # Лежащий /embeddings не должен съедать срок каждого вопроса на повторах: пауза, потом проверка снова.
                self._dense_failures += 1
                if self._dense_failures >= DENSE_PAUSE_AFTER:
                    self._dense_paused_until = time.monotonic() + DENSE_PAUSE_SEC
        order = [int(i) for i in np.argsort(-scores)[: config.CANDIDATES]]
        if config.USE_RERANK:
            order = self._rerank(dialog or query, order)
        if dense is not None:
            order = self._aspect_first(dense, order)
        return [{**self.chunks[i], "score": round(float(scores[i]), 4)} for i in order]

    def _aspect_first(self, dense, order):
        """Перед найденным ставит разделы той же темы с ограничениями и со списком нужных от клиента сведений.

        Оговорка и уточняющий вопрос ответа берутся из них, а по тексту вопроса они почти не находятся.
        """
        picked = []
        for aspect in self.aspects.T:
            top = np.argsort(-(dense + config.ASPECT_WEIGHT * aspect))[: config.ASPECT_SECTIONS]
            picked.extend(int(i) for i in top if int(i) not in picked)
        return picked + [i for i in order if i not in picked]

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

        by_file = {}
        for number, section in sections.items():
            by_file.setdefault(section["file"], []).append(number)

        # Целиком идут документы с лучшей оценкой поиска, остальные - только найденными разделами.
        # 0 - без ограничения: любой найденный документ тянется целиком, пока хватает бюджета.
        whole_files = None
        if config.WHOLE_DOCUMENTS_LIMIT:
            best = {}
            for hit in hits:
                best[hit["file"]] = max(best.get(hit["file"], hit["score"]), hit["score"])
            whole_files = set(sorted(best, key=best.get, reverse=True)[: config.WHOLE_DOCUMENTS_LIMIT])

        for hit in hits:
            new_file = all(sections[n]["file"] != hit["file"] for n in chosen)
            rest = [n for n in by_file[hit["file"]] if n not in chosen]
            whole = config.WHOLE_DOCUMENTS and (whole_files is None or hit["file"] in whole_files)
            if whole and used + sum(len(part) for n in rest for part in sections[n]["parts"]) <= config.CONTEXT_CHARS:
                for number in rest:
                    take(number, hit["score"])
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
