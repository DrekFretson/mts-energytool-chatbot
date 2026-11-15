"""Заглушка OpenAI-совместимого прокси для разработки без GPU. Только stdlib."""

import json
import os
import re
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 128
CHAT_DELAY = float(os.getenv("FAKE_CHAT_DELAY", "0"))
WORD_DELAY = float(os.getenv("FAKE_WORD_DELAY", "0"))  # пауза между словами потокового ответа
ANSWER = "[fake-llm] Ответ-заглушка: подключите настоящую модель. Какой у вас объект?"


def vector(text):
    result = [0.0] * DIM
    for word in re.findall(r"\w+", text.lower()):
        result[zlib.crc32(word.encode()) % DIM] += 1.0
    return result


def route(transcript):
    """Маршрут для классификатора: метка [operator] или [ready] в последней реплике клиента, иначе question."""
    last = [line for line in transcript.splitlines() if line.startswith("Клиент:")][-1:]
    found = re.search(r"\[(operator|ready)\]", last[0] if last else "")
    page = re.search(r"\[link:(\w+)\]", last[0] if last else "")
    return {"reason": "fake", "route": found.group(1) if found else "question", "product": page.group(1) if page else "none", "extra": "none"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            result = {"data": [{"index": i, "embedding": vector(t)} for i, t in enumerate(texts)]}
        elif self.path.endswith("/rerank"):
            result = {"results": [{"index": i, "relevance_score": 1.0} for i in range(len(body["documents"]))]}
        elif "response_format" in body:
            result = {"choices": [{"message": {"role": "assistant", "content": json.dumps(route(body["messages"][-1]["content"]))}}]}
        elif body.get("stream"):
            return self.stream()
        else:
            time.sleep(CHAT_DELAY)
            result = {"choices": [{"message": {"role": "assistant", "content": ANSWER}}]}
        payload = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


    def stream(self):
        """Поток как у прокси с парсером рассуждений: куски reasoning_content, затем content по словам."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        words = ANSWER.split(" ")
        deltas = [{"reasoning_content": "думаю "}] * 2 + [{"content": word + (" " if i < len(words) - 1 else "")} for i, word in enumerate(words)]
        for delta in deltas:
            time.sleep(CHAT_DELAY / 2 if "reasoning_content" in delta else WORD_DELAY)
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': delta}]}, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


def make_server(port=0):
    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


if __name__ == "__main__":
    make_server(int(os.getenv("PORT", "9000"))).serve_forever()
