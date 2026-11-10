"""Заглушка OpenAI-совместимого прокси для разработки без GPU. Только stdlib."""

import json
import os
import re
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 128
CHAT_DELAY = float(os.getenv("FAKE_CHAT_DELAY", "0"))


def vector(text):
    result = [0.0] * DIM
    for word in re.findall(r"\w+", text.lower()):
        result[zlib.crc32(word.encode()) % DIM] += 1.0
    return result


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
        else:
            time.sleep(CHAT_DELAY)
            content = "[fake-llm] Ответ-заглушка: подключите настоящую модель. Какой у вас объект?"
            result = {"choices": [{"message": {"role": "assistant", "content": content}}]}
        payload = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def make_server(port=0):
    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


if __name__ == "__main__":
    make_server(int(os.getenv("PORT", "9000"))).serve_forever()
