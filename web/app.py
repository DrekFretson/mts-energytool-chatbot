"""Веб-контур: чат, заявка, история сессий, отладка. Логика ответа только в core/."""

import hmac
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import config  # noqa: E402
import kb  # noqa: E402
import pipeline  # noqa: E402
import retrieval  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
DB_PATH = os.getenv("DB_PATH", "data/app.sqlite")
KB_PATH = os.getenv("KB_PATH", "data/knowledge-base.zip")
DEBUG_TOKEN = os.getenv("DEBUG_TOKEN", "")
HISTORY_LIMIT = 12
OFFER_AFTER_USER_TURNS = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS leads (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id), name TEXT NOT NULL, phone TEXT NOT NULL,
    company TEXT NOT NULL, consent_at REAL NOT NULL, summary TEXT NOT NULL, created_at REAL NOT NULL);
"""


def query(sql, *args):
    # shortcut: SQLite и соединение на запрос, один писатель; при нескольких репликах заменить на Postgres.
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            return [dict(row) for row in connection.execute(sql, args).fetchall()]
    finally:
        connection.close()


state = {}


@asynccontextmanager
async def lifespan(_):
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.executescript(SCHEMA)
    connection.close()
    state["index"] = retrieval.Index(kb.load_chunks(KB_PATH))
    yield


app = FastAPI(title="МТС EnergyTool консультант", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class Message(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=2000)]


class Lead(BaseModel):
    name: Annotated[str, Field(min_length=2, max_length=200)]
    phone: Annotated[str, Field(min_length=10, max_length=30)]
    company: Annotated[str, Field(min_length=1, max_length=200)]
    consent: bool


def has_debug_access(token):
    return bool(DEBUG_TOKEN) and bool(token) and hmac.compare_digest(token.encode(), DEBUG_TOKEN.encode())


def require_debug_access(token):
    if not has_debug_access(token):
        raise HTTPException(403, "Нужен токен отладки")


def require_session(session_id):
    if not query("SELECT 1 FROM sessions WHERE id = ?", session_id):
        raise HTTPException(404, "Сессия не найдена")


def load_history(session_id, limit=-1):
    rows = query(
        "SELECT role, content FROM (SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
        session_id, limit,
    )
    return rows


def normalize_phone(raw):
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11 and digits[0] in "78":
        return "+7" + digits[1:]
    if len(digits) == 10:
        return "+7" + digits
    raise HTTPException(422, "Проверьте номер телефона")


@app.post("/api/sessions")
def create_session():
    session_id = str(uuid.uuid4())
    query("INSERT INTO sessions (id, created_at) VALUES (?, ?)", session_id, time.time())
    return {"id": session_id}


@app.post("/api/sessions/{session_id}/messages")
def send_message(session_id: str, message: Message, x_debug_token: Annotated[str | None, Header()] = None):
    require_session(session_id)
    query(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
        session_id, message.text, time.time(),
    )
    try:
        result = pipeline.answer(state["index"], load_history(session_id, HISTORY_LIMIT))
    except Exception as exc:
        print(f"Ошибка генерации: {type(exc).__name__}", file=sys.stderr, flush=True)
        result = {"text": config.FALLBACK_ANSWER, "sources": [], "timings": {}, "unverified_numbers": [], "degraded": True}

    meta = {key: value for key, value in result.items() if key != "text"}
    query(
        "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?, 'assistant', ?, ?, ?)",
        session_id, result["text"], json.dumps(meta, ensure_ascii=False), time.time(),
    )
    user_turns = query("SELECT count(*) AS n FROM messages WHERE session_id = ? AND role = 'user'", session_id)[0]["n"]
    # shortcut: момент предложения по числу реплик; заменить на разбор намерений (сигнал покупки, отказ).
    response = {"text": result["text"], "offer_lead": user_turns == OFFER_AFTER_USER_TURNS}
    if has_debug_access(x_debug_token):
        response["debug"] = meta
    return response


@app.post("/api/sessions/{session_id}/lead")
def create_lead(session_id: str, lead: Lead):
    require_session(session_id)
    if not lead.consent:
        raise HTTPException(422, "Нужно согласие на обработку персональных данных")
    phone = normalize_phone(lead.phone)
    if query("SELECT 1 FROM leads WHERE session_id = ?", session_id):
        return {"status": "exists"}
    try:
        summary = pipeline.summarize(load_history(session_id))
    except Exception as exc:
        # Заявку нельзя терять из-за сбоя LLM: менеджер увидит диалог в истории.
        print(f"Резюме не построено: {type(exc).__name__}", file=sys.stderr, flush=True)
        summary = "Резюме не сформировано, см. историю диалога."
    now = time.time()
    query(
        "INSERT OR IGNORE INTO leads (session_id, name, phone, company, consent_at, summary, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        session_id, lead.name.strip(), phone, lead.company.strip(), now, summary, now,
    )
    return {"status": "created", "summary": summary}


@app.get("/api/admin/sessions")
def list_sessions(x_debug_token: Annotated[str | None, Header()] = None):
    require_debug_access(x_debug_token)
    return query(
        "SELECT s.id, s.created_at, count(m.id) AS messages, l.session_id IS NOT NULL AS has_lead "
        "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id LEFT JOIN leads l ON l.session_id = s.id "
        "GROUP BY s.id ORDER BY s.created_at DESC LIMIT 500"
    )


@app.get("/api/admin/sessions/{session_id}")
def get_session(session_id: str, x_debug_token: Annotated[str | None, Header()] = None):
    require_debug_access(x_debug_token)
    require_session(session_id)
    leads = query("SELECT * FROM leads WHERE session_id = ?", session_id)
    return {
        "messages": query("SELECT role, content, meta, created_at FROM messages WHERE session_id = ? ORDER BY id", session_id),
        "lead": leads[0] if leads else None,
    }


@app.get("/healthz")
def health():
    return {"status": "ok" if "index" in state else "starting"}


@app.get("/")
def index_page():
    return FileResponse(STATIC / "index.html")


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC / "admin.html")
