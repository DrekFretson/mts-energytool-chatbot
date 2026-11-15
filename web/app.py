"""Веб-контур: чат, заявка, история сессий, отладка. Логика ответа только в core/."""

import hmac
import json
import os
import queue
import re
import sqlite3
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import config  # noqa: E402
import kb  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402
import retrieval  # noqa: E402

from web import site_links  # noqa: E402
from web.stream import answer as answer_stream  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
DB_PATH = os.getenv("DB_PATH", "data/app.sqlite")
KB_PATH = os.getenv("KB_PATH", "data/knowledge-base.zip")
DEBUG_TOKEN = os.getenv("DEBUG_TOKEN", "")
HISTORY_LIMIT = 12
ANSWER_BUDGET_SEC = float(os.getenv("ANSWER_BUDGET_SEC", "90"))  # дольше ответа пользователь не ждёт: отдаём заглушку
MESSAGES_PER_MINUTE = int(os.getenv("MESSAGES_PER_MINUTE", "20"))  # на один IP
MAX_BODY_BYTES = 64 * 1024
# Сайты, которым разрешено встраивать чат во фрейм (через пробел); свой адрес сервиса разрешён всегда.
EMBED_ORIGINS = " ".join(["'self'", *re.findall(r"https?://[\w.*:-]+", os.getenv("EMBED_ORIGINS", "https://energytool.mts.ru"))])
# В вебе граф включён: классификатор параллельно с поиском, правило про заявку в промпте остаётся страховкой (G+P).
config.ROUTER = os.getenv("WEB_ROUTER", "parallel")
NOTE_FORM = ("Особенность этого канала (веб-чат), она важнее образцов: заявка оформляется через форму, которая открывается "
             "под сообщением, когда клиент хочет связаться с менеджером. Поэтому не проси писать ФИО, телефон и компанию в чат, "
             "а попроси заполнить форму заявки под сообщением и скажи, что менеджер получит краткое резюме диалога.")
NOTE_LEAD_DONE = ("Клиент уже отправил заявку в этом диалоге: контакты больше не спрашивай и заявку не предлагай. "
                  "Если он снова просит менеджера, скажи, что заявка уже передана и менеджер свяжется с ним.")

if not Path(KB_PATH).is_file():
    sys.exit(f"Не найден файл базы знаний {KB_PATH}. Положите архив knowledge-base.zip из материалов кейса "
             "в data/knowledge-base.zip (в Docker он монтируется в /kb/knowledge-base.zip) и запустите снова.")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS leads (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id), name TEXT NOT NULL, phone TEXT NOT NULL,
    company TEXT NOT NULL, consent_at REAL NOT NULL, summary TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS manager_leads (
    session_id TEXT PRIMARY KEY REFERENCES leads(session_id), accepted_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS manager_reports (
    session_id TEXT PRIMARY KEY REFERENCES manager_leads(session_id), report TEXT NOT NULL, closed_at REAL NOT NULL);
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


# Сбой модели в чате называем прямо: текст сабмита "в материалах не нашёл" здесь вводил бы клиента в заблуждение.
UNAVAILABLE_ANSWER = ("Сейчас не получается ответить: сервис временно недоступен. Попробуйте ещё раз через минуту "
                      "или нажмите «Связаться с менеджером» - мы передадим вашу заявку.")
state = {}
_pool = ThreadPoolExecutor(max_workers=8)


@asynccontextmanager
async def lifespan(_):
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.executescript(SCHEMA)
    # WAL: чтение истории не ждёт записи соседних сессий; режим сохраняется в самом файле базы.
    connection.execute("PRAGMA journal_mode=WAL")
    connection.close()
    if config.REASONING_EFFORT != "none":
        llm.reasoning_mode()
    state["index"] = retrieval.Index(kb.load_chunks(KB_PATH))
    yield


app = FastAPI(title="МТС EnergyTool консультант", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def limit_body(request, call_next):
    if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "Слишком большой запрос"}, status_code=413)
    return await call_next(request)


@app.middleware("http")
async def frame_policy(request, call_next):
    # Во фрейм встраивается только страница чата и только на разрешённых сайтах: форма заявки не должна
    # оказаться внутри чужой страницы.
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = f"frame-ancestors {EMBED_ORIGINS if request.url.path == '/chat' else chr(39) + 'none' + chr(39)}"
    return response


# shortcut: ограничения в памяти одного процесса; при нескольких репликах перенести в Valkey.
_guard_lock = threading.Lock()
_busy_sessions = set()
_streams = {}  # сессия -> её текущий потоковый ответ, чтобы запрос отмены мог его остановить
STREAM_STALE_SEC = 10.0  # читатель не забирает события дольше этого - он ушёл (сам поток шлёт событие не реже раза в секунду)
_recent = {}


def rate_limit(request, kind="messages", limit=None):
    """Не больше limit действий одного вида в минуту с одного IP (для сообщений - MESSAGES_PER_MINUTE)."""
    limit = limit or MESSAGES_PER_MINUTE
    now = time.monotonic()
    with _guard_lock:
        hits = _recent.setdefault((kind, request.client.host if request.client else ""), deque())
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= limit:
            raise HTTPException(429, "Слишком много запросов, подождите минуту")
        hits.append(now)


class session_busy:
    """Одна обработка на сессию: двойной клик или повтор не задваивает ответ и заявку.

    wait - сколько секунд подождать, пока освободится: отменённый ответ отпускает сессию не мгновенно.
    """

    def __init__(self, session_id, wait=0.0):
        self.session_id, self.wait = session_id, wait

    def __enter__(self):
        deadline = time.monotonic() + self.wait
        while True:
            with _guard_lock:
                if self.session_id not in _busy_sessions:
                    _busy_sessions.add(self.session_id)
                    return
            if time.monotonic() >= deadline:
                raise HTTPException(429, "Предыдущий запрос ещё обрабатывается")
            time.sleep(0.1)

    def __exit__(self, *exc):
        with _guard_lock:
            _busy_sessions.discard(self.session_id)


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
        "SELECT role, content FROM (SELECT * FROM messages WHERE session_id = ? AND role IN ('user', 'assistant') ORDER BY id DESC LIMIT ?) ORDER BY id",
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
def create_session(request: Request):
    rate_limit(request, "sessions", 60)
    session_id = str(uuid.uuid4())
    query("INSERT INTO sessions (id, created_at) VALUES (?, ?)", session_id, time.time())
    return {"id": session_id}


def accept(session_id, text):
    """Сохраняет реплику клиента: (её id, ведёт ли сейчас диалог менеджер, есть ли заявка)."""
    message_id = query(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, ?) RETURNING id",
        session_id, text, time.time(),
    )[0]["id"]
    manager = query("SELECT 1 FROM manager_leads a LEFT JOIN manager_reports r ON r.session_id=a.session_id WHERE a.session_id=? AND r.session_id IS NULL", session_id)
    return message_id, bool(manager), bool(query("SELECT 1 FROM leads WHERE session_id = ?", session_id))


def fallback(exc):
    print(f"Ошибка генерации: {type(exc).__name__}", file=sys.stderr, flush=True)
    return {"text": UNAVAILABLE_ANSWER, "sources": [], "timings": {}, "unverified_numbers": [], "degraded": True}


def record(session_id, result, links, has_lead, x_debug_token):
    """Сохраняет ответ бота и собирает ответ клиенту. links - отложенный подбор ссылок на сайт."""
    offered = query("SELECT 1 FROM messages WHERE session_id = ? AND json_extract(meta, '$.offer_lead')", session_id)
    open_form = result.get("route") == "operator" and not has_lead
    offer_lead = result.get("route") == "ready" and not has_lead and not offered
    # Под заглушкой и под просьбой о менеджере ссылки на продукты неуместны.
    try:
        shown = [] if result.get("degraded") or result.get("route") == "operator" else links.result(timeout=config.ROUTER_TIMEOUT)
    except Exception:  # noqa: BLE001 - ссылки не должны ронять ответ
        shown = []
    meta = {key: value for key, value in result.items() if key != "text"} | {"offer_lead": offer_lead, "links": shown}
    query(
        "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?, 'assistant', ?, ?, ?)",
        session_id, result["text"], json.dumps(meta, ensure_ascii=False), time.time(),
    )
    response = {"text": result["text"], "open_form": open_form, "offer_lead": offer_lead, "links": shown}
    if has_debug_access(x_debug_token):
        response["debug"] = meta
    elif x_debug_token is not None:
        response["debug_error"] = "Неверный токен отладки" if DEBUG_TOKEN else "Отладка отключена на сервере: не задан DEBUG_TOKEN"
    return response


@app.post("/api/sessions/{session_id}/messages")
def send_message(session_id: str, message: Message, request: Request, x_debug_token: Annotated[str | None, Header()] = None):
    require_session(session_id)
    rate_limit(request)
    with session_busy(session_id):
        return reply(session_id, message, x_debug_token)


def reply(session_id, message, x_debug_token):
    _, manager, has_lead = accept(session_id, message.text)
    if manager:
        return {"manager_active": True}
    history = load_history(session_id, HISTORY_LIMIT)
    links = _pool.submit(site_links.pick, history)
    llm.set_deadline(ANSWER_BUDGET_SEC)
    try:
        result = pipeline.answer(state["index"], history, NOTE_LEAD_DONE if has_lead else NOTE_FORM)
    except Exception as exc:
        result = fallback(exc)
    finally:
        llm.set_deadline(None)
    return record(session_id, result, links, has_lead, x_debug_token)


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class Stream:
    """Один потоковый ответ. Очистка (снять блокировку сессии, убрать реплику клиента, оставшуюся без ответа)
    выполняется ровно один раз, кто бы ни заметил конец первым: читатель ответа, поток генерации или запрос отмены.
    На завершение генератора ответа полагаться нельзя: при обрыве соединения его закрывает сборщик мусора, когда захочет."""

    def __init__(self, session_id, message_id, busy):
        self.session_id, self.message_id, self.busy = session_id, message_id, busy
        self.events, self.cancelled, self.lock = queue.Queue(), threading.Event(), threading.Lock()
        self.answered = self.closed = False
        self.started = self.alive = time.monotonic()

    def gone(self):
        """Читателя больше нет: клиент нажал отмену или перестал забирать события (закрыл страницу)."""
        return self.cancelled.is_set() or time.monotonic() - self.alive > STREAM_STALE_SEC

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.cancelled.set()
        with _guard_lock:
            if _streams.get(self.session_id) is self:
                del _streams[self.session_id]
        if not self.answered:
            query("DELETE FROM messages WHERE id = ?", self.message_id)
        self.busy.__exit__()


def generate(stream, history, note):
    # Срок ответа хранится в потоке, а куски генератора ответа FastAPI читает из разных потоков пула:
    # вся генерация идёт в одном своём потоке и отдаёт события через очередь.
    llm.set_deadline(ANSWER_BUDGET_SEC)
    try:
        result = answer_stream(state["index"], history, note, lambda kind, value: stream.events.put((kind, value)), stream.gone)
        stream.events.put(("done", result))
    except Exception as exc:
        stream.events.put(("done", fallback(exc)))
    finally:
        llm.set_deadline(None)
    # Читатель мог уйти молча: тогда очистку делает этот поток, как только это станет видно.
    while not stream.closed and not stream.gone():
        time.sleep(0.2)
    stream.close()


def read_stream(stream, links, has_lead, x_debug_token):
    try:
        while True:
            stream.alive = time.monotonic()
            try:
                kind, value = stream.events.get(timeout=1)
            except queue.Empty:
                if stream.closed:
                    return
                if time.monotonic() - stream.started > ANSWER_BUDGET_SEC + 15:
                    # Генерация не уложилась в срок и не сообщила об этом сама: отвечаем заглушкой, а не молчим.
                    kind, value = "done", fallback(TimeoutError())
                else:
                    # Пока модель рассуждает, данных нет: комментарий не даёт промежуточным прокси закрыть соединение.
                    yield ": ping\n\n"
                    continue
            if kind == "done":
                with stream.lock:
                    if stream.closed:
                        return
                    stream.answered = True
                yield sse("done", record(stream.session_id, value, links, has_lead, x_debug_token))
                return
            yield sse("delta", {"text": value}) if kind == "text" else sse("status", {"stage": kind})
    finally:
        stream.close()


@app.post("/api/sessions/{session_id}/messages/stream")
def send_message_stream(session_id: str, message: Message, request: Request, x_debug_token: Annotated[str | None, Header()] = None):
    """Тот же ответ потоком событий: status (searched, thinking), delta (кусок текста), done (как ответ /messages)."""
    require_session(session_id)
    rate_limit(request)
    busy = session_busy(session_id, wait=2.0)
    busy.__enter__()
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    try:
        message_id, manager, has_lead = accept(session_id, message.text)
        if manager:
            busy.__exit__()
            return StreamingResponse(iter([sse("done", {"manager_active": True})]), media_type="text/event-stream", headers=headers)
        history = load_history(session_id, HISTORY_LIMIT)
        links = _pool.submit(site_links.pick, history)
        stream = Stream(session_id, message_id, busy)
        with _guard_lock:
            _streams[session_id] = stream
        threading.Thread(target=generate, args=(stream, history, NOTE_LEAD_DONE if has_lead else NOTE_FORM), daemon=True).start()
    except BaseException:
        busy.__exit__()
        raise
    return StreamingResponse(read_stream(stream, links, has_lead, x_debug_token), media_type="text/event-stream", headers=headers)


@app.post("/api/sessions/{session_id}/cancel")
def cancel_message(session_id: str):
    """Клиент нажал "Отмена": генерация останавливается, его реплика без ответа убирается из истории."""
    with _guard_lock:
        stream = _streams.get(session_id)
    if stream:
        stream.close()
    return {"status": "cancelled" if stream else "idle"}


@app.post("/api/sessions/{session_id}/lead")
def create_lead(session_id: str, lead: Lead, request: Request):
    require_session(session_id)
    rate_limit(request, "leads", 10)
    if not lead.consent:
        raise HTTPException(422, "Нужно согласие на обработку персональных данных")
    phone = normalize_phone(lead.phone)
    if query("SELECT 1 FROM leads WHERE session_id = ?", session_id):
        return {"status": "exists"}
    with session_busy(session_id):
        history = load_history(session_id)
        if not history:
            summary = "Чат пустой."
        else:
            llm.set_deadline(ANSWER_BUDGET_SEC)
            try:
                summary = pipeline.summarize(history)
            except Exception as exc:
                # Заявку нельзя терять из-за сбоя LLM: менеджер увидит диалог в истории.
                print(f"Резюме не построено: {type(exc).__name__}", file=sys.stderr, flush=True)
                summary = "Резюме не сформировано, см. историю диалога."
            finally:
                llm.set_deadline(None)
        now = time.time()
        created = query(
            "INSERT OR IGNORE INTO leads (session_id, name, phone, company, consent_at, summary, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING session_id",
            session_id, lead.name.strip(), phone, lead.company.strip(), now, summary, now,
        )
    return {"status": "created", "summary": summary} if created else {"status": "exists"}


@app.get("/api/admin/sessions")
def list_sessions(x_debug_token: Annotated[str | None, Header()] = None):
    require_debug_access(x_debug_token)
    return query(
        "SELECT s.id, s.created_at, count(m.id) AS messages, l.session_id IS NOT NULL AS has_lead, "
        "EXISTS(SELECT 1 FROM manager_reports mr WHERE mr.session_id=s.id) AS reviewed "
        "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id LEFT JOIN leads l ON l.session_id = s.id "
        "GROUP BY s.id ORDER BY s.created_at DESC LIMIT 500"
    )


@app.get("/api/admin/sessions/{session_id}")
def get_session(session_id: str, x_debug_token: Annotated[str | None, Header()] = None):
    require_debug_access(x_debug_token)
    require_session(session_id)
    leads = query("SELECT * FROM leads WHERE session_id = ?", session_id)
    reports = query("SELECT report,closed_at FROM manager_reports WHERE session_id=?", session_id)
    return {
        "messages": query("SELECT role, content, meta, created_at FROM messages WHERE session_id = ? ORDER BY id", session_id),
        "lead": leads[0] if leads else None,
        "manager_report": reports[0] if reports else None,
    }


@app.get("/api/manager/queue")
def manager_queue(x_debug_token: Annotated[str | None, Header()] = None):
    require_debug_access(x_debug_token)
    return query("SELECT l.* FROM leads l LEFT JOIN manager_leads m ON m.session_id=l.session_id WHERE m.session_id IS NULL ORDER BY l.created_at")


@app.get("/api/manager/active")
def manager_active(x_debug_token: Annotated[str | None, Header()] = None):
    require_debug_access(x_debug_token)
    return query("SELECT l.session_id,l.name,l.company,l.summary,l.created_at, m.content AS last_message,m.created_at AS last_message_at FROM manager_leads a JOIN leads l ON l.session_id=a.session_id LEFT JOIN manager_reports r ON r.session_id=a.session_id LEFT JOIN messages m ON m.id=(SELECT id FROM messages WHERE session_id=l.session_id ORDER BY id DESC LIMIT 1) WHERE r.session_id IS NULL ORDER BY COALESCE(m.created_at,l.created_at) DESC")


@app.post("/api/manager/leads/{session_id}/accept")
def manager_accept(session_id: str, x_debug_token: Annotated[str | None, Header()] = None):
    require_debug_access(x_debug_token)
    if not query("SELECT 1 FROM leads WHERE session_id=?", session_id):
        raise HTTPException(404, "Заявка не найдена")
    query("INSERT OR IGNORE INTO manager_leads(session_id,accepted_at) VALUES(?,?)", session_id, time.time())
    if not query("SELECT 1 FROM messages WHERE session_id=? AND role='system' AND content='Менеджер подключился к чату'", session_id):
        query("INSERT INTO messages(session_id,role,content,created_at) VALUES(?,'system','Менеджер подключился к чату',?)", session_id, time.time())
    return {"status": "active"}


class ManagerReport(BaseModel):
    report: Annotated[str, Field(min_length=1, max_length=5000)]


@app.post("/api/manager/sessions/{session_id}/close")
def manager_close(session_id: str, body: ManagerReport, x_debug_token: Annotated[str | None, Header()] = None):
    require_debug_access(x_debug_token)
    if not query("SELECT 1 FROM manager_leads WHERE session_id=?", session_id):
        raise HTTPException(404, "Активный чат не найден")
    query("INSERT OR IGNORE INTO manager_reports(session_id,report,closed_at) VALUES(?,?,?)", session_id, body.report.strip(), time.time())
    closure_notice = "Менеджер обработал вашу заявку"
    if not query("SELECT 1 FROM messages WHERE session_id=? AND role='system' AND content=?", session_id, closure_notice):
        query("INSERT INTO messages(session_id,role,content,created_at) VALUES(?,'system',?,?)", session_id, closure_notice, time.time())
    return {"status": "closed"}


@app.post("/api/manager/sessions/{session_id}/messages")
def manager_send(session_id: str, message: Message, x_debug_token: Annotated[str | None, Header()] = None):
    require_debug_access(x_debug_token)
    require_session(session_id)
    if not query("SELECT 1 FROM manager_leads WHERE session_id=?", session_id):
        raise HTTPException(409, "Сначала примите заявку")
    if query("SELECT 1 FROM manager_reports WHERE session_id=?", session_id):
        raise HTTPException(409, "Заявка уже закрыта")
    now = time.time()
    query("INSERT INTO messages(session_id,role,content,created_at) VALUES(?,'manager',?,?)", session_id, message.text, now)
    row = query("SELECT id FROM messages WHERE session_id=? AND role='manager' AND created_at=? ORDER BY id DESC LIMIT 1", session_id, now)[0]
    return {"id": row["id"], "content": message.text, "created_at": now}


@app.get("/api/manager/sessions/{session_id}")
def manager_get_session(session_id: str, x_debug_token: Annotated[str | None, Header()] = None):
    require_debug_access(x_debug_token)
    require_session(session_id)
    leads = query("SELECT * FROM leads WHERE session_id = ?", session_id)
    return {
        "messages": query("SELECT id, role, content, meta, created_at FROM messages WHERE session_id = ? ORDER BY id", session_id),
        "lead": leads[0] if leads else None,
    }


@app.get("/api/sessions/{session_id}/manager-messages")
def client_manager_messages(session_id: str, after_id: int = 0):
    require_session(session_id)
    return query("SELECT id,role,content,created_at FROM messages WHERE session_id=? AND role IN ('manager','system') AND id>? ORDER BY id", session_id, after_id)


@app.get("/healthz")
def health():
    return {"status": "ok" if "index" in state else "starting"}


@app.get("/")
def site_page():
    """Демонстрационная страница сайта с виджетом: так чат выглядит у клиента."""
    return FileResponse(STATIC / "site.html")


@app.get("/chat")
def chat_page():
    return FileResponse(STATIC / "index.html")


@app.get("/widget.js")
def widget_script():
    return FileResponse(STATIC / "widget.js", media_type="application/javascript")


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC / "admin.html")


@app.get("/manager")
def manager_page():
    return FileResponse(STATIC / "manager.html")
