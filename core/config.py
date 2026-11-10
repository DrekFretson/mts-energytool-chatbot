"""Все ручки эксперимента в одном месте. Один сабмит = одна правка здесь или в prompts.py."""

MAX_CHUNK_CHARS = 1500  # bge-m3 видит ~512 токенов, длиннее режем по абзацам

TOP_K = 5
CANDIDATES = 20
DENSE_WEIGHT = 0.85
HISTORY_WEIGHT_DENSE = 0.3
HISTORY_WEIGHT_BM25 = 0.5

USE_RERANK = True
RERANK_TIMEOUT = 3.0

THINKING_OFF = True
TEMPERATURE = 0.0  # намеренно не берём TEMPERATURE платформы: нужна воспроизводимость
MAX_ANSWER_TOKENS = 400
LLM_TIMEOUT = 60.0

DEADLINE_SEC = 18 * 60  # лимит платформы 20 минут, 2 минуты запаса

FALLBACK_ANSWER = (
    "Точного ответа в материалах сейчас не нашёл. "
    "Лучше уточнить это у специалиста МТС EnergyTool: он разберёт вашу ситуацию отдельно."
)
