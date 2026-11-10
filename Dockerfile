FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt ./
COPY core/requirements.txt core/
RUN pip install --no-cache-dir -r requirements.txt

COPY core core
COPY web web
COPY tests/fake_llm.py tests/fake_llm.py

RUN useradd --create-home app && mkdir /data && chown app /data
USER app

EXPOSE 8000
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
