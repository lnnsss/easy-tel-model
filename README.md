# ML service (FastAPI): CV + AI Chat (Qwen2.5-1.5B)

This service now exposes:
- `POST /predict` (legacy image classification)
- `POST /chat` (local LLM chat for татарский tutor)
- `GET /health`

## 1) Setup

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

or with reload:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 3) Optional env vars

- `CHAT_MODEL_NAME` (default: `Qwen/Qwen2.5-1.5B-Instruct`)
- `CHAT_TIMEOUT_SEC` (default: `50`)
- `CHAT_MAX_CONTEXT_CHARS` (default: `12000`)
- `CHAT_MAX_NEW_TOKENS_DEFAULT` (default: `256`)
- `CHAT_MAX_NEW_TOKENS_LIMIT` (default: `512`)
- `CHAT_TEMPERATURE_DEFAULT` (default: `0.6`)
- `CV_MODEL_NAME` (default: `microsoft/resnet-50`)

## 4) Quick check

- Open docs: `http://localhost:8000/docs`
- Health: `GET /health`
- Chat: `POST /chat` with:

```json
{
  "messages": [
    { "role": "system", "content": "You are a татарский tutor." },
    { "role": "user", "content": "Привет, научи меня здороваться по-татарски." }
  ],
  "temperature": 0.6,
  "max_new_tokens": 180
}
```
