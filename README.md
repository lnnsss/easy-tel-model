## 1) Установка

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2.1) Запуск

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 2.2) Запуск с перезагрузкой

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```