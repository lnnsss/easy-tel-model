# Запуск FastAPI сервиса с моделью ResNet-50

## 1. Подготовка окружения

Создать и активировать виртуальное окружение:

```bash
python3 -m venv venv
source venv/bin/activate   # Mac / Linux
venv\Scripts\activate    # Windows
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

---

## 2. Запуск локального сервера

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 3. Режим разработки (авто-перезапуск)

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## 4. Проверка работы

Открыть в браузере:

```
http://localhost:8000/docs
```