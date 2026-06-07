import asyncio
import io
import os
import time
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoTokenizer,
)

# FastAPI-приложение поднимает отдельный ML-сервис.
# Node backend обращается сюда по HTTP: /chat для AI-тьютора и /predict для распознавания фото.
app = FastAPI()


# Блок компьютерного зрения загружается сразу при старте сервиса.
# CV_MODEL_NAME можно переопределить через env, но по умолчанию используется ResNet-50.
CV_MODEL_NAME = os.getenv("CV_MODEL_NAME", "microsoft/resnet-50")
# Возвращаем не меньше 5 и не больше 20 вариантов, даже если env задан слишком маленьким или большим.
CV_TOP_K = min(max(int(os.getenv("CV_TOP_K", "10")), 5), 20)
# processor превращает PIL-картинку в тензоры нужного размера и формата для модели.
cv_processor = AutoImageProcessor.from_pretrained(CV_MODEL_NAME)
# cv_model классифицирует изображение и возвращает вероятности ImageNet-классов.
cv_model = AutoModelForImageClassification.from_pretrained(CV_MODEL_NAME)
# eval выключает режим обучения: dropout и другие training-механизмы не нужны для предсказаний.
cv_model.eval()


# Блок чат-модели. Значения из env позволяют менять модель и лимиты без правки кода.
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
# Мягкий лимит истории: перед генерацией старые сообщения будут обрезаны по количеству символов.
CHAT_MAX_CONTEXT_CHARS = max(int(os.getenv("CHAT_MAX_CONTEXT_CHARS", "12000")), 2000)
# Сколько новых токенов модель генерирует по умолчанию, если backend не передал свой лимит.
CHAT_MAX_NEW_TOKENS_DEFAULT = max(int(os.getenv("CHAT_MAX_NEW_TOKENS_DEFAULT", "256")), 64)
# Верхняя граница для max_new_tokens в запросе, чтобы случайно не запустить слишком долгую генерацию.
CHAT_MAX_NEW_TOKENS_LIMIT = max(int(os.getenv("CHAT_MAX_NEW_TOKENS_LIMIT", "512")), CHAT_MAX_NEW_TOKENS_DEFAULT)
# temperature управляет случайностью ответа: 0 ближе к детерминированному ответу, больше 0 дает вариативность.
CHAT_TEMPERATURE_DEFAULT = float(os.getenv("CHAT_TEMPERATURE_DEFAULT", "0.6"))
# Таймаут защищает backend и пользователя от бесконечно долгой генерации на слабом железе.
CHAT_TIMEOUT_SEC = max(int(os.getenv("CHAT_TIMEOUT_SEC", "50")), 5)


# Глобальное состояние чат-модели. Она загружается не при старте, а при первом POST /chat.
chat_tokenizer = None
chat_model = None
# Если доступна CUDA, модель будет работать на видеокарте и в float16; иначе на CPU и в float32.
chat_device = "cuda" if torch.cuda.is_available() else "cpu"
chat_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
chat_ready = False
# Lock нужен, чтобы два одновременных первых запроса не начали загружать одну модель два раза.
chat_init_lock = Lock()


class ChatMessage(BaseModel):
    # Один элемент истории диалога: системная инструкция, сообщение пользователя или ответ ассистента.
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        # Pydantic пропускает сюда строку до передачи в бизнес-логику; пустые сообщения отсекаем сразу.
        text = value.strip()
        if not text:
            raise ValueError("content must not be empty")
        return text


class ChatRequest(BaseModel):
    # Контракт POST /chat: backend присылает историю, режим и необязательные параметры генерации.
    messages: List[ChatMessage] = Field(default_factory=list, min_length=1, max_length=40)
    mode: Optional[Literal["tutor", "translate", "correct"]] = "tutor"
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_new_tokens: Optional[int] = Field(default=None, ge=32, le=CHAT_MAX_NEW_TOKENS_LIMIT)


def _initialize_chat_model() -> None:
    """Лениво загружает tokenizer и Qwen-модель перед первым запросом чата."""
    global chat_tokenizer, chat_model, chat_ready
    if chat_ready:
        return

    with chat_init_lock:
        # Повторная проверка внутри lock закрывает гонку между параллельными первыми запросами.
        if chat_ready:
            return

        print(f"Loading chat model: {CHAT_MODEL_NAME} on {chat_device}")
        # tokenizer знает формат prompt'а конкретной instruct-модели и превращает текст в токены.
        chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME, trust_remote_code=True)
        # low_cpu_mem_usage уменьшает пик потребления памяти во время загрузки весов.
        chat_model = AutoModelForCausalLM.from_pretrained(
            CHAT_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=chat_dtype,
            low_cpu_mem_usage=True
        )
        # Переносим модель на выбранное устройство и переводим ее в режим инференса.
        chat_model.to(chat_device)
        chat_model.eval()
        chat_ready = True
        print("Chat model loaded")


def _trim_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Оставляет последние сообщения, которые помещаются в мягкий лимит контекста."""
    total = 0
    kept: List[ChatMessage] = []
    # Идем с конца, потому что последние сообщения важнее старых для текущего ответа.
    for message in reversed(messages):
        size = len(message.content)
        if kept and total + size > CHAT_MAX_CONTEXT_CHARS:
            break
        kept.append(message)
        total += size
    return list(reversed(kept))


def _generate_reply(payload: ChatRequest) -> Dict[str, Any]:
    """Готовит prompt, запускает локальную модель и возвращает текст ответа с метриками."""
    if not chat_ready:
        _initialize_chat_model()

    # После _initialize_chat_model эти объекты должны быть заполнены; assert помогает типам и раннему падению.
    assert chat_tokenizer is not None
    assert chat_model is not None

    messages = _trim_messages(payload.messages)
    if not any(message.role == "user" for message in messages):
        raise HTTPException(status_code=400, detail="At least one user message is required")

    # Transformers ожидает историю в формате role/content, а chat template добавляет спец-токены модели.
    conversation = [{"role": item.role, "content": item.content} for item in messages]
    prompt_text = chat_tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    # Токенизируем prompt и переносим входные тензоры на то же устройство, где лежит модель.
    model_inputs = chat_tokenizer(prompt_text, return_tensors="pt").to(chat_device)

    max_new_tokens = payload.max_new_tokens or CHAT_MAX_NEW_TOKENS_DEFAULT
    temperature = payload.temperature if payload.temperature is not None else CHAT_TEMPERATURE_DEFAULT
    do_sample = temperature > 0.0

    started_at = time.time()
    with torch.no_grad():
        # generate продолжает prompt новыми токенами; no_grad экономит память, так как обучение не идет.
        output_ids = chat_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=chat_tokenizer.eos_token_id
        )
    elapsed_ms = int((time.time() - started_at) * 1000)

    # В output_ids лежит и исходный prompt, и новый ответ; срезом оставляем только сгенерированную часть.
    generated_ids = output_ids[0][model_inputs["input_ids"].shape[-1]:]
    reply = chat_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Ответ возвращается backend'у: сам текст, имя модели, время генерации и примерная статистика токенов.
    return {
        "reply": reply,
        "model": CHAT_MODEL_NAME,
        "timingMs": elapsed_ms,
        "usage": {
            "promptTokens": int(model_inputs["input_ids"].shape[-1]),
            "completionTokens": int(generated_ids.shape[-1])
        }
    }


@app.get("/health")
def health():
    """Возвращает статус сервиса и показывает, какая ML-конфигурация сейчас активна."""
    return {
        "ok": True,
        "chatReady": chat_ready,
        "chatModel": CHAT_MODEL_NAME,
        "chatDevice": chat_device,
        "cvModel": CV_MODEL_NAME
    }


@app.post("/chat")
async def chat(payload: ChatRequest):
    """HTTP endpoint для AI-чата; тяжелая генерация вынесена в отдельный поток с таймаутом."""
    try:
        # to_thread не блокирует event loop FastAPI, пока PyTorch считает ответ.
        return await asyncio.wait_for(
            asyncio.to_thread(_generate_reply, payload),
            timeout=CHAT_TIMEOUT_SEC
        )
    except asyncio.TimeoutError as err:
        raise HTTPException(status_code=504, detail="Generation timeout") from err
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {err}") from err


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """HTTP endpoint для распознавания фото; принимает файл и возвращает top-k классов."""
    # FastAPI отдает файл как UploadFile, поэтому сначала читаем байты и открываем их через PIL.
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # processor делает resize/normalize и собирает torch tensors в формате, который ожидает ResNet.
    inputs = cv_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = cv_model(**inputs)

    # logits превращаем в вероятности и берем самые вероятные CV_TOP_K классов.
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    topk = probs.topk(CV_TOP_K)

    results = []
    for score, idx in zip(topk.values[0], topk.indices[0]):
        # id2label переводит числовой индекс класса ImageNet в человекочитаемую подпись.
        label = cv_model.config.id2label[idx.item()]
        results.append({
            "label": label,
            "score": float(score)
        })

    return results
