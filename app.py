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

app = FastAPI()


# Старый модуль компьютерного зрения с уже существующей функциональностью
CV_MODEL_NAME = os.getenv("CV_MODEL_NAME", "microsoft/resnet-50")
CV_TOP_K = min(max(int(os.getenv("CV_TOP_K", "10")), 5), 20)
cv_processor = AutoImageProcessor.from_pretrained(CV_MODEL_NAME)
cv_model = AutoModelForImageClassification.from_pretrained(CV_MODEL_NAME)
cv_model.eval()


# Конфигурация языковой модели
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
CHAT_MAX_CONTEXT_CHARS = max(int(os.getenv("CHAT_MAX_CONTEXT_CHARS", "12000")), 2000)
CHAT_MAX_NEW_TOKENS_DEFAULT = max(int(os.getenv("CHAT_MAX_NEW_TOKENS_DEFAULT", "256")), 64)
CHAT_MAX_NEW_TOKENS_LIMIT = max(int(os.getenv("CHAT_MAX_NEW_TOKENS_LIMIT", "512")), CHAT_MAX_NEW_TOKENS_DEFAULT)
CHAT_TEMPERATURE_DEFAULT = float(os.getenv("CHAT_TEMPERATURE_DEFAULT", "0.6"))
CHAT_TIMEOUT_SEC = max(int(os.getenv("CHAT_TIMEOUT_SEC", "50")), 5)


chat_tokenizer = None
chat_model = None
chat_device = "cuda" if torch.cuda.is_available() else "cpu"
chat_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
chat_ready = False
chat_init_lock = Lock()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("content must not be empty")
        return text


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list, min_length=1, max_length=40)
    mode: Optional[Literal["tutor", "translate", "correct"]] = "tutor"
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_new_tokens: Optional[int] = Field(default=None, ge=32, le=CHAT_MAX_NEW_TOKENS_LIMIT)


# Лениво загружает языковую модель перед первым запросом чата.
def _initialize_chat_model() -> None:
    global chat_tokenizer, chat_model, chat_ready
    if chat_ready:
        return

    with chat_init_lock:
        if chat_ready:
            return

        print(f"Loading chat model: {CHAT_MODEL_NAME} on {chat_device}")
        chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME, trust_remote_code=True)
        chat_model = AutoModelForCausalLM.from_pretrained(
            CHAT_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=chat_dtype,
            low_cpu_mem_usage=True
        )
        chat_model.to(chat_device)
        chat_model.eval()
        chat_ready = True
        print("Chat model loaded")


# Обрезает историю сообщений до безопасного лимита контекста модели.
def _trim_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    # Оставляем хвост, который помещается в мягкий лимит символов
    total = 0
    kept: List[ChatMessage] = []
    for message in reversed(messages):
        size = len(message.content)
        if kept and total + size > CHAT_MAX_CONTEXT_CHARS:
            break
        kept.append(message)
        total += size
    return list(reversed(kept))


# Готовит prompt и генерирует ответ локальной языковой модели.
def _generate_reply(payload: ChatRequest) -> Dict[str, Any]:
    if not chat_ready:
        _initialize_chat_model()

    assert chat_tokenizer is not None
    assert chat_model is not None

    messages = _trim_messages(payload.messages)
    if not any(message.role == "user" for message in messages):
        raise HTTPException(status_code=400, detail="At least one user message is required")

    conversation = [{"role": item.role, "content": item.content} for item in messages]
    prompt_text = chat_tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = chat_tokenizer(prompt_text, return_tensors="pt").to(chat_device)

    max_new_tokens = payload.max_new_tokens or CHAT_MAX_NEW_TOKENS_DEFAULT
    temperature = payload.temperature if payload.temperature is not None else CHAT_TEMPERATURE_DEFAULT
    do_sample = temperature > 0.0

    started_at = time.time()
    with torch.no_grad():
        output_ids = chat_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=chat_tokenizer.eos_token_id
        )
    elapsed_ms = int((time.time() - started_at) * 1000)

    generated_ids = output_ids[0][model_inputs["input_ids"].shape[-1]:]
    reply = chat_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
# Возвращает статус ML-сервиса и загруженных моделей.
def health():
    return {
        "ok": True,
        "chatReady": chat_ready,
        "chatModel": CHAT_MODEL_NAME,
        "chatDevice": chat_device,
        "cvModel": CV_MODEL_NAME
    }


@app.post("/chat")
async def chat(payload: ChatRequest):
    try:
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
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = cv_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = cv_model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    topk = probs.topk(CV_TOP_K)

    results = []
    for score, idx in zip(topk.values[0], topk.indices[0]):
        label = cv_model.config.id2label[idx.item()]
        results.append({
            "label": label,
            "score": float(score)
        })

    return results
