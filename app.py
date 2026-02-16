from fastapi import FastAPI, File, UploadFile
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io

app = FastAPI()

print("🔄 Загружаем модель ResNet-50...")

MODEL_NAME = "microsoft/resnet-50"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

print("✅ Модель загружена")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)

    top5 = probs.topk(5)

    results = []
    for score, idx in zip(top5.values[0], top5.indices[0]):
        label = model.config.id2label[idx.item()]
        results.append({
            "label": label,
            "score": float(score)
        })

    return results
