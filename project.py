from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from cars_with_prices import CAR_PRICES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "stanford-car-vit"
DEFAULT_IMAGE_PATH = BASE_DIR / "image" / "car.jpg"

_processor = None
_model = None


@dataclass
class PredictionResult:
    label: str
    brand: str
    model: str
    year: str
    confidence: float
    price_from_table: int | None

    def to_dict(self) -> Dict[str, int | str | None]:
        return {
            "label": self.label,
            "brand": self.brand,
            "model": self.model,
            "year": self.year,
            "confidence": self.confidence,
            "price_from_table": self.price_from_table,
        }


def load_car_model() -> Tuple[AutoImageProcessor, AutoModelForImageClassification]:
    global _processor, _model

    if _processor is None or _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)

    return _processor, _model


def parse_label(label: str) -> Tuple[str, str, str]:
    parts = label.strip().split()
    if not parts:
        return "-", "-", "-"

    year = parts.pop() if parts and parts[-1].isdigit() and len(parts[-1]) == 4 else "-"
    brand = parts.pop(0) if parts else "-"
    model = " ".join(parts) if parts else "-"
    return brand, model, year


def _open_image(image_source: str | Path | bytes | BinaryIO) -> Image.Image:
    if isinstance(image_source, (str, Path)):
        image = Image.open(image_source)
    elif isinstance(image_source, bytes):
        image = Image.open(BytesIO(image_source))
    else:
        image = Image.open(image_source)

    return image.convert("RGB")


def predict_car(image_source: str | Path | bytes | BinaryIO) -> PredictionResult:
    processor, model = load_car_model()
    image = _open_image(image_source)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_idx = logits.argmax(-1).item()
    confidence = probabilities[0, predicted_class_idx].item() * 100
    label = model.config.id2label[predicted_class_idx]
    brand, model_name, year = parse_label(label)
    price = CAR_PRICES.get(label)

    return PredictionResult(
        label=label,
        brand=brand,
        model=model_name,
        year=year,
        confidence=round(confidence, 2),
        price_from_table=price,
    )


def main() -> None:
    result = predict_car(DEFAULT_IMAGE_PATH)

    print("Samochód:", result.label)
    if result.price_from_table is not None:
        print(f"Szacowana cena: {result.price_from_table:,} PLN")
    else:
        print("Brak ceny dla tego modelu")


if __name__ == "__main__":
    main()
