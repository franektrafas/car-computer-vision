from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Sequence, Tuple

from cars_with_prices import CAR_PRICES

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "price_model.json"


def parse_label(label: str) -> Dict[str, object]:
    parts = label.strip().split()
    year = int(parts[-1]) if parts and parts[-1].isdigit() else 0
    if year:
        parts = parts[:-1]

    brand = parts[0] if parts else "-"
    body_type = parts[-1] if len(parts) > 1 else "-"
    model = " ".join(parts[1:]) if len(parts) > 1 else "-"
    model_tokens = [token.lower() for token in model.split() if token]

    return {
        "label": label,
        "brand": brand,
        "model": model,
        "body_type": body_type,
        "year": year,
        "model_tokens": model_tokens,
    }


@dataclass
class PriceTrainingRow:
    label: str
    brand: str
    model: str
    body_type: str
    year: int
    price: int
    model_tokens: List[str]


@dataclass
class PriceModelMetrics:
    train_size: int
    test_size: int
    mae: float
    mape_percent: float


class CustomPriceRegressor:
    def __init__(self, neighbors: int = 5):
        self.neighbors = neighbors
        self.rows: List[PriceTrainingRow] = []
        self.brand_average: Dict[str, float] = {}
        self.body_average: Dict[str, float] = {}
        self.global_average: float = 0.0

    def fit(self, rows: Sequence[PriceTrainingRow]) -> None:
        self.rows = list(rows)
        if not self.rows:
            raise ValueError("Brak danych treningowych dla modelu ceny.")

        self.global_average = sum(row.price for row in self.rows) / len(self.rows)
        self.brand_average = _mean_by_key(self.rows, lambda row: row.brand)
        self.body_average = _mean_by_key(self.rows, lambda row: row.body_type)

    def predict_from_label(self, label: str) -> int:
        parsed = parse_label(label)
        return self.predict_features(
            brand=str(parsed["brand"]),
            model=str(parsed["model"]),
            body_type=str(parsed["body_type"]),
            year=int(parsed["year"]),
            model_tokens=list(parsed["model_tokens"]),
        )

    def predict_features(
        self,
        brand: str,
        model: str,
        body_type: str,
        year: int,
        model_tokens: Sequence[str] | None = None,
    ) -> int:
        if not self.rows:
            raise ValueError("Model ceny nie został jeszcze wytrenowany.")

        model_tokens = list(model_tokens or [token.lower() for token in model.split() if token])
        scored_rows = sorted(
            ((self._distance(brand, body_type, year, model_tokens, row), row) for row in self.rows),
            key=lambda item: item[0],
        )

        nearest = scored_rows[: self.neighbors]
        weighted_sum = 0.0
        total_weight = 0.0

        for distance, row in nearest:
            weight = 1.0 / (distance + 0.001)
            weighted_sum += row.price * weight
            total_weight += weight

        neighbor_prediction = weighted_sum / total_weight if total_weight else self.global_average
        brand_baseline = self.brand_average.get(brand, self.global_average)
        body_baseline = self.body_average.get(body_type, self.global_average)

        final_prediction = (neighbor_prediction * 0.7) + (brand_baseline * 0.2) + (body_baseline * 0.1)
        return max(10000, round(final_prediction))

    def _distance(
        self,
        brand: str,
        body_type: str,
        year: int,
        model_tokens: Sequence[str],
        row: PriceTrainingRow,
    ) -> float:
        distance = 0.0

        if brand != row.brand:
            distance += 3.5

        if body_type != row.body_type:
            distance += 1.8

        year_gap = abs(year - row.year)
        distance += year_gap / 4 if year and row.year else 1.5

        overlap = len(set(model_tokens) & set(row.model_tokens))
        total_tokens = len(set(model_tokens) | set(row.model_tokens))
        similarity = overlap / total_tokens if total_tokens else 0.0
        distance += (1.0 - similarity) * 4.0

        return distance

    def save(self, path: Path = MODEL_ARTIFACT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "neighbors": self.neighbors,
            "global_average": self.global_average,
            "brand_average": self.brand_average,
            "body_average": self.body_average,
            "rows": [asdict(row) for row in self.rows],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = MODEL_ARTIFACT_PATH) -> "CustomPriceRegressor":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(neighbors=payload["neighbors"])
        model.global_average = payload["global_average"]
        model.brand_average = payload["brand_average"]
        model.body_average = payload["body_average"]
        model.rows = [PriceTrainingRow(**row) for row in payload["rows"]]
        return model


def _mean_by_key(rows: Sequence[PriceTrainingRow], key_fn) -> Dict[str, float]:
    buckets: Dict[str, List[int]] = {}

    for row in rows:
        key = key_fn(row)
        buckets.setdefault(key, []).append(row.price)

    return {key: sum(values) / len(values) for key, values in buckets.items()}


def build_training_rows(price_map: Dict[str, int] | None = None) -> List[PriceTrainingRow]:
    rows = []
    source = price_map or CAR_PRICES

    for label, price in source.items():
        parsed = parse_label(label)
        rows.append(
            PriceTrainingRow(
                label=label,
                brand=str(parsed["brand"]),
                model=str(parsed["model"]),
                body_type=str(parsed["body_type"]),
                year=int(parsed["year"]),
                price=price,
                model_tokens=list(parsed["model_tokens"]),
            )
        )

    return rows


def train_price_model(rows: Sequence[PriceTrainingRow] | None = None) -> Tuple[CustomPriceRegressor, PriceModelMetrics]:
    dataset = list(rows or build_training_rows())
    if len(dataset) < 10:
        raise ValueError("Za mało danych do treningu modelu ceny.")

    train_rows, test_rows = split_dataset(dataset, seed=42, test_ratio=0.2)
    regressor = CustomPriceRegressor(neighbors=5)
    regressor.fit(train_rows)

    absolute_errors: List[float] = []
    percentage_errors: List[float] = []

    for row in test_rows:
        prediction = regressor.predict_features(
            brand=row.brand,
            model=row.model,
            body_type=row.body_type,
            year=row.year,
            model_tokens=row.model_tokens,
        )
        absolute_error = abs(prediction - row.price)
        absolute_errors.append(absolute_error)
        if row.price:
            percentage_errors.append((absolute_error / row.price) * 100)

    metrics = PriceModelMetrics(
        train_size=len(train_rows),
        test_size=len(test_rows),
        mae=round(sum(absolute_errors) / len(absolute_errors), 2),
        mape_percent=round(sum(percentage_errors) / len(percentage_errors), 2),
    )

    final_model = CustomPriceRegressor(neighbors=5)
    final_model.fit(dataset)
    return final_model, metrics


def split_dataset(
    rows: Sequence[PriceTrainingRow],
    *,
    seed: int,
    test_ratio: float,
) -> Tuple[List[PriceTrainingRow], List[PriceTrainingRow]]:
    shuffled = list(rows)
    Random(seed).shuffle(shuffled)

    test_size = max(1, math.ceil(len(shuffled) * test_ratio))
    test_rows = shuffled[:test_size]
    train_rows = shuffled[test_size:]
    return train_rows, test_rows


def ensure_price_model(path: Path = MODEL_ARTIFACT_PATH) -> Tuple[CustomPriceRegressor, PriceModelMetrics | None]:
    if path.exists():
        return CustomPriceRegressor.load(path), None

    model, metrics = train_price_model()
    model.save(path)
    return model, metrics
