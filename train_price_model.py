from __future__ import annotations

import json

from price_model import ARTIFACTS_DIR, MODEL_ARTIFACT_PATH, train_price_model


def main() -> None:
    model, metrics = train_price_model()
    model.save(MODEL_ARTIFACT_PATH)

    report_path = ARTIFACTS_DIR / "price_model_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "train_size": metrics.train_size,
                "test_size": metrics.test_size,
                "mae": metrics.mae,
                "mape_percent": metrics.mape_percent,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Model ceny zapisany w: {MODEL_ARTIFACT_PATH}")
    print(f"Raport metryk zapisany w: {report_path}")
    print(f"MAE: {metrics.mae}")
    print(f"MAPE: {metrics.mape_percent}%")


if __name__ == "__main__":
    main()
