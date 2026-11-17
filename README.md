# Car Recognition & Price Estimation

A small web app that classifies a car's make/model/year from a photo and estimates
its price two different ways: an exact lookup against a reference price table, and
an experimental regression model trained on that same table.

![App UI](docs/images/ui-screenshot.jpg)

## What it does

1. You upload a photo of a car through the web UI.
2. A Vision Transformer (ViT) image classifier predicts one of 196 car
   make/model/year labels (the [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
   label set).
3. The predicted label is priced two ways:
   - **Reference table** — an exact lookup in a hand-compiled price table
     (`cars_with_prices.py`), 196 labels → PLN price.
   - **Regression model estimate** — a custom weighted k-nearest-neighbours
     regressor (`price_model.py`) trained on the same table, using brand,
     body type, year, and model-name token overlap as features.

## Why two prices?

The table lookup is exact but only works for the 196 labels it already contains.
The regression model is the actual "trained" ML component — it's kept and shown
separately, labeled as experimental, rather than hidden, because its real accuracy
(below) is modest and shouldn't be presented as equivalent to the table value.

## Data

- **Image classifier**: a pretrained ViT checkpoint (`google/vit`-family,
  `ViTForImageClassification`, 196-way classification head) adapted for this
  project — **not trained from scratch here**. The checkpoint's `config.json`
  carries no recorded source (`_name_or_path` is empty), so if you're asking
  about this in an interview, be upfront that the classifier was reused/adapted,
  not trained end-to-end in this repo. The original ML work in this repo is the
  price regression model and the app itself.
- **Price table** (`cars_with_prices.py`): 196 hand-compiled label → price (PLN)
  entries. The exact sourcing isn't documented in the original project files —
  treat these as illustrative reference prices, not live market data, unless you
  can cite where they came from.
- **Model weights are not included in this repo** (~330 MB) — see Quick start.
- **Sample car photos used for manual testing are not included** either —
  they were pulled from various web sources for local testing and their
  redistribution rights aren't established.

## Approach

```
photo --> ViT image classifier --> label (e.g. "BMW M3 Coupe 2012")
                                      |--> exact price lookup (CAR_PRICES table)
                                      '--> price_model.py KNN regressor estimate
```

The regressor (`CustomPriceRegressor` in `price_model.py`) finds the 5 nearest
rows in the price table by a hand-weighted distance over brand, body type, year
gap, and model-name token overlap, then blends the neighbour-weighted price
(70%), the brand average (20%), and the body-type average (10%).

## Results

From `artifacts/price_model_metrics.json`, reproducible by running
`python train_price_model.py` (fixed random seed, 156 train / 40 test rows,
80/20 split of the 196-row price table):

| | MAE (PLN) | MAPE |
|---|---|---|
| **Price model** | **51,732** | **90.3%** |
| Naive baseline — always predict the mean | 258,934 | 651.1% |
| Naive baseline — always predict the median | 131,709 | **44.1%** |

The model clearly beats a naive mean predictor. It also beats a naive median
predictor on MAE — but it's *worse* than that same naive median on MAPE. That's
a genuine, reproducible result, not a cherry-picked one: the model does better
on absolute errors (it gets expensive/cheap cars closer in PLN terms) but worse
on relative errors (its misses are proportionally larger on cheaper cars than
"just guess the median" would be). The baseline numbers above aren't in the
original project artifacts — they were computed with the same train/test split
to give the MAE/MAPE values a point of comparison.

### Classifier, on photos it has never seen

Run manually against 9 real car photos (not from the Stanford Cars dataset,
not included in this repo):

| Prediction | Confidence |
|---|---|
| BMW M3 Coupe 2012 | 99.8% |
| Buick Enclave SUV 2012 | 99.0% |
| Chevrolet Impala Sedan 2007 | 98.1% |
| Rolls-Royce Phantom Sedan 2012 | 99.7% |
| Jeep Grand Cherokee SUV 2012 | 86.0% |
| Mitsubishi Lancer Sedan 2012 | 89.8% |
| Audi S5 photo → predicted **Audi TTS Coupe 2012** | 84.5% |
| Audi S5 photo → predicted **Audi S5 Coupe 2012** | 51.2% |
| Audi S5 photo → predicted **Audi TTS Coupe 2012** | 35.1% |

2 of 3 real Audi S5 photos were misclassified as Audi TTS (a plausible mix-up,
also partly because "S5 Sportback" isn't one of the 196 trained classes) — kept
here deliberately as an honest failure case rather than only showing confident,
correct predictions.

## Quick start

```bash
git clone <this-repo-url>
cd car-computer-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repo does **not** include the classifier's weights (~330 MB) or sample
images. Place your own fine-tuned/obtained checkpoint at:

```
models/stanford-car-vit/
  config.json
  preprocessor_config.json
  model.safetensors
```

Then run the app and open `http://localhost:5050`:

```bash
python app.py
```

To retrain the price regression model and regenerate `artifacts/`:

```bash
python train_price_model.py
```

## Repository structure

```
app.py                  Flask API (serves the UI, /health, /predict)
project.py               Loads the ViT classifier and runs inference
price_model.py           Custom KNN price regressor + training/eval
train_price_model.py     Retrains the price model, writes artifacts/
cars_with_prices.py      196-entry label -> price (PLN) reference table
index.html / script.js / styles.css   Frontend
artifacts/                Reproducible training outputs (model + metrics JSON)
models/      (gitignored) Classifier weights — not included, see Quick start
image/       (gitignored) Local test photos — not included
```

## Limitations

- The image classifier was adapted, not trained from scratch in this repo —
  there's no training script or evaluation log for it here, only the manual
  spot-check above.
- The price regression model's MAPE (90.3%) is worse than a naive
  "always predict the median" baseline (44.1%) — it should be described
  honestly as an experimental/weak estimator, not a production-ready model.
- The price table's source and currency conversion (PLN) aren't documented
  beyond what's in this README.
- This is a local development server (`debug=False` by default, but still
  Flask's built-in dev server) — not configured for production deployment.

## License

No license file is included; all rights reserved by default. Contact the
author before reusing any part of this project.
