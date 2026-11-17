const API_URL =
  window.location.port === "5050"
    ? "/predict"
    : "http://127.0.0.1:5050/predict";

const form = document.getElementById("car-form");
const imageInput = document.getElementById("image-input");
const uploadBox = document.getElementById("upload-box");
const previewWrap = document.getElementById("preview-wrap");
const previewImage = document.getElementById("preview-image");
const message = document.getElementById("message");
const statusBadge = document.getElementById("status-badge");
const brandValue = document.getElementById("brand-value");
const modelValue = document.getElementById("model-value");
const yearValue = document.getElementById("year-value");
const priceValue = document.getElementById("price-value");
const modelEstimateValue = document.getElementById("model-estimate-value");
const fullLabel = document.getElementById("full-label");
const confidenceValue = document.getElementById("confidence-value");
const submitButton = form.querySelector("button");

function setStatus(type, text) {
  statusBadge.className = `status-badge ${type}`;
  statusBadge.textContent = text;
}

function resetResult() {
  brandValue.textContent = "-";
  modelValue.textContent = "-";
  yearValue.textContent = "-";
  priceValue.textContent = "-";
  modelEstimateValue.textContent = "-";
  fullLabel.textContent = "-";
  confidenceValue.textContent = "-";
}

function formatPrice(price) {
  if (price === null || price === undefined || price === "") {
    return "Brak ceny";
  }

  const numericPrice = Number(price);
  if (Number.isNaN(numericPrice)) {
    return String(price);
  }

  return `${numericPrice.toLocaleString("pl-PL")} PLN`;
}

function parseLabel(label) {
  if (!label || typeof label !== "string") {
    return { brand: "-", model: "-", year: "-" };
  }

  const parts = label.trim().split(/\s+/);
  const year = parts.at(-1) && /^\d{4}$/.test(parts.at(-1)) ? parts.pop() : "-";
  const brand = parts.shift() || "-";
  const model = parts.length ? parts.join(" ") : "-";

  return { brand, model, year };
}

function formatConfidence(confidence) {
  if (confidence === null || confidence === undefined || confidence === "") {
    return "Brak danych";
  }

  const numericConfidence = Number(confidence);
  if (Number.isNaN(numericConfidence)) {
    return String(confidence);
  }

  return `${numericConfidence.toFixed(2)}%`;
}

function updateResult(data) {
  const label = data.label || data.prediction || "-";
  const parsed = parseLabel(label);

  brandValue.textContent = data.brand || parsed.brand;
  modelValue.textContent = data.model || parsed.model;
  yearValue.textContent = data.year || parsed.year;
  priceValue.textContent = formatPrice(data.price_reference_table);
  modelEstimateValue.textContent = formatPrice(data.price_model_estimate);
  fullLabel.textContent = label;
  confidenceValue.textContent = formatConfidence(data.confidence);
}

function showPreview(file) {
  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  previewWrap.hidden = false;
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) {
    previewWrap.hidden = true;
    return;
  }

  showPreview(file);
  message.textContent = "Zdjęcie gotowe do analizy. Kliknij przycisk, aby rozpocząć rozpoznawanie.";
  setStatus("neutral", "Gotowe");
  resetResult();
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadBox.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadBox.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadBox.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadBox.classList.remove("dragover");
  });
});

uploadBox.addEventListener("drop", (event) => {
  const droppedFile = event.dataTransfer.files[0];
  if (!droppedFile) {
    return;
  }

  imageInput.files = event.dataTransfer.files;
  showPreview(droppedFile);
  message.textContent = "Zdjęcie gotowe do analizy. Kliknij przycisk, aby rozpocząć rozpoznawanie.";
  setStatus("neutral", "Gotowe");
  resetResult();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files[0];
  if (!file) {
    setStatus("error", "Brak pliku");
    message.textContent = "Najpierw dodaj zdjęcie samochodu.";
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  submitButton.disabled = true;
  setStatus("loading", "Analiza");
  message.textContent = "Trwa wysyłanie zdjęcia do modelu i pobieranie wyniku.";
  resetResult();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Błąd serwera: ${response.status}`);
    }

    const data = await response.json();
    updateResult(data);

    setStatus("success", "Gotowe");
    message.textContent = "Zdjęcie zostało przeanalizowane pomyślnie.";
  } catch (error) {
    setStatus("error", "Backend");
    message.textContent =
      "Frontend działa poprawnie, ale backend Pythona nie jest jeszcze podpięty lub nie odpowiada pod adresem http://127.0.0.1:5050/predict.";
    fullLabel.textContent = "Brak odpowiedzi z modelu";
    console.error(error);
  } finally {
    submitButton.disabled = false;
  }
});
