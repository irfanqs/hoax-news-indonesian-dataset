%%writefile app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    MBartForConditionalGeneration,
    AutoModelForSequenceClassification
)
import torch
import torch.nn.functional as F
import spacy

# ── Konfigurasi ──────────────────────────────────────────────
BART_MODEL_DIR = "/content/drive/MyDrive/mbart_models/mbart_indosum_final"
BERT_MODEL_DIR = "/content/drive/MyDrive/indobert-hoax-model/checkpoint-7185"
BERT_TOKENIZER_DIR = "indobenchmark/indobert-base-p1"  # tokenizer dari HuggingFace

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load BART (Summarization) ─────────────────────────────────
print("Loading BART tokenizer...")
bart_tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_DIR, use_fast=True)
bart_tokenizer.src_lang = "id_ID"

print("Loading BART model...")
bart_model = MBartForConditionalGeneration.from_pretrained(BART_MODEL_DIR)
bart_model.to(device)
bart_model.eval()
print(f"BART loaded on {device}")

# ── Load BERT (Classification) ────────────────────────────────
print("Loading BERT tokenizer...")
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER_DIR)

print("Loading BERT model...")
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
bert_model.to(device)
bert_model.eval()

# Label mapping dari config BERT
id2label = bert_model.config.id2label  # e.g. {0: "LABEL_0", 1: "LABEL_1"}
print(f"BERT loaded on {device}, labels: {id2label}")

# ── Load spaCy (NER) ──────────────────────────────────────────
print("Loading spaCy model...")
try:
    nlp = spacy.load("id_core_news_sm")
    print("spaCy loaded: id_core_news_sm")
except OSError:
    nlp = spacy.load("xx_ent_wiki_sm")
    print("spaCy loaded: xx_ent_wiki_sm (fallback)")

# ── FastAPI ──────────────────────────────────────────────────
app = FastAPI(title="Hoax Detection API")

class Request(BaseModel):
    text: str

class NERRequest(BaseModel):
    summary: str
    articles: list[str]

@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoints": {
            "POST /summarize": "Summarize teks dengan mBART",
            "POST /predict":   "Klasifikasi hoax dengan IndoBERT",
            "POST /ner":       "Ekstrak & bandingkan entitas NER"
        }
    }

# ── Endpoint BART: Summarize ──────────────────────────────────
@app.post("/summarize")
def summarize(req: Request):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text tidak boleh kosong")

    inputs = bart_tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        summary_ids = bart_model.generate(
            **inputs,
            forced_bos_token_id=bart_tokenizer.convert_tokens_to_ids("id_ID"),
            max_length=128,
            min_length=20,
            num_beams=4,
            early_stopping=True
        )

    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

# ── Endpoint BERT: Predict ────────────────────────────────────
@app.post("/predict")
def predict(req: Request):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text tidak boleh kosong")

    inputs = bert_tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = round(probs[0][pred_id].item() * 100, 2)

    raw_label = id2label.get(pred_id, str(pred_id))

    # Normalisasi label ke HOAKS/BENAR
    label_map = {
        "label_0": "BENAR",
        "label_1": "HOAKS",
        "0": "BENAR",
        "1": "HOAKS",
        "valid": "BENAR",
        "hoax": "HOAKS",
    }
    label = label_map.get(raw_label.lower(), raw_label)

    return {
        "label": label,
        "confidence": confidence,
        "raw_label": raw_label
    }

# ── Endpoint spaCy: NER ───────────────────────────────────────
@app.post("/ner")
def ner(req: NERRequest):
    if not req.summary.strip():
        raise HTTPException(status_code=400, detail="Summary tidak boleh kosong")

    def extract_entities(text):
        doc = nlp(text)
        entities = {}
        for ent in doc.ents:
            label = ent.label_
            if label not in entities:
                entities[label] = []
            if ent.text not in entities[label]:
                entities[label].append(ent.text)
        return entities

    summary_ents = extract_entities(req.summary)

    article_ents = {}
    for article in req.articles:
        for label, values in extract_entities(article).items():
            if label not in article_ents:
                article_ents[label] = []
            for v in values:
                if v not in article_ents[label]:
                    article_ents[label].append(v)

    # Deteksi mismatch: entitas di summary yang tidak ada di artikel manapun
    mismatch = {}
    for label, values in summary_ents.items():
        missing = [v for v in values if v not in article_ents.get(label, [])]
        if missing:
            mismatch[label] = missing

    return {
        "summary_entities": summary_ents,
        "article_entities": article_ents,
        "mismatch": mismatch,
        "has_mismatch": len(mismatch) > 0
    }
