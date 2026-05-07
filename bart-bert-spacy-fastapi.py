%%writefile app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    MBartForConditionalGeneration,
    AutoModelForSequenceClassification,
    pipeline,
)
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
import re
import os
from difflib import SequenceMatcher

# Konfigurasi path model
BART_MODEL_DIR    = "/content/drive/MyDrive/mbart_indosum_final"
BERT_MODEL_DIR    = "/content/drive/MyDrive/indobert-hoax-model/checkpoint-7185"
BERT_TOKENIZER    = "indobenchmark/indobert-base-p1"

_NER_DRIVE     = "/content/drive/MyDrive/bert-indonesian-ner"
NER_MODEL_NAME = _NER_DRIVE if os.path.isdir(_NER_DRIVE) else "cahya/bert-base-indonesian-NER"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load BART
print("Loading BART...")
bart_tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_DIR, use_fast=True)
bart_tokenizer.src_lang = "id_ID"
bart_model = MBartForConditionalGeneration.from_pretrained(BART_MODEL_DIR)
bart_model.to(device).eval()
print("✓ BART loaded")

# Load BERT Classification
print("Loading BERT Classification...")
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
bert_model.to(device).eval()
id2label = bert_model.config.id2label
print(f"✓ BERT Classification loaded — labels: {id2label}")

# Load BERT NER (cahya/bert-base-indonesian-NER)
print(f"Loading BERT NER dari: {NER_MODEL_NAME}")
ner_model = pipeline(
    "ner",
    model=NER_MODEL_NAME,
    tokenizer=NER_MODEL_NAME,
    aggregation_strategy="simple",
    device=0 if device == "cuda" else -1,
)
print("✓ BERT NER loaded")

_MONTH_MAP = {
    'januari': 1, 'februari': 2, 'maret': 3, 'april': 4,
    'mei': 5, 'juni': 6, 'juli': 7, 'agustus': 8,
    'september': 9, 'oktober': 10, 'november': 11, 'desember': 12,
    'january': 1, 'february': 2, 'march': 3, 'may': 5,
    'june': 6, 'july': 7, 'august': 8, 'october': 10, 'december': 12,
}

def _parse_date(text: str) -> Optional[tuple]:
    """
    Parse tanggal dari teks TIM entity → (day, month, year) atau None.
    """
    t = text.lower()
    month = next((v for k, v in _MONTH_MAP.items() if k in t), None)
    numbers = [int(n) for n in re.findall(r'\d+', t)]
    year  = next((n for n in numbers if 2000 <= n <= 2100), None)
    day   = next((n for n in numbers if 1 <= n <= 31 and n != year), None)
    if day is not None and month is not None and year is not None:
        return (day, month, year)
    return None


def _extract_number(text: str) -> Optional[float]:
    """Ekstrak nilai numerik dari teks entitas QTY."""
    t = text.lower()
    cleaned = re.sub(r'(?<=\d)\.(?=\d{3}(?!\d))', '', t)
    cleaned = cleaned.replace(',', '.')
    match = re.search(r'\d+(?:\.\d+)?', cleaned)
    if not match:
        return None
    val = float(match.group())
    if 'triliun' in t or 'trillion' in t:  val *= 1e12
    elif 'miliar' in t or 'billion' in t:  val *= 1e9
    elif 'juta'   in t or 'million' in t:  val *= 1e6
    elif 'ribu'   in t or 'thousand' in t: val *= 1e3
    return val


def _fuzzy(a: str, b: str) -> float:
    """Kemiripan string 0–1 menggunakan SequenceMatcher."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _tim_match(u_text: str, e_text: str) -> bool:
    """
    Bandingkan dua entitas TIM (tanggal/waktu).
    Jika keduanya adalah tanggal lengkap, bandingkan hari+bulan+tahun secara eksak.
    Fallback ke fuzzy ≥ 0.92 (threshold tinggi agar beda hari kecil tetap terdeteksi).
    """
    u_date = _parse_date(u_text)
    e_date = _parse_date(e_text)
    if u_date and e_date:
        return u_date == e_date  # harus sama persis
    return _fuzzy(u_text, e_text) >= 0.92


def _entities_match(u: Dict, e: Dict, etype: str) -> bool:
    """
    Cek apakah dua entitas dianggap cocok.
    - TIM → bandingkan tanggal eksak (hari+bulan+tahun)
    - QTY → bandingkan nilai numerik, toleransi 10%
    - PER / ORG / LOC / EVT → fuzzy string ≥ 0.72
    """
    u_text = u['word'].strip()
    e_text = e['word'].strip()

    if etype == 'TIM':
        return _tim_match(u_text, e_text)

    if etype == 'QTY':
        u_val = _extract_number(u_text)
        e_val = _extract_number(e_text)
        if u_val is not None and e_val is not None and min(u_val, e_val) > 0:
            return max(u_val, e_val) / min(u_val, e_val) <= 1.1  # toleransi 10%
        return _fuzzy(u_text, e_text) >= 0.72

    return _fuzzy(u_text, e_text) >= 0.72


def _calc_severity(u: Dict, ev_ents: List[Dict], etype: str) -> str:
    """Hitung tingkat keparahan mismatch: HIGH / MEDIUM / LOW."""
    u_text = u['word'].strip()

    if etype == 'TIM':
        # Tanggal berbeda tapi bisa di-parse → selalu HIGH
        return 'HIGH' if _parse_date(u_text) else 'MEDIUM'

    if etype == 'QTY':
        u_val = _extract_number(u_text)
        if u_val is not None and u_val > 0:
            e_vals = [_extract_number(e['word']) for e in ev_ents]
            e_vals = [v for v in e_vals if v is not None and v > 0]
            if e_vals:
                best_ratio = min(max(u_val, v) / min(u_val, v) for v in e_vals)
                if best_ratio > 2.0:  return 'HIGH'
                elif best_ratio > 1.5: return 'MEDIUM'
                else:                  return 'LOW'

    best_sim = max((_fuzzy(u_text, e['word']) for e in ev_ents), default=0.0)
    if best_sim < 0.4:   return 'HIGH'
    elif best_sim < 0.6: return 'MEDIUM'
    else:                return 'LOW'


_DATE_RE = re.compile(
    r'\b(\d{1,2})\s+(' + '|'.join(_MONTH_MAP.keys()) + r')\s+(\d{4})\b',
    re.IGNORECASE
)

def _extract_dates_regex(text: str) -> List[Dict]:
    """
    Suplemen BERT NER: ekstrak tanggal "DD Bulan YYYY" pakai regex.
    Diperlukan karena model cahya/bert-base-indonesian-NER tidak punya label TIM.
    """
    results = []
    seen = set()
    for m in _DATE_RE.finditer(text):
        key = m.group(0).lower().strip()
        if key in seen:
            continue
        seen.add(key)
        results.append({
            'word':         m.group(0).strip(),
            'entity_group': 'TIM',
            'score':        1.0,
            'start':        m.start(),
            'end':          m.end(),
        })
    return results


def _extract_ner(text: str) -> List[Dict]:
    """
    Ekstrak entitas: BERT NER (PER/ORG/LOC) + regex (TIM/tanggal).
    Teks panjang dipotong per kalimat agar tidak melebihi 512 token.
    """
    if not text or not text.strip():
        return []

    # Bersihkan emoji/karakter non-standar yang bisa crash tokenizer
    clean_text = re.sub(r'[^\w\s.,\-:/\'\"()!?]', ' ', text, flags=re.UNICODE)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    results = []
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    offset = 0
    for sent in sentences:
        if not sent.strip():
            offset += len(sent) + 1
            continue
        try:
            chunk = sent[:500]
            ents = ner_model(chunk)
            for r in ents:
                if r.get('score', 0) < 0.70:
                    continue
                raw_label = r.get('entity_group') or r.get('entity', 'O')
                entity_group = raw_label.split('-', 1)[-1] if '-' in raw_label else raw_label
                if entity_group in ('O', ''):
                    continue
                word = r.get('word', '').replace('##', '').strip()
                # Filter artefak tokenizer: abaikan entitas ≤ 1 karakter
                if len(word) <= 1:
                    continue
                results.append({
                    'word':         word,
                    'entity_group': entity_group,
                    'score':        round(float(r.get('score', 0)), 4),
                    'start':        int(r.get('start', 0)) + offset,
                    'end':          int(r.get('end', 0)) + offset,
                })
        except Exception as exc:
            print(f"[NER] chunk error: {exc}")
        offset += len(sent) + 1

    # Tambahkan entitas tanggal dari regex (TIM tidak ada di label BERT model)
    results.extend(_extract_dates_regex(clean_text))

    return results


def compare_ner_entities(user_text: str, evidence_texts: List[str]) -> Dict:
    """
    Bandingkan entitas antara user_text dan evidence.
    Hanya flag mismatch ketika evidence MEMILIKI entitas tipe yang sama
    tapi tidak ada yang cocok — menghindari false positive.
    """
    user_ents = _extract_ner(user_text)

    ev_ents_all: List[Dict] = []
    for t in evidence_texts:
        ev_ents_all.extend(_extract_ner(t))

    def group(ents: List[Dict]) -> Dict[str, List[Dict]]:
        g: Dict[str, List[Dict]] = {}
        for e in ents:
            g.setdefault(e['entity_group'], []).append(e)
        return g

    user_by_type = group(user_ents)
    ev_by_type   = group(ev_ents_all)

    severity_weights = {'HIGH': 30, 'MEDIUM': 15, 'LOW': 5}
    mismatches: List[Dict] = []

    for etype, u_list in user_by_type.items():
        ev_list = ev_by_type.get(etype, [])
        if not ev_list:
            # Evidence tidak punya entitas tipe ini → tidak bisa dibandingkan
            continue

        for u_ent in u_list:
            matched = any(_entities_match(u_ent, e, etype) for e in ev_list)
            if matched:
                continue

            severity = _calc_severity(u_ent, ev_list, etype)

            # Deduplikasi: skip jika entitas sangat mirip sudah ada
            duplicate = any(
                m['category'] == etype and _fuzzy(m['user_text'], u_ent['word']) > 0.85
                for m in mismatches
            )
            if duplicate:
                continue

            mismatches.append({
                'category':     etype,
                'user_text':    u_ent['word'],
                # user_value sengaja sama dengan user_text agar kompatibel dengan api.js
                'user_value':   u_ent['word'],
                'user_context': user_text[max(0, u_ent['start'] - 60): u_ent['end'] + 60],
                # Kirim teks unik dari evidence (maks 5) untuk referensi
                'evidence_values': list({e['word'] for e in ev_list})[:5],
                'severity':     severity,
            })

    risk_score = min(
        sum(severity_weights.get(m['severity'], 0) for m in mismatches),
        100.0
    )

    return {
        'user_entities':     [{'text': e['word'], 'category': e['entity_group'], 'score': float(e['score'])} for e in user_ents],
        'evidence_entities': [{'text': e['word'], 'category': e['entity_group'], 'score': float(e['score'])} for e in ev_ents_all],
        'mismatches':        mismatches,
        'has_mismatch':      len(mismatches) > 0,
        'mismatch_count':    int(len(mismatches)),
        'risk_score':        float(risk_score),
    }

app = FastAPI(title="Hoax Detection API")

class TextRequest(BaseModel):
    text: str

class NERRequest(BaseModel):
    summary: str
    articles: List[str]


@app.get("/")
def root():
    return {
        "status": "ok",
        "ner_model": NER_MODEL_NAME,
        "endpoints": {
            "POST /summarize": "Summarize teks dengan mBART",
            "POST /predict":   "Klasifikasi hoax dengan IndoBERT",
            "POST /ner":       "Ekstrak & bandingkan entitas (BERT Indonesian NER)",
        },
    }


@app.post("/summarize")
def summarize(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text tidak boleh kosong")

    inputs = bart_tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    ).to(device)

    with torch.no_grad():
        ids = bart_model.generate(
            **inputs,
            forced_bos_token_id=bart_tokenizer.convert_tokens_to_ids("id_ID"),
            max_length=128,
            min_length=20,
            num_beams=4,
            early_stopping=True,
        )

    return {"summary": bart_tokenizer.decode(ids[0], skip_special_tokens=True)}


@app.post("/predict")
def predict(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text tidak boleh kosong")

    inputs = bert_tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        probs   = F.softmax(bert_model(**inputs).logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = round(probs[0][pred_id].item() * 100, 2)

    raw_label = id2label.get(pred_id, str(pred_id))
    label_map = {
        "label_0": "BENAR", "label_1": "HOAKS",
        "0": "BENAR",       "1": "HOAKS",
        "valid": "BENAR",   "hoax": "HOAKS",
    }
    return {
        "label":      label_map.get(raw_label.lower(), raw_label),
        "confidence": confidence,
        "raw_label":  raw_label,
    }


@app.post("/ner")
def ner(req: NERRequest):
    if not req.summary.strip():
        raise HTTPException(status_code=400, detail="Summary tidak boleh kosong")
    try:
        return compare_ner_entities(req.summary, req.articles)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"NER error: {exc}")
