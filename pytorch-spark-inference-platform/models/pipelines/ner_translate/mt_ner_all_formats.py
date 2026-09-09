"""Routed offline multilingual NER pipeline — multi-format, batched.

For each input document (any supported format: text, PDF, image, Office doc,
HTML, CSV/JSON, ...):
  1. Extract plain text from the file (format-aware; OCR for images / scanned PDFs).
  2. Detect language (py3langid, offline).
  3. If the language is one GLiNER handles well, run GLiNER on the original text.
  4. Otherwise, translate to English (NLLB, offline), then run GLiNER.
  5. Tag every result with its detected language and whether it was translated.

Runs fully offline from locally saved models. Supports GPU with batching and
optional fp16.

Supported input formats
------------------------
  Plain text : .txt .md .log .csv .tsv .json .jsonl .html .htm .xml
  Documents  : .pdf .docx .odt .rtf .epub
  Slides     : .pptx
  Spreadsheet: .xlsx .xlsm .xls .ods   (text of all cells)
  Images     : .png .jpg .jpeg .bmp .tif .tiff .webp .gif  (via OCR)

A single run / batch can freely mix all of these.

One-time setup (on a connected machine), then copy the folders offline:

    pip install gliner transformers torch sentencepiece py3langid

    # Optional, per format you need:
    pip install pypdf pdf2image pillow pytesseract      # PDF + image OCR
    pip install python-docx openpyxl python-pptx odfpy  # Office + ODF
    pip install ebooklib beautifulsoup4 striprtf xlrd   # epub / html / rtf / .xls
    #   OCR also needs the system binaries: tesseract-ocr and poppler-utils

    # GLiNER model
    python -c "from gliner import GLiNER; \
GLiNER.from_pretrained('urchade/gliner_multi-v2.1').save_pretrained('./models/gliner-multi')"

    # NLLB translation model
    python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
m='facebook/nllb-200-distilled-600M'; \
AutoTokenizer.from_pretrained(m).save_pretrained('./models/nllb-200-distilled-600M'); \
AutoModelForSeq2SeqLM.from_pretrained(m).save_pretrained('./models/nllb-200-distilled-600M')"
"""

import os
import re
import json
import glob
import unicodedata

# ---- Force offline mode for HF libraries: no network calls at all ----
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Force Arrow's runtime to load before torch's (Windows OpenMP/MKL clash)
import pyarrow  # noqa: F401  -- must come before torch
import py3langid as langid
from gliner import GLiNER          # pulls in pandas/sklearn/transformers -> needs arrow first
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import torch                        # safe to load now

# ---- Config: paths ----
# Resolved relative to this file (not process CWD) so this module works the
# same whether invoked directly, imported by pipeline.py, or run inside a
# Spark executor whose working directory may differ from the driver's.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "weights"))
GLINER_DIR = os.path.join(_WEIGHTS_DIR, "gliner-multi")
NLLB_DIR = os.path.join(_WEIGHTS_DIR, "nllb-200-distilled-600M")
LID_PATH = os.path.join(_WEIGHTS_DIR, "lid.176.bin")  # (unused; py3langid ships its own)

# ---- Config: behavior ----
SCORE_THRESHOLD = 0.35       # lower than English-only default; non-EN scores run low
MAX_CHARS = 1500             # approx chunk size in characters
BATCH_SIZE = 16              # GLiNER chunks per forward pass
USE_FP16 = True             # half precision on GPU (ignored on CPU)
TRANSLATE_MAX_TOKENS = 400   # per-chunk cap for NLLB generation
TRANSLATE_BATCH_SIZE = 8     # chunks per NLLB forward pass
# Tesseract OCR languages. This MUST include the scripts your images actually
# contain, or OCR returns garbage and NER sees almost nothing. Join packs with
# '+'. Each language needs its Tesseract pack installed on the machine, e.g.:
#   Ubuntu/Debian: sudo apt install tesseract-ocr-mar tesseract-ocr-hin \
#                       tesseract-ocr-tel tesseract-ocr-tam ...
#   or drop <lang>.traineddata files into your tessdata folder.
# List installed packs with:  tesseract --list-langs
OCR_LANG = "eng+mar+hin+tel+tam+kan+ben+guj+pan+mal+urd"
OCR_DPI = 300                # rasterization DPI for scanned PDFs (300 = better OCR)
OCR_AUTO_FILTER = True       # silently drop OCR langs whose packs aren't installed

# ---- Supported extensions (dispatch key for extraction) ----
TEXT_EXTS  = {".txt", ".md", ".log", ".csv", ".tsv", ".json", ".jsonl",
              ".xml", ".yaml", ".yml", ".ini", ".rst"}
HTML_EXTS  = {".html", ".htm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
DOCX_EXTS  = {".docx"}
ODT_EXTS   = {".odt"}
RTF_EXTS   = {".rtf"}
EPUB_EXTS  = {".epub"}
PPTX_EXTS  = {".pptx"}
XLSX_EXTS  = {".xlsx", ".xlsm"}
XLS_EXTS   = {".xls"}
ODS_EXTS   = {".ods"}
PDF_EXTS   = {".pdf"}

SUPPORTED_EXTS = (TEXT_EXTS | HTML_EXTS | IMAGE_EXTS | DOCX_EXTS | ODT_EXTS |
                  RTF_EXTS | EPUB_EXTS | PPTX_EXTS | XLSX_EXTS | XLS_EXTS |
                  ODS_EXTS | PDF_EXTS)

# Languages GLiNER (gliner_multi-v2.1) handles well -> send original text直接.
WELL_SUPPORTED = {"en", "fr", "de", "es", "it", "pt"}

# Map ISO-639-1 codes -> NLLB BCP-47 codes (extend as needed).
NLLB_LANG = {
    "hi": "hin_Deva", "mr": "mar_Deva", "te": "tel_Telu", "kn": "kan_Knda",
    "ta": "tam_Taml", "bn": "ben_Beng", "gu": "guj_Gujr", "pa": "pan_Guru",
    "ml": "mal_Mlym", "ur": "urd_Arab", "ar": "arb_Arab", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "ru": "rus_Cyrl", "fa": "pes_Arab",
    "tr": "tur_Latn", "vi": "vie_Latn", "th": "tha_Thai", "id": "ind_Latn",
    "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "es": "spa_Latn", "it": "ita_Latn", "pt": "por_Latn",
}

# Defence / ICIC starter schema — prune and extend after testing on real docs.
DEFAULT_LABELS = [
    "person", "rank", "organization", "military unit", "weapon system",
    "vehicle", "aircraft", "vessel", "equipment", "facility", "location",
    "coordinates", "operation name", "event", "date", "time", "nationality",
    "communication identifier", "phone number",
]


# =====================================================================
# Text extraction — turns any supported file into plain text
# =====================================================================
def _lazy_import(name, pip_hint):
    """Import a module on demand, with a clear message if it's missing."""
    try:
        return __import__(name)
    except ImportError as e:
        raise ImportError(
            f"'{name}' is required to read this file type. "
            f"Install it with: pip install {pip_hint}"
        ) from e


_OCR_LANG_CACHE = None


def _resolve_ocr_lang():
    """Return an OCR language string limited to packs Tesseract actually has.

    Prevents the common failure where OCR_LANG names a script (e.g. 'mar') whose
    traineddata isn't installed — Tesseract errors out and the document yields no
    text. Falls back to 'eng', then to whatever the first available pack is.
    """
    global _OCR_LANG_CACHE
    if _OCR_LANG_CACHE is not None:
        return _OCR_LANG_CACHE

    requested = [x for x in OCR_LANG.split("+") if x]
    if not OCR_AUTO_FILTER:
        _OCR_LANG_CACHE = OCR_LANG
        return _OCR_LANG_CACHE

    try:
        import pytesseract
        available = set(pytesseract.get_languages(config=""))
    except Exception:
        _OCR_LANG_CACHE = OCR_LANG
        return _OCR_LANG_CACHE

    keep = [l for l in requested if l in available]
    missing = [l for l in requested if l not in available]
    if missing:
        print(f"  [warn] OCR language pack(s) not installed, skipping: "
              f"{', '.join(missing)}  (install e.g. tesseract-ocr-<lang>)")
    if not keep:
        keep = ["eng"] if "eng" in available else sorted(available)[:1]
        print(f"  [warn] falling back to OCR lang(s): {'+'.join(keep) or 'none'}")
    _OCR_LANG_CACHE = "+".join(keep)
    return _OCR_LANG_CACHE


def _extract_pdf(path: str) -> str:
    """Extract text from a PDF; OCR pages that have no text layer."""
    text_parts = []
    pypdf = _lazy_import("pypdf", "pypdf")
    try:
        reader = pypdf.PdfReader(path)
    except Exception:
        reader = None

    page_texts = []
    if reader is not None:
        for page in reader.pages:
            try:
                page_texts.append(page.extract_text() or "")
            except Exception:
                page_texts.append("")

    # If we got little/no text, the PDF is likely scanned -> OCR the whole thing.
    joined = "\n".join(page_texts).strip()
    if len(joined) >= 20:
        return joined

    # OCR fallback (needs pdf2image + poppler + pytesseract + tesseract).
    try:
        pdf2image = _lazy_import("pdf2image", "pdf2image")
        pytesseract = _lazy_import("pytesseract", "pytesseract")
        ocr_lang = _resolve_ocr_lang()
        images = pdf2image.convert_from_path(path, dpi=OCR_DPI)
        for img in images:
            text_parts.append(pytesseract.image_to_string(img, lang=ocr_lang))
        ocr_text = "\n".join(text_parts).strip()
        if ocr_text:
            return ocr_text
    except Exception as e:
        print(f"  [warn] PDF OCR failed for {path}: {e}")

    return joined  # whatever little we had


def _extract_image(path: str) -> str:
    """OCR an image into text."""
    Image = _lazy_import("PIL", "pillow").Image  # noqa
    from PIL import Image
    pytesseract = _lazy_import("pytesseract", "pytesseract")
    img = Image.open(path)
    return pytesseract.image_to_string(img, lang=_resolve_ocr_lang())


def _extract_docx(path: str) -> str:
    docx = _lazy_import("docx", "python-docx")
    from docx import Document
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append("\t".join(c.text for c in row.cells))
    return "\n".join(parts)


def _extract_pptx(path: str) -> str:
    pptx = _lazy_import("pptx", "python-pptx")
    from pptx import Presentation
    prs = Presentation(path)
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    parts.append("".join(r.text for r in para.runs))
            if shape.has_table:
                for row in shape.table.rows:
                    parts.append("\t".join(c.text for c in row.cells))
    return "\n".join(parts)


def _extract_xlsx(path: str) -> str:
    openpyxl = _lazy_import("openpyxl", "openpyxl")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        parts.append(f"# Sheet: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                parts.append("\t".join(cells))
    return "\n".join(parts)


def _extract_xls(path: str) -> str:
    pd = _lazy_import("pandas", "pandas xlrd")
    import pandas as pd
    sheets = pd.read_excel(path, engine="xlrd", sheet_name=None, header=None)
    parts = []
    for name, df in sheets.items():
        parts.append(f"# Sheet: {name}")
        parts.append(df.to_csv(sep="\t", index=False, header=False))
    return "\n".join(parts)


def _extract_ods(path: str) -> str:
    pd = _lazy_import("pandas", "pandas odfpy")
    import pandas as pd
    sheets = pd.read_excel(path, engine="odf", sheet_name=None, header=None)
    parts = []
    for name, df in sheets.items():
        parts.append(f"# Sheet: {name}")
        parts.append(df.to_csv(sep="\t", index=False, header=False))
    return "\n".join(parts)


def _extract_odt(path: str) -> str:
    _lazy_import("odf", "odfpy")
    from odf.opendocument import load
    from odf import text as odftext, teletype
    doc = load(path)
    parts = [teletype.extractText(el)
             for el in doc.getElementsByType(odftext.P)]
    return "\n".join(parts)


def _extract_rtf(path: str) -> str:
    striprtf = _lazy_import("striprtf", "striprtf")
    from striprtf.striprtf import rtf_to_text
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return rtf_to_text(f.read())


def _extract_epub(path: str) -> str:
    _lazy_import("ebooklib", "ebooklib beautifulsoup4")
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup
    book = epub.read_epub(path)
    parts = []
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            parts.append(soup.get_text(" "))
    return "\n".join(parts)


def _extract_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(raw, "html.parser").get_text(" ")
    except ImportError:
        return re.sub(r"<[^>]+>", " ", raw)  # crude tag strip fallback


def _extract_json(path: str) -> str:
    """Flatten JSON / JSONL into readable text (all string values joined)."""
    def walk(obj, out):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v, out)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, out)
        elif isinstance(obj, str):
            out.append(obj)
        elif obj is not None:
            out.append(str(obj))

    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    if path.lower().endswith(".jsonl"):
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                walk(json.loads(line), out)
            except json.JSONDecodeError:
                out.append(line)
    else:
        try:
            walk(json.loads(content), out)
        except json.JSONDecodeError:
            out.append(content)
    return "\n".join(out)


def _extract_plaintext(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_text(path: str) -> str:
    """Turn any supported file into plain text via the right extractor.

    Dispatches on file extension. Raises ValueError for unsupported types.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in PDF_EXTS:
        return _extract_pdf(path)
    if ext in IMAGE_EXTS:
        return _extract_image(path)
    if ext in DOCX_EXTS:
        return _extract_docx(path)
    if ext in PPTX_EXTS:
        return _extract_pptx(path)
    if ext in XLSX_EXTS:
        return _extract_xlsx(path)
    if ext in XLS_EXTS:
        return _extract_xls(path)
    if ext in ODS_EXTS:
        return _extract_ods(path)
    if ext in ODT_EXTS:
        return _extract_odt(path)
    if ext in RTF_EXTS:
        return _extract_rtf(path)
    if ext in EPUB_EXTS:
        return _extract_epub(path)
    if ext in HTML_EXTS:
        return _extract_html(path)
    if ext in (".json", ".jsonl"):
        return _extract_json(path)
    if ext in TEXT_EXTS:
        return _extract_plaintext(path)
    raise ValueError(f"Unsupported file type: {ext}")


# =====================================================================
# Phone / mobile number extraction (regex fallback)
# =====================================================================
PHONE_RE = re.compile(
    r"(?<![\w])"
    r"(?:\+?\d{1,3}[\s.-]?)?"
    r"(?:\(\d{1,4}\)[\s.-]?)?"
    r"\d{3,4}[\s.-]?\d{3,4}"
    r"(?:[\s.-]?\d{2,4})?"
    r"(?![\w])"
)


def extract_phone_numbers(text: str, doc_offset: int = 0):
    """Regex-extract phone / mobile numbers as a reliable fallback."""
    found = []
    for m in PHONE_RE.finditer(text):
        raw = m.group().strip()
        digits = re.sub(r"\D", "", raw)
        if not (7 <= len(digits) <= 15):
            continue
        found.append({
            "text": raw, "type": "phone number", "score": 1.0,
            "start": m.start() + doc_offset, "end": m.end() + doc_offset,
        })
    return found


# =====================================================================
# Model loading
# =====================================================================
def load_models():
    """Load GLiNER, NLLB, and the py3langid language-ID model for offline use."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected — running on CPU.")

    gliner_model = GLiNER.from_pretrained(GLINER_DIR, local_files_only=True)
    gliner_model = gliner_model.to(device)
    gliner_model.eval()
    if device == "cuda" and USE_FP16:
        gliner_model = gliner_model.half()
        print("GLiNER using fp16.")

    nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_DIR, local_files_only=True)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_DIR, local_files_only=True)
    nllb_model = nllb_model.to(device)
    nllb_model.eval()

    lid_model = langid   # the module itself is the "model"; nothing to load

    return gliner_model, nllb_model, nllb_tokenizer, lid_model, device


# =====================================================================
# Preprocessing / chunking
# =====================================================================
def preprocess(text: str) -> str:
    """Normalize and clean raw document text."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = MAX_CHARS):
    """Split text into character-bounded chunks while tracking offsets."""
    sentences = re.split(r"(?<=[.!?。!?])\s+", text)
    chunks, offsets = [], []
    cur, cur_start, cursor = "", 0, 0

    for sent in sentences:
        sent_start = text.find(sent, cursor)
        cursor = sent_start + len(sent)
        candidate = (cur + " " + sent).strip() if cur else sent
        if len(candidate) > max_chars and cur:
            chunks.append(cur); offsets.append(cur_start)
            cur, cur_start = sent, sent_start
        else:
            if not cur:
                cur_start = sent_start
            cur = candidate

    if cur:
        chunks.append(cur); offsets.append(cur_start)
    return chunks, offsets


# =====================================================================
# Language detection
# =====================================================================
def detect_language(lid_model, text: str) -> str:
    """Detect the dominant language of a text with py3langid."""
    sample = text.replace("\n", " ").strip()[:1000]
    if not sample:
        return "unknown"
    try:
        lang, _ = lid_model.classify(sample)
        return lang
    except Exception:
        return "unknown"


# =====================================================================
# Translation (only used for non-well-supported languages) — batched
# =====================================================================
def translate_chunks(nllb_model, nllb_tokenizer, chunks, src_code: str, device: str):
    """Translate text chunks into English with NLLB, batched."""
    if not chunks:
        return []
    nllb_tokenizer.src_lang = src_code
    eng_id = nllb_tokenizer.convert_tokens_to_ids("eng_Latn")

    translations = []
    with torch.no_grad():
        for i in range(0, len(chunks), TRANSLATE_BATCH_SIZE):
            batch = chunks[i:i + TRANSLATE_BATCH_SIZE]
            inputs = nllb_tokenizer(
                batch, return_tensors="pt", truncation=True, padding=True,
                max_length=TRANSLATE_MAX_TOKENS,
            ).to(device)
            out = nllb_model.generate(
                **inputs, forced_bos_token_id=eng_id,
                max_length=TRANSLATE_MAX_TOKENS,
            )
            translations.extend(
                nllb_tokenizer.batch_decode(out, skip_special_tokens=True)
            )
    return translations


# =====================================================================
# GLiNER prediction + cleanup
# =====================================================================
def predict_chunks(gliner_model, chunks, labels, batch_size: int = BATCH_SIZE):
    """Run GLiNER over text chunks, batched for GPU efficiency."""
    results = []
    with torch.no_grad():
        if hasattr(gliner_model, "batch_predict_entities"):
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                results.extend(
                    gliner_model.batch_predict_entities(
                        batch, labels, threshold=SCORE_THRESHOLD
                    )
                )
        else:
            for chunk in chunks:
                results.append(
                    gliner_model.predict_entities(
                        chunk, labels, threshold=SCORE_THRESHOLD
                    )
                )
    return results


def postprocess(entities, doc_offset: int = 0):
    """Clean, filter, and reposition raw entities from one chunk."""
    cleaned = []
    for e in entities:
        word = e["text"].strip()
        if not word or not re.search(r"\w", word):
            continue
        if e["score"] < SCORE_THRESHOLD:
            continue
        cleaned.append({
            "text": word, "type": e["label"],
            "score": round(float(e["score"]), 4),
            "start": int(e["start"]) + doc_offset,
            "end": int(e["end"]) + doc_offset,
        })
    return cleaned


def dedupe(entities):
    """Collapse duplicate mentions into unique (text, type) pairs."""
    seen = {}
    for e in entities:
        key = (e["text"].lower(), e["type"])
        if key not in seen or e["score"] > seen[key]["score"]:
            seen[key] = e
    return list(seen.values())


# =====================================================================
# Batched multi-document processing
# =====================================================================
def _build_document_records(paths, lid_model):
    """Extract + preprocess + language-detect + route every document.

    Returns a list of per-document records. Chunks from ALL documents are later
    pooled so GLiNER/NLLB run on cross-document batches (a batch can freely mix
    formats and languages).
    """
    records = []
    for fp in paths:
        rec = {"path": fp, "error": None}
        try:
            raw = extract_text(fp)
        except Exception as e:
            rec["error"] = str(e)
            print(f"  [skip] {fp}: {e}")
            records.append(rec)
            continue

        original = preprocess(raw)
        lang = detect_language(lid_model, original)
        translate = lang not in WELL_SUPPORTED and lang in NLLB_LANG

        rec.update({
            "original": original,
            "language": lang,
            "translate": translate,
            "src_code": NLLB_LANG.get(lang) if translate else None,
        })
        records.append(rec)
    return records


def process_paths_batched(models, paths, labels=DEFAULT_LABELS):
    """Process many files of any supported format in cross-document batches.

    Pipeline:
      1. Extract text from every file and detect language / route.
      2. Pool translation chunks from all translate-path docs -> batched NLLB.
      3. Pool GLiNER chunks from all docs -> batched GLiNER.
      4. Scatter results back to each document; run the regex phone pass.

    Returns a dict mapping each file's basename to its result (same shape as the
    original single-file output, plus a "path" key).
    """
    gliner_model, nllb_model, nllb_tokenizer, lid_model, device = models

    records = _build_document_records(paths, lid_model)

    # ---- Stage 1: batched translation, grouped by source language ----
    # (NLLB needs one src_lang per batch, so group translate-docs by language.)
    for rec in records:
        rec["text_used"] = None
    by_src = {}
    for idx, rec in enumerate(records):
        if rec.get("error") or not rec.get("translate"):
            continue
        by_src.setdefault(rec["src_code"], []).append(idx)

    for src_code, idxs in by_src.items():
        pooled_chunks, spans = [], []
        for idx in idxs:
            chunks, _ = chunk_text(records[idx]["original"])
            spans.append((idx, len(pooled_chunks), len(chunks)))
            pooled_chunks.extend(chunks)
        eng = translate_chunks(nllb_model, nllb_tokenizer,
                               pooled_chunks, src_code, device)
        for idx, start, n in spans:
            records[idx]["text_used"] = " ".join(eng[start:start + n])

    # Direct-path (and unmapped) docs use their original text.
    for rec in records:
        if rec.get("error"):
            continue
        if rec["text_used"] is None:
            rec["text_used"] = rec["original"]

    # ---- Stage 2: batched GLiNER over pooled chunks from every doc ----
    pooled_chunks, spans, doc_chunk_offsets = [], [], []
    for idx, rec in enumerate(records):
        if rec.get("error"):
            continue
        chunks, offsets = chunk_text(rec["text_used"])
        spans.append((idx, len(pooled_chunks), len(chunks)))
        doc_chunk_offsets.append((idx, chunks, offsets))
        pooled_chunks.extend(chunks)

    pooled_results = predict_chunks(gliner_model, pooled_chunks, labels) \
        if pooled_chunks else []

    # ---- Stage 3: scatter GLiNER results + regex phone pass ----
    results = {}
    span_map = {idx: (start, n) for idx, start, n in spans}
    chunk_map = {idx: (chunks, offsets) for idx, chunks, offsets in doc_chunk_offsets}

    for rec in records:
        fp = rec["path"]
        name = os.path.basename(fp)
        if rec.get("error"):
            results[name] = {"path": fp, "error": rec["error"]}
            continue

        idx = records.index(rec)
        all_entities = []
        if idx in span_map:
            start, n = span_map[idx]
            chunks, offsets = chunk_map[idx]
            for j in range(n):
                ents = pooled_results[start + j]
                off = offsets[j]
                all_entities.extend(postprocess(ents, doc_offset=off))
                all_entities.extend(extract_phone_numbers(chunks[j], doc_offset=off))

        # Translated docs: also scan ORIGINAL text so true digits survive.
        if rec["translate"]:
            orig_chunks, orig_offsets = chunk_text(rec["original"])
            for chunk, off in zip(orig_chunks, orig_offsets):
                all_entities.extend(extract_phone_numbers(chunk, doc_offset=off))

        data = {
            "path": fp,
            "language": rec["language"],
            "translated": rec["translate"],
            "text_used": rec["text_used"],
            "original_text": rec["original"],
            "entities_all": all_entities,
            "entities_unique": dedupe(all_entities),
        }
        results[name] = data
        route = "translated" if rec["translate"] else "direct"
        print(f"[done] {fp}  lang={rec['language']}  route={route}  "
              f"-> {len(data['entities_unique'])} unique entities")

    return results


def collect_files(path):
    """Resolve a file, directory, or glob into a sorted list of supported files."""
    if os.path.isdir(path):
        candidates = sorted(
            f for f in glob.glob(os.path.join(path, "**", "*"), recursive=True)
            if os.path.isfile(f)
        )
    elif any(c in path for c in "*?["):
        candidates = sorted(glob.glob(path, recursive=True))
    else:
        candidates = [path]

    files, skipped = [], []
    for f in candidates:
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
            files.append(f)
        else:
            skipped.append(f)
    if skipped:
        print(f"[info] Skipping {len(skipped)} unsupported file(s): "
              + ", ".join(os.path.basename(s) for s in skipped[:10])
              + (" ..." if len(skipped) > 10 else ""))
    return files


def print_language_summary(results):
    """Print a per-language summary so the run yields a clear verdict."""
    by_lang = {}
    for data in results.values():
        if data.get("error"):
            continue
        lang = data["language"]
        by_lang.setdefault(lang, {"docs": 0, "entities": 0, "translated": False})
        by_lang[lang]["docs"] += 1
        by_lang[lang]["entities"] += len(data["entities_unique"])
        by_lang[lang]["translated"] = data["translated"]

    print("\n================ PER-LANGUAGE SUMMARY ================")
    print(f"{'lang':<8}{'route':<12}{'docs':<7}{'avg_entities':<14}")
    print("-" * 41)
    for lang, s in sorted(by_lang.items()):
        route = "translated" if s["translated"] else "direct"
        avg = s["entities"] / s["docs"] if s["docs"] else 0
        print(f"{lang:<8}{route:<12}{s['docs']:<7}{avg:<14.1f}")
    print("=" * 41)
    print("\nReview tip: for any 'translated' language with low avg_entities, open "
          "a document and check whether the NAMES survived translation.")


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    INPUT = os.path.join(_THIS_DIR, "..", "..", "..", "data", "ner_samples")

    models = load_models()
    print(f"Running on: {models[-1]}\n")

    files = collect_files(INPUT)
    if not files:
        print("No supported files found.")
        raise SystemExit(0)
    print(f"Processing {len(files)} file(s) in batches...\n")

    results = process_paths_batched(models, files)

    with open("ner_output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for fname, data in results.items():
        if data.get("error"):
            print(f"\n=== {fname} ===  [ERROR] {data['error']}")
            continue
        print(f"\n=== {fname} ===")
        for e in data["entities_unique"]:
            print(f"  {e['text']:<40} {e['type']:<24} {e['score']}")

    print_language_summary(results)