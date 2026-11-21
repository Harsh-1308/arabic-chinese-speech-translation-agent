"""Document text extraction: PDF, DOCX and images (OCR), plus chunking for translation.

Everything runs locally:
- PDF:    PyMuPDF reads the text layer; pages without one (scans) are rendered and OCR'd.
- DOCX:   python-docx reads paragraphs and tables in document order.
- Images: RapidOCR (PaddleOCR models on ONNX Runtime) with separate Chinese and Arabic
          recognition models. Images are upscaled/greyscaled first when that helps.

Output is a list of Chunks: paragraph-sized pieces (~CHUNK_CHARS) that the agent translates one
at a time, so long documents stream results and stay within the model's context window.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config

PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_DOC_EXTENSIONS = PDF_EXTS | DOCX_EXTS | IMAGE_EXTS

ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
CHINESE_RE = re.compile(r"[一-鿿㐀-䶿]")


class DocumentError(ValueError):
    """Raised when a document fails a guardrail check or cannot be read."""


@dataclass
class Block:
    """A unit of extracted text (paragraph, table row or OCR line group) with its origin."""
    text: str
    where: str  # e.g. "page 2", "page 2 (OCR)", "table 1", "image (OCR)"
    ocr_confidence: float | None = None  # mean OCR score, None if not OCR'd


@dataclass
class Chunk:
    """A piece of text to translate in one agent call."""
    text: str
    where: str
    lang: str | None  # "ar" / "zh" / None (nothing to translate: passed through as-is)
    ocr_confidence: float | None = None


@dataclass
class ExtractedDocument:
    kind: str  # "pdf" | "docx" | "image"
    pages: int
    ocr_pages: int
    chunks: list[Chunk] = field(default_factory=list)


# ---------------------------------------------------------------- validation
def validate_document(path: str | Path) -> Path:
    p = Path(str(path).strip().strip('"')).expanduser().resolve()
    if not p.is_file():
        raise DocumentError(f"File not found: {p}")
    if p.suffix.lower() not in ALLOWED_DOC_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".") for e in ALLOWED_DOC_EXTENSIONS))
        raise DocumentError(f"Unsupported document type '{p.suffix}'. Allowed: {allowed}")
    size = p.stat().st_size
    if size == 0:
        raise DocumentError("The document is empty.")
    if size > config.MAX_DOC_BYTES:
        raise DocumentError(f"The document is larger than {config.MAX_DOC_BYTES // (1024 * 1024)} MB.")
    return p


# ---------------------------------------------------------------- text normalisation
def normalize(text: str) -> str:
    """NFKC turns Arabic presentation forms (common in PDFs) back into normal letters;
    also tidies whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def script_language(text: str) -> str | None:
    """'ar' or 'zh' by dominant script, None if the text has neither."""
    ar, zh = len(ARABIC_RE.findall(text)), len(CHINESE_RE.findall(text))
    if not ar and not zh:
        return None
    return "ar" if ar >= zh else "zh"


# ---------------------------------------------------------------- OCR
_ocr_engines: dict[str, object] = {}
_ocr_lock = threading.Lock()  # engines are loaded once and used one call at a time


def _ocr_engine(lang: str):
    """Chinese engine: PP-OCRv6 (also reads Latin text). Arabic engine: PP-OCRv5 Arabic model.
    Models are downloaded on first use (a few MB each)."""
    if lang not in _ocr_engines:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

        params = {"Global.log_level": "warning"}
        if lang == "ar":
            params.update({"Rec.lang_type": LangRec.ARABIC, "Rec.ocr_version": OCRVersion.PPOCRV5,
                           "Rec.model_type": ModelType.MOBILE})
        elif lang == "latin_rec":
            # recognition only, for small crops (numbers inside Arabic lines). A separate instance
            # because per-call use_det=False overrides persist on a RapidOCR object.
            params.update({"Global.use_det": False, "Global.use_cls": False})
        _ocr_engines[lang] = RapidOCR(params=params)
    return _ocr_engines[lang]


def _prepare_image(img) -> np.ndarray:
    """Fix phone-photo rotation (EXIF) and upscale only *tiny* images.

    Tested on Arabic samples: greyscale conversion and moderate upscaling made recognition
    worse (one line became garbage), so the image is otherwise left as-is.
    """
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(img).convert("RGB")
    short = min(img.size)
    if short < 500:
        scale = min(3.0, 800 / short)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return np.ascontiguousarray(np.array(img))


Line = tuple[str, float, tuple[float, float, float]]  # (text, score, (top, left, right))


def _x_range(box) -> tuple[float, float]:
    xs = [p[0] for p in box]
    return min(xs), max(xs)


def _ocr_lines(image: np.ndarray, lang: str) -> list[Line]:
    """Run one OCR engine and return one entry per detected text line."""
    if lang == "ar":
        return _ocr_lines_arabic(image)
    with _ocr_lock:
        out = _ocr_engine(lang)(image)
    if out.txts is None:
        return []
    lines = []
    for box, txt, score in zip(out.boxes, out.txts, out.scores):
        left, right = _x_range(box)
        lines.append((txt, float(score), (min(p[1] for p in box), left, right)))
    return lines


# Tokens the Latin/Chinese model may recover from gaps inside Arabic lines
_RECOVERABLE = re.compile(r"^[0-9A-Za-z.,:;%/+\-()$€£]+$")


def _ocr_lines_arabic(image: np.ndarray) -> list[Line]:
    """Arabic OCR with number recovery.

    The Arabic recognition model drops Western digits/Latin text that appear *inside* an Arabic
    line (it leaves a blank gap; tested: "عدد الموظفين 120 موظفاً" -> "عدد الموظفين  موظفاً").
    Per-word boxes show where those gaps are, so each wide gap is cropped and read with the
    Latin/Chinese model (which reads digits reliably), then put back in reading order.
    """
    with _ocr_lock:
        out = _ocr_engine("ar")(image, return_word_box=True)
        if out.txts is None:
            return []
        lines = []
        for box, line_txt, score, words in zip(out.boxes, out.txts, out.scores, out.word_results):
            ys = [p[1] for p in box]
            top, bottom = min(ys), max(ys)
            left, right = _x_range(box)
            if not words:
                lines.append((line_txt, float(score), (top, left, right)))
                continue

            # word texts come in visual order: reverse each to get logical Arabic order
            items = [(w[::-1], *_x_range(wbox)) for w, _s, wbox in words]
            items.sort(key=lambda it: it[1])  # left to right on the page
            height = bottom - top
            edges = [left] + [x for _, l, r in items for x in (l, r)] + [right]
            for gap_l, gap_r in zip(edges[0::2], edges[1::2]):  # spaces between/around words
                if gap_r - gap_l < 0.8 * height:  # a normal word space, nothing missing
                    continue
                y0, y1 = max(0, int(top) - 2), int(bottom) + 2
                crop = image[y0:y1, max(0, int(gap_l) - 2):int(gap_r) + 2]
                if crop.size == 0:
                    continue
                rec = _ocr_engine("latin_rec")(np.ascontiguousarray(crop))
                token = (rec.txts or [""])[0].strip()
                if token and _RECOVERABLE.match(token) and (rec.scores or [0])[0] >= 0.7:
                    token = re.sub(r"^%(\d[\d.,]*)$", r"\1%", token)  # RTL display shows 12% as %12
                    items.append((token, gap_l, gap_r))

            items.sort(key=lambda it: -it[2])  # Arabic reading order: right to left
            lines.append((" ".join(t for t, _, _ in items), float(score), (top, left, right)))
    return lines


def _group_lines(lines, lang: str) -> list[str]:
    """Sort OCR lines into reading order and group them into paragraphs by vertical gaps."""
    if not lines:
        return []
    # rows first (12px tolerance), then right-to-left for Arabic / left-to-right otherwise
    lines = sorted(lines, key=lambda l: (round(l[2][0] / 12), -l[2][2] if lang == "ar" else l[2][1]))
    paragraphs, current, last_top = [], [], None
    gap = np.median([abs(b[2][0] - a[2][0]) for a, b in zip(lines, lines[1:])] or [40]) * 1.8
    for txt, _score, (top, _l, _r) in lines:
        if last_top is not None and top - last_top > max(gap, 40):
            paragraphs.append(current)
            current = []
        current.append(txt)
        last_top = top
    paragraphs.append(current)
    # keep line breaks: they preserve headings/list structure and the LLM handles wrapped lines
    return ["\n".join(p).strip() for p in paragraphs if p]


def _rescale(image: np.ndarray, scale: float) -> np.ndarray:
    from PIL import Image

    im = Image.fromarray(image)
    return np.ascontiguousarray(np.array(im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)))


def _ocr_best_scale(image: np.ndarray, lang: str) -> list[Line]:
    """The Arabic model is sensitive to text size (tested: a 1.3x upscale turned a perfect line
    into garbage). If any line comes back with low confidence, retry at other scales and keep
    the result with the best overall quality. Chinese is robust, so it runs once."""
    lines = _ocr_lines(image, lang)
    if lang != "ar" or not lines or min(s for _, s, _ in lines) >= 0.8:
        return lines
    best, best_q = lines, _quality(lines, lang)
    for scale in (0.75, 1.5):
        alt = _ocr_lines(_rescale(image, scale), lang)
        if alt and (q := _quality(alt, lang)) > best_q:
            best, best_q = alt, q
    return best


def _quality(lines: list[Line], lang: str) -> float:
    """Recognised target-script characters weighted by confidence."""
    pattern = ARABIC_RE if lang == "ar" else CHINESE_RE
    return sum(len(pattern.findall(t)) * s for t, s, _ in lines)


def ocr_image(image: np.ndarray, lang_hint: str | None = None) -> tuple[list[str], float, str | None]:
    """OCR an image. Without a hint, both engines run and the one whose output contains more
    Arabic/Chinese characters (weighted by confidence) wins.

    Returns (paragraphs, mean_confidence, language).
    """
    candidates = [lang_hint] if lang_hint in ("ar", "zh") else ["zh", "ar"]
    best = ([], 0.0, None, -1.0)
    for lang in candidates:
        lines = _ocr_best_scale(image, lang)
        if not lines:
            continue
        mean = float(np.mean([s for _, s, _ in lines]))
        quality = _quality(lines, lang)
        if quality > best[3]:
            best = (_group_lines(lines, lang), mean, lang, quality)
    return best[0], best[1], best[2]


# ---------------------------------------------------------------- extractors
def _page_image(doc, page) -> np.ndarray:
    """Image of a scanned page for OCR.

    A typical scan is one image covering the page: use it at its native resolution (re-rendering
    rescales the text, which hurts Arabic OCR). Otherwise render the page at OCR_DPI.
    """
    import io

    from PIL import Image

    images = page.get_images(full=True)
    if len(images) == 1:
        rects = page.get_image_rects(images[0][0])
        page_area = page.rect.width * page.rect.height
        if rects and rects[0].width * rects[0].height >= 0.6 * page_area:
            try:
                data = doc.extract_image(images[0][0])["image"]
                with Image.open(io.BytesIO(data)) as im:
                    return _prepare_image(im)
            except Exception:
                pass  # unusual image encoding: fall back to rendering
    pix = page.get_pixmap(dpi=config.OCR_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    return np.ascontiguousarray(img)


def _extract_pdf(path: Path, lang_hint: str | None) -> tuple[list[Block], int, int]:
    import pymupdf

    try:
        doc = pymupdf.open(path)
    except Exception:  # don't echo the library message: it contains the server's temp path
        raise DocumentError("Could not open the PDF: the file is damaged or not a real PDF.")
    with doc:
        if doc.needs_pass:
            raise DocumentError("The PDF is password-protected.")
        if doc.page_count > config.MAX_DOC_PAGES:
            raise DocumentError(f"The PDF has {doc.page_count} pages; the limit is {config.MAX_DOC_PAGES}.")
        blocks, ocr_pages = [], 0
        for i, page in enumerate(doc, start=1):
            # sort=True gives reading order (top-to-bottom, then left-to-right) for text blocks
            texts = [normalize(b[4]) for b in page.get_text("blocks", sort=True) if b[6] == 0]
            texts = [t for t in texts if t]
            if sum(len(t) for t in texts) >= 20:
                blocks += [Block(t, f"page {i}") for t in texts]
                continue
            # No usable text layer: treat as a scanned page
            paras, conf, _ = ocr_image(_page_image(doc, page), lang_hint)
            if paras:
                ocr_pages += 1
                blocks += [Block(normalize(p), f"page {i} (OCR)", conf) for p in paras]
        return blocks, doc.page_count, ocr_pages


def _extract_docx(path: Path) -> tuple[list[Block], int, int]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = Document(str(path))
    except Exception:
        raise DocumentError("Could not open the Word file: it is damaged or not a .docx document.")
    blocks, table_no = [], 0
    for item in doc.iter_inner_content():  # paragraphs and tables in document order
        if isinstance(item, Paragraph):
            t = normalize(item.text)
            if t:
                blocks.append(Block(t, "paragraph"))
        elif isinstance(item, Table):
            table_no += 1
            for row in item.rows:
                cells, seen = [], set()
                for c in row.cells:  # merged cells repeat; keep each once
                    if id(c._tc) not in seen:
                        seen.add(id(c._tc))
                        cells.append(normalize(c.text))
                line = " | ".join(c for c in cells if c)
                if line:
                    blocks.append(Block(line, f"table {table_no}"))
    return blocks, 1, 0


def _extract_image(path: Path, lang_hint: str | None) -> tuple[list[Block], int, int]:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            arr = _prepare_image(img)
    except (UnidentifiedImageError, OSError):
        raise DocumentError("Could not read the image: the file is damaged or not a supported image.")
    paras, conf, _ = ocr_image(arr, lang_hint)
    return [Block(normalize(p), "image (OCR)", conf) for p in paras], 1, 1


# ---------------------------------------------------------------- chunking
def _split_long(text: str, limit: int) -> list[str]:
    """Split text longer than `limit` at sentence ends (Arabic, Chinese and Latin punctuation)."""
    if len(text) <= limit:
        return [text]
    sentences = re.findall(r".+?(?:[.!?؟。！？؛;]+[\"'”’」』)]*\s*|$)", text, re.S)
    parts, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) > limit:
            parts.append(cur.strip())
            cur = ""
        cur += s
    if cur.strip():
        parts.append(cur.strip())
    # a single sentence longer than the limit is hard-split as a last resort
    return [p[i:i + limit] for p in parts for i in range(0, len(p), limit)]


def make_chunks(blocks: list[Block]) -> list[Chunk]:
    """Merge small neighbouring blocks (same place, same language) up to CHUNK_CHARS and split
    long ones, so each agent call gets a paragraph-sized piece of text."""
    limit = config.CHUNK_CHARS
    chunks: list[Chunk] = []
    for b in blocks:
        for piece in _split_long(b.text, limit):
            lang = script_language(piece)
            prev = chunks[-1] if chunks else None
            if (prev and prev.where == b.where and prev.lang == lang
                    and len(prev.text) + len(piece) + 1 <= limit):
                prev.text += "\n" + piece
            else:
                chunks.append(Chunk(piece, b.where, lang, b.ocr_confidence))
    return chunks


# ---------------------------------------------------------------- public API
def extract_document(path: str | Path, lang_hint: str | None = None) -> ExtractedDocument:
    """Validate, extract and chunk a document. Raises DocumentError on any problem."""
    p = validate_document(path)
    ext = p.suffix.lower()
    if ext in PDF_EXTS:
        kind, (blocks, pages, ocr_pages) = "pdf", _extract_pdf(p, lang_hint)
    elif ext in DOCX_EXTS:
        kind, (blocks, pages, ocr_pages) = "docx", _extract_docx(p)
    else:
        kind, (blocks, pages, ocr_pages) = "image", _extract_image(p, lang_hint)

    if not blocks:
        raise DocumentError("No readable text was found in the document.")
    total = sum(len(b.text) for b in blocks)
    if total > config.MAX_DOC_CHARS:
        raise DocumentError(f"The document has {total:,} characters of text; the limit is "
                            f"{config.MAX_DOC_CHARS:,} (about {config.MAX_DOC_CHARS // 2000} pages).")
    chunks = make_chunks(blocks)
    if not any(c.lang for c in chunks):
        raise DocumentError("No Arabic or Chinese text was found in the document.")
    return ExtractedDocument(kind, pages, ocr_pages, chunks)


def preload_ocr() -> None:
    """Load (and on first run download) both OCR engines."""
    with _ocr_lock:
        for name in ("zh", "ar", "latin_rec"):
            _ocr_engine(name)
