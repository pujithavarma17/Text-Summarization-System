"""Read text from PDF / DOCX files and clean it before summarizing."""

import re
from functools import lru_cache
from io import BytesIO

from pypdf import PdfReader

MIN_WORDS_FOR_TEXT_PAGE = 15  # fewer words -> the page is probably an image
PAGE_FOOTER = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE)
BULLET = re.compile(r"\s*[•●▪◦■►➢✓]\s*")


@lru_cache(maxsize=1)
def load_ocr():
    """RapidOCR reads text from images (None if it is not installed)."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    return RapidOCR()


def ocr_page(page):
    """Read the text inside the images of one PDF page."""
    ocr = load_ocr()
    if ocr is None:
        return ""

    import numpy as np

    lines = []
    for image_file in page.images:
        try:
            image = image_file.image.convert("RGB")
        except Exception:
            continue
        if image.width >= 100 and image.height >= 50:  # skip icons and logos
            result, _ = ocr(np.array(image))
            lines += [text for _, text, _ in (result or [])]
    return "\n".join(lines)


def extract_pdf_text(file_bytes):
    """Read text from a PDF. OCR is used only for scanned PDFs.

    A PDF is treated as scanned when most of its pages have no text.
    Normal PDFs and slide decks skip OCR, because reading every diagram
    is slow (~90 s per file) and adds little useful text.
    """
    pdf = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in pdf.pages:
        try:
            text = page.extract_text(extraction_mode="layout") or ""  # keeps lines intact
        except Exception:
            text = page.extract_text() or ""
        pages.append(text)

    text_pages = sum(len(t.split()) >= MIN_WORDS_FOR_TEXT_PAGE for t in pages)
    if text_pages < len(pages) * 0.25:  # scanned PDF: under 25% of pages have text
        pages = [
            t if len(t.split()) >= MIN_WORDS_FOR_TEXT_PAGE else f"{t}\n{ocr_page(p)}".strip()
            for t, p in zip(pages, pdf.pages)
        ]
    return "\n\n".join(pages)


def extract_docx_text(file_bytes):
    """Read the paragraphs of a Word (.docx) file."""
    from docx import Document

    document = Document(BytesIO(file_bytes))
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())


def clean_text(text):
    """Turn raw extracted text into clean paragraphs, one per line."""
    text = text.replace("�", "-")                       # characters that failed to decode
    text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)     # "synchroni-\nzation" -> "synchronization"
    text = BULLET.sub("\n\n", text)                     # every slide bullet starts a new paragraph

    paragraphs, current = [], []
    for line in text.splitlines():
        line = re.sub(r"\.{4,}\s*\d*", " ", line)       # table-of-contents dots: "Intro ..... 5"
        line = re.sub(r"\s+", " ", line).strip()
        if PAGE_FOOTER.match(line):                     # page numbers: "Page 3 of 7", "3"
            line = ""

        if line:
            current.append(line)
        elif current:                                   # a blank line ends the paragraph
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))

    # Join a short label like "Advantages:" with the text after it
    merged = []
    for paragraph in paragraphs:
        if merged and merged[-1].endswith(":") and len(merged[-1]) < 60:
            merged[-1] += " " + paragraph
        else:
            merged.append(paragraph)
    return "\n".join(merged)
