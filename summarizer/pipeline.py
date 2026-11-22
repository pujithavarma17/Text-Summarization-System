"""Full pipeline for one document: read -> clean -> tokenize -> summarize -> extras."""

from summarizer.abstractive import abstractive_summary
from summarizer.extractive import rank_sentences, top_sentences
from summarizer.keywords import important_topics
from summarizer.preprocessing import split_sentences, split_words
from summarizer.text_cleaning import clean_text, extract_docx_text, extract_pdf_text

EXTRACTIVE_SENTENCES = {"short": 3, "medium": 5, "long": 8}
MIN_SENTENCES = 3


def read_file(name, data):
    """Raw text from the bytes of a PDF, DOCX or TXT file."""
    name = name.lower()
    try:
        if name.endswith(".pdf"):
            return extract_pdf_text(data)
        if name.endswith(".docx"):
            return extract_docx_text(data)
        return data.decode("utf-8", errors="ignore")
    except Exception as error:
        raise ValueError("Could not read this file. It may be damaged or password-protected.") from error


def summarize_document(raw_text, method="extractive", length="medium", ratio=None):
    """Summarize one document.

    method: "extractive" or "abstractive"
    length: "short" / "medium" / "long"
    ratio:  optional custom size (0.1-0.9 of the original), overrides length
    """
    text = clean_text(raw_text)
    sentences = split_sentences(text)
    if len(sentences) < MIN_SENTENCES:
        raise ValueError("Too short to summarize (needs at least 3 full sentences).")

    scores = rank_sentences(sentences)  # computed once, used for summary and key points

    if method == "extractive":
        n = max(1, round(ratio * len(sentences))) if ratio else EXTRACTIVE_SENTENCES[length]
        selected = top_sentences(sentences, scores, n)
        summary = " ".join(selected)
    else:
        summary, selected = abstractive_summary(text, length, ratio=ratio), None

    return {
        "summary": summary,
        "selected": selected,
        "sentences": sentences,
        "key_points": top_sentences(sentences, scores, 3, keep_order=False),
        "topics": important_topics(text, 6),
        "original_words": len(split_words(text)),
        "summary_words": len(split_words(summary)),
    }
