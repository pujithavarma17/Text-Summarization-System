"""Tokenization: split text into sentences and words, and drop noisy lines."""

import re

import nltk

MIN_SENTENCE_WORDS = 4  # shorter "sentences" are usually headings or noise


def _punkt_available():
    """Download NLTK's sentence tokenizer data once; fall back to regex if offline."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
        return True
    except LookupError:
        try:
            return nltk.download("punkt_tab", quiet=True)
        except Exception:
            return False


_USE_NLTK = _punkt_available()


def tokenize_sentences(text):
    """Plain sentence split, no filtering (used to show a summary as bullets)."""
    if _USE_NLTK:
        return nltk.sent_tokenize(text)
    return re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text)


def split_sentences(text):
    """Split text into clean, unique sentences.

    NLTK's Punkt tokenizer knows that "Dr." or "U.S." does not end a sentence.
    """
    sentences = []
    for paragraph in text.splitlines():  # one paragraph per line after cleaning
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for part in tokenize_sentences(paragraph):
            part = part.strip()
            if not part:
                continue
            if sentences and part[0].islower():  # wrongly split, e.g. after a quote
                sentences[-1] += " " + part
            else:
                sentences.append(part)

    unique, seen = [], set()
    for s in sentences:
        key = s.lower()
        if key not in seen and len(s.split()) >= MIN_SENTENCE_WORDS and not is_noise(s):
            seen.add(key)  # repeated slides give the same sentence twice
            unique.append(s)
    return unique


def is_noise(sentence):
    """True for headings, title/name lines, figure captions and links."""
    if re.search(r"https?://|www\.", sentence):
        return True
    if re.match(r"(Figure|Fig\.|Table)\s*\d", sentence, re.IGNORECASE):
        return True

    words = re.findall(r"[A-Za-z][A-Za-z.'-]*", sentence)
    if not words:
        return True

    capitalized = sum(w[0].isupper() for w in words) / len(words)
    all_caps = sum(w.isupper() and len(w) > 1 for w in words) / len(words)
    ends_like_sentence = sentence.rstrip("\"'”’)").endswith((".", "!", "?", ":"))

    return (
        all_caps >= 0.5                                   # "PROPOSED SOLUTION IN xv6"
        or capitalized >= 0.8                             # "John A. Smith (Student ID."
        or (not ends_like_sentence and capitalized >= 0.5)  # "Department of Computer Science"
    )


def split_words(text):
    """Split text into words (punctuation removed)."""
    return re.findall(r"\b\w+(?:'\w+)?\b", text)
