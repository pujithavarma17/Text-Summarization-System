"""
Important topics (KeyBERT idea): candidate 1-2 word phrases are embedded with
MiniLM, and the phrases closest in meaning to the whole document win.
"""

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from summarizer.extractive import load_embedding_model

MAX_CANDIDATES = 40


def candidate_phrases(text):
    """Frequent 1-2 word phrases without stop words."""
    words = re.findall(r"[A-Za-z][A-Za-z-]+", text.lower())
    counts = Counter()
    for i, word in enumerate(words):
        if word in ENGLISH_STOP_WORDS or len(word) < 3:
            continue
        counts[word] += 1
        if i + 1 < len(words) and words[i + 1] not in ENGLISH_STOP_WORDS and len(words[i + 1]) >= 3:
            counts[f"{word} {words[i + 1]}"] += 1

    # A 2-word phrase must appear at least twice to count
    return [p for p, c in counts.most_common() if " " not in p or c >= 2][:MAX_CANDIDATES]


def important_topics(text, top_n=6):
    candidates = candidate_phrases(text)
    if not candidates:
        return []

    model = load_embedding_model()
    doc_vector = model.encode([text], normalize_embeddings=True)[0]
    similarity = model.encode(candidates, normalize_embeddings=True) @ doc_vector

    topics = []
    for idx in similarity.argsort()[::-1]:
        phrase = candidates[idx]
        if any(set(phrase.split()) & set(t.split()) for t in topics):
            continue  # skip overlaps like "model" vs "deep learning model"
        topics.append(phrase)
        if len(topics) == top_n:
            break
    return [t.title() for t in topics]
