"""
Extractive summarization with TextRank on sentence embeddings.

1. Turn every sentence into a vector with MiniLM (similar meaning -> similar vector).
2. Build a similarity matrix between all sentences.
3. Run PageRank: a sentence is important if it is similar to many important sentences.
4. Keep the top sentences in their original order.
"""

from functools import lru_cache

import numpy as np

from summarizer.preprocessing import split_sentences

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def pagerank(similarity, damping=0.85, iterations=100, tolerance=1e-6):
    """PageRank by power iteration: score = (1-d)/n + d * M^T * score."""
    n = len(similarity)
    np.fill_diagonal(similarity, 0)            # a sentence can't vote for itself
    similarity = np.clip(similarity, 0, None)  # ignore negative similarity

    row_sums = similarity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = similarity / row_sums         # each sentence's votes add up to 1

    scores = np.ones(n) / n
    for _ in range(iterations):
        new_scores = (1 - damping) / n + damping * transition.T @ scores
        if np.abs(new_scores - scores).sum() < tolerance:
            break
        scores = new_scores
    return scores


def rank_sentences(sentences):
    """One importance score per sentence."""
    embeddings = load_embedding_model().encode(sentences, normalize_embeddings=True)
    return pagerank(embeddings @ embeddings.T)  # dot product = cosine similarity


def top_sentences(sentences, scores, n, keep_order=True):
    """The n best sentences, in document order (or by importance)."""
    best = np.argsort(scores)[::-1][:n]
    if keep_order:
        best = sorted(best)
    return [sentences[i] for i in best]


def extractive_summary(text, num_sentences=5):
    """Returns (summary_text, selected_sentences)."""
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences), sentences
    selected = top_sentences(sentences, rank_sentences(sentences), num_sentences)
    return " ".join(selected), selected
