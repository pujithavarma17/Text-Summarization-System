"""
Evaluation metrics, both reported on a 0-100 scale.

- ROUGE (accuracy): word overlap between a summary and a human reference.
- Coherence: how smoothly a summary flows. Each sentence is embedded with
  MiniLM, and we average the cosine similarity of every pair of neighbouring
  sentences. Jumping between unrelated topics gives a low score.
"""

import numpy as np
from rouge_score import rouge_scorer

from summarizer.extractive import load_embedding_model
from summarizer.preprocessing import tokenize_sentences

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
_scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)


def rouge_scores(generated, reference):
    scores = _scorer.score(reference, generated)
    return {name: round(scores[name].fmeasure * 100, 2) for name in ROUGE_TYPES}


def coherence_score(summary):
    """Average similarity of neighbouring sentences (None if under 2 sentences)."""
    sentences = [s for s in tokenize_sentences(summary) if s.strip()]
    if len(sentences) < 2:
        return None
    vectors = load_embedding_model().encode(sentences, normalize_embeddings=True)
    neighbours = np.sum(vectors[:-1] * vectors[1:], axis=1)  # cosine of sentence i and i+1
    return round(float(neighbours.mean()) * 100, 2)


def average_scores(score_list):
    """Average each metric over all summaries, skipping missing values."""
    averages = {}
    for name in score_list[0] if score_list else ROUGE_TYPES:
        values = [s[name] for s in score_list if s.get(name) is not None]
        averages[name] = round(sum(values) / len(values), 2) if values else None
    return averages
