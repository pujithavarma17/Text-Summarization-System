"""
Abstractive summarization with DistilBART (fine-tuned on CNN/DailyMail).

BART is an encoder-decoder Transformer: the encoder reads the text and the
decoder writes a new summary word by word. The model reads at most 1024
tokens, so long text is split into chunks of whole sentences, each chunk is
summarized, and the results are joined (and compressed again if too long).
"""

import re
from functools import lru_cache

from summarizer.preprocessing import split_sentences, tokenize_sentences

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
MAX_CHUNK_TOKENS = 900                           # below the 1024-token limit
LENGTHS = {"short": 80, "medium": 150, "long": 250}  # whole-summary length in tokens
MIN_TOKENS_PER_CHUNK = 40
MAX_TOKENS_PER_CHUNK = 250                       # longer outputs start to ramble
BATCH_SIZE = 4                                   # chunks per model call


@lru_cache(maxsize=1)
def load_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def chunk_text(text, tokenizer):
    """Group whole sentences into chunks that fit in the model."""
    chunks, current, current_len = [], [], 0
    for sentence in split_sentences(text) or [text]:
        n_tokens = len(tokenizer.tokenize(sentence))
        if current and current_len + n_tokens > MAX_CHUNK_TOKENS:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def generate_summaries(texts, tokenizer, model, min_tokens, max_tokens):
    """Summarize a batch of chunks in one model call."""
    import torch

    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=4,                           # beam search over 4 drafts
            min_new_tokens=min_tokens,
            max_new_tokens=int(max_tokens * 1.3),  # room to finish the last sentence
            no_repeat_ngram_size=3,                # no repeated 3-word phrases
            repetition_penalty=1.2,                # no repeated single words
            length_penalty=2.0,
            early_stopping=True,
        )

    summaries = []
    for ids in output_ids:
        summary = tokenizer.decode(ids, skip_special_tokens=True).strip()
        summary = re.sub(r"\s+([.,!?;:])", r"\1", summary)  # "word ." -> "word."
        summaries.append(drop_unfinished_sentence(summary))
    return summaries


def drop_repetitive_sentences(summary):
    """Remove sentences that repeat a word pair, e.g.
    "Data structures are used to swap data structures." (a typical model failure)."""
    kept = []
    for sentence in tokenize_sentences(summary):
        words = re.findall(r"\w+", sentence.lower())
        pairs = list(zip(words, words[1:]))
        if len(pairs) == len(set(pairs)):
            kept.append(sentence)
    return " ".join(kept)


def drop_unfinished_sentence(summary):
    """Cut off a sentence that stopped mid-way at the length limit."""
    if summary.endswith((".", "!", "?", '"', "”")):
        return summary
    last_end = max(summary.rfind(". "), summary.rfind("! "), summary.rfind("? "))
    return summary[: last_end + 1] if last_end > 0 else summary + "."


def abstractive_summary(text, length="medium", ratio=None):
    """Summarize text of any length (preset length, or `ratio` of each chunk)."""
    tokenizer, model = load_model()
    target = LENGTHS[length]
    chunks = chunk_text(text, tokenizer)

    def limits(chunk, budget):
        chunk_len = len(tokenizer.tokenize(chunk))
        hi = max(MIN_TOKENS_PER_CHUNK, min(budget, MAX_TOKENS_PER_CHUNK, chunk_len))
        return min(hi // 2, chunk_len // 2), hi

    def budget(chunk):
        if ratio is not None:
            return int(len(tokenizer.tokenize(chunk)) * ratio)
        return target // len(chunks)  # the preset length is shared between chunks

    partial = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_limits = [limits(c, budget(c)) for c in batch]
        lo = min(l for l, _ in batch_limits)
        hi = max(h for _, h in batch_limits)
        partial += generate_summaries(batch, tokenizer, model, lo, hi)
    combined = " ".join(partial)

    # Still too long? Summarize the summaries once more.
    if ratio is None and len(tokenizer.tokenize(combined)) > target * 1.5:
        lo, hi = limits(combined, target)
        combined = generate_summaries([combined], tokenizer, model, lo, hi)[0]
    return drop_repetitive_sentences(combined) or combined
