"""
Evaluate on CNN/DailyMail: accuracy (ROUGE) and coherence for
Lead-3 baseline vs Extractive vs Abstractive (+ human summaries as a reference).

    python evaluate.py --samples 50 [--skip-abstractive]
"""

import argparse
import json
import time

from datasets import load_dataset
from tqdm import tqdm

from summarizer.abstractive import abstractive_summary
from summarizer.evaluation import average_scores, coherence_score, rouge_scores
from summarizer.extractive import extractive_summary
from summarizer.preprocessing import split_sentences


def lead_3(text):
    return " ".join(split_sentences(text)[:3])


def evaluate(summarize, data, name):
    scores, seconds = [], 0.0
    for example in tqdm(data, desc=name):
        start = time.time()
        summary = summarize(example["article"])
        seconds += time.time() - start
        scores.append({
            **rouge_scores(summary, example["highlights"]),
            "coherence": coherence_score(summary),
        })
    result = average_scores(scores)
    result["seconds_per_article"] = round(seconds / len(data), 2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--skip-abstractive", action="store_true")
    args = parser.parse_args()

    data = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{args.samples}]")

    methods = {
        "Lead-3 baseline": lead_3,
        "Extractive (TextRank)": lambda text: extractive_summary(text, 3)[0],
    }
    if not args.skip_abstractive:
        methods["Abstractive (DistilBART)"] = lambda text: abstractive_summary(text, "short")

    results = {name: evaluate(summarize, data, name) for name, summarize in methods.items()}

    # Coherence of the human-written summaries, as a reference point
    human = [coherence_score(example["highlights"]) for example in data]
    human = [c for c in human if c is not None]
    results["Human reference"] = {"coherence": round(sum(human) / len(human), 2)}

    print(f"\nScores (0-100) on {len(data)} CNN/DailyMail test articles\n")
    print(f"{'Method':<28}{'ROUGE-1':>9}{'ROUGE-2':>9}{'ROUGE-L':>9}{'Coherence':>11}{'sec/doc':>9}")
    for name, r in results.items():
        cells = [r.get(k, "-") for k in ["rouge1", "rouge2", "rougeL"]]
        print(f"{name:<28}{cells[0]:>9}{cells[1]:>9}{cells[2]:>9}"
              f"{r['coherence']:>11}{r.get('seconds_per_article', '-'):>9}")

    with open("results/evaluation_results.json", "w") as f:
        json.dump({"samples": len(data), "results": results}, f, indent=2)
    print("\nSaved to results/evaluation_results.json")


if __name__ == "__main__":
    main()
