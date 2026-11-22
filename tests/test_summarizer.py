"""Unit tests. Run: python -m pytest -q"""

import numpy as np

from summarizer.evaluation import rouge_scores
from summarizer.extractive import extractive_summary, pagerank
from summarizer.preprocessing import split_sentences, split_words
from summarizer.text_cleaning import clean_text

ARTICLE = (
    "The city council approved a new public park on Monday evening. "
    "The park will cost two million dollars and open next summer. "
    "Residents said they were happy to get more green space in the centre. "
    "The mayor said the park project will also create fifty local jobs. "
    "My neighbour's cat was sleeping in the sun all afternoon."
)


def test_clean_text_removes_page_numbers_and_joins_lines():
    raw = "This is a long\nsentence over two lines.\nPage 3\n\nNew para-\ngraph here."
    assert clean_text(raw) == "This is a long sentence over two lines.\nNew paragraph here."


def test_split_sentences_handles_abbreviations():
    sentences = split_sentences("Dr. Smith works at the U.S. office. He likes it there a lot.")
    assert sentences == ["Dr. Smith works at the U.S. office.", "He likes it there a lot."]


def test_split_words_ignores_punctuation():
    assert split_words("Hello, world! It's fine.") == ["Hello", "world", "It's", "fine"]


def test_pagerank_gives_central_sentence_highest_score():
    # Sentence 0 is similar to both others; 1 and 2 are not similar to each other
    similarity = np.array([[1, 0.9, 0.9], [0.9, 1, 0.0], [0.9, 0.0, 1]], dtype=float)
    scores = pagerank(similarity)
    assert scores.argmax() == 0
    assert abs(scores.sum() - 1) < 1e-6


def test_extractive_summary_keeps_order_and_drops_off_topic_sentence():
    summary, selected = extractive_summary(ARTICLE, num_sentences=3)
    assert len(selected) == 3
    assert "cat" not in summary  # the off-topic sentence is not chosen
    positions = [ARTICLE.index(s) for s in selected]
    assert positions == sorted(positions)  # original order is kept


def test_rouge_identical_text_scores_100():
    assert rouge_scores("the cat sat", "the cat sat") == {"rouge1": 100.0, "rouge2": 100.0, "rougeL": 100.0}


def test_noise_filter_drops_headings_captions_and_links():
    from summarizer.preprocessing import is_noise

    assert is_noise("Department of Computer Science and Engineering")  # title
    assert is_noise("PROPOSED SOLUTION AND RESULTS 4.1.")               # heading
    assert is_noise("John A. Smith (Student ID.")                       # names
    assert is_noise("Figure 2: Screenshot of the sleep() function")     # caption
    assert is_noise("See https://example.com/docs for the code.")       # link
    assert not is_noise("The scheduler selects the runnable process for execution.")
    assert not is_noise("The president met the prime minister in Berlin on Monday.")


def test_split_sentences_rejoins_sentence_split_after_quote():
    text = "The terminal printed “Hello there!!” was also shown on the screen today."
    assert len(split_sentences(text)) == 1


def test_slide_bullets_become_separate_sentences_and_duplicates_are_removed():
    raw = "• The disk arm moves across the disk. • It serves requests in order.\n• The disk arm moves across the disk."
    assert split_sentences(clean_text(raw)) == [
        "The disk arm moves across the disk.",
        "It serves requests in order.",
    ]


def test_exports_are_organized():
    from io import BytesIO

    from openpyxl import load_workbook

    from summarizer.export import to_excel, to_txt

    results = [
        {"name": "a.pdf", "method": "Extractive", "summary": "First point here. Second point here.",
         "key_points": ["First point here."], "topics": ["Points"],
         "original_words": 100, "summary_words": 6},
        {"name": "bad.pdf", "error": "Could not read this file."},
    ]
    txt = to_txt(results, "Bullet Points")
    assert "  • First point here." in txt and "  • Second point here." in txt
    assert "NOT SUMMARIZED: Could not read this file." in txt

    workbook = load_workbook(BytesIO(to_excel(results)))
    assert workbook.sheetnames == ["Overview", "Summaries"]
    assert workbook["Overview"].max_row == 3      # header + 2 files
    assert workbook["Summaries"].max_row == 3     # header + 2 summary points


def test_coherence_is_higher_for_a_summary_that_stays_on_topic():
    from summarizer.evaluation import coherence_score

    on_topic = ("The council approved a new park. The park will open next summer. "
                "Families are happy about the new green space.")
    jumping = ("The council approved a new park. Bitcoin prices fell sharply today. "
               "The recipe needs two cups of flour.")
    assert coherence_score(on_topic) > coherence_score(jumping)
    assert coherence_score("Only one sentence here.") is None


def test_repetitive_generated_sentences_are_removed():
    from summarizer.abstractive import drop_repetitive_sentences

    text = "Disk requests are scheduled. Data structures are used to swap data structures."
    assert drop_repetitive_sentences(text) == "Disk requests are scheduled."
