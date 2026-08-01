"""
sentiment_engines.py
--------------------
Shared sentiment scoring functions, used by both the batch stage
(sentiment.py) and new-data scoring (predict.py) so training and
prediction can never drift apart.

Each engine takes a list of strings and returns (labels, scores):
    labels: "positive" | "negative" | "neutral" per document
    scores: finbert -> class confidence (0..1)
            lexicon -> VADER compound polarity (-1..+1)
"""

# Loughran-McDonald-style credit vocabulary injected into VADER (-4..+4 scale).
CREDIT_LEXICON = {
    # negative
    "breach": -2.5, "breached": -2.5, "violation": -2.5, "default": -3.0,
    "delinquent": -2.5, "downgrade": -2.0, "impaired": -2.5, "impairment": -2.5,
    "waiver": -1.5, "restated": -2.0, "restatement": -2.0, "lawsuit": -2.0,
    "litigation": -1.8, "bankruptcy": -3.2, "insolvency": -3.2, "arrears": -2.5,
    "deteriorating": -2.2, "deterioration": -2.2, "declined": -1.5,
    "decline": -1.5, "contracted": -1.5, "erosion": -1.8, "shortfall": -2.0,
    "missed": -2.0, "overdue": -2.2, "understaffed": -1.5, "burn": -1.5,
    "squeeze": -1.6, "stretched": -1.4, "elevated": -1.0, "concern": -1.5,
    "concerns": -1.5, "weakness": -1.8, "weak": -1.5, "adverse": -2.0,
    "qualified": -1.5, "unresolved": -1.5, "headwinds": -1.8,
    "compression": -1.5, "compressed": -1.5, "turnover": -1.2,
    "resigned": -1.5, "departed": -1.2, "dispute": -1.8, "sanctions": -2.2,
    "investigation": -1.8, "remediation": -1.2, "distress": -2.5,
    "insufficient": -2.0, "negative": -1.8, "risk": -0.8, "exposure": -0.8,
    # positive
    "refinanced": 1.8, "deleveraging": 2.0, "paydown": 1.8, "prepayment": 1.5,
    "cured": 1.8, "improved": 1.8, "improving": 1.8, "recovery": 1.5,
    "recovered": 1.6, "turnaround": 1.8, "profitability": 1.5, "headroom": 1.0,
    "undrawn": 1.2, "stable": 1.0, "strengthened": 1.8, "traction": 1.3,
    "record": 1.5, "wins": 1.5, "diversifying": 1.2, "raised": 0.8,
    "ahead": 1.2, "comfortable": 1.4,
}


def score_finbert(docs, batch_size=32, max_length=256):
    """ProsusAI/finbert. Requires model weights (HuggingFace or internal mirror)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    MODEL = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    labels_ = [model.config.id2label[i] for i in range(model.config.num_labels)]

    all_labels, all_scores = [], []
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            enc = tok(list(docs[i:i + batch_size]), padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            all_labels += [labels_[j] for j in idx.tolist()]
            all_scores += [round(s, 4) for s in conf.tolist()]
    return all_labels, all_scores


def score_lexicon(docs, neg_threshold=-0.15, pos_threshold=0.15):
    """Offline fallback: VADER + credit-domain lexicon overrides."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(CREDIT_LEXICON)

    labels, scores = [], []
    for d in docs:
        c = analyzer.polarity_scores(d)["compound"]
        labels.append("negative" if c <= neg_threshold
                      else ("positive" if c >= pos_threshold else "neutral"))
        scores.append(round(c, 4))
    return labels, scores


def score(docs, engine):
    if engine == "finbert":
        return score_finbert(docs)
    if engine == "lexicon":
        return score_lexicon(docs)
    raise ValueError(f"unknown engine: {engine}")
