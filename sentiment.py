"""
sentiment.py
------------
Stage 3: Sentiment scoring of warning-signal comments.

Input : results/doc_topics.csv
Output: results/doc_topics_sentiment.csv (adds sentiment label + score)

Engines (--engine flag), implemented in sentiment_engines.py and shared
with predict.py:
    finbert  ProsusAI/finbert (recommended) — needs model weights
    lexicon  VADER + Loughran-McDonald-style credit lexicon (offline)
"""
import argparse
import pandas as pd
from sentiment_engines import score

parser = argparse.ArgumentParser()
parser.add_argument("--engine", choices=["finbert", "lexicon"], default="finbert")
args = parser.parse_args()

df = pd.read_csv("results/doc_topics.csv")
labels, scores = score(df["warning_signal_comment"].tolist(), args.engine)
df["sentiment"] = labels
df["sentiment_score"] = scores

df.to_csv("results/doc_topics_sentiment.csv", index=False)
print(df["sentiment"].value_counts().to_string())
print("Saved results/doc_topics_sentiment.csv")
