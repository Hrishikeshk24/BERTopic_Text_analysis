"""
predict.py
----------
Score NEW warning-signal comments against the FROZEN, previously fitted
models — no refitting. Scoring logic lives in scoring.py (shared with the
Gradio app); sentiment engines in sentiment_engines.py.

Usage:
    python predict.py --input data/new_comments.csv \
                      --output results/predictions.csv \
                      [--engine finbert|lexicon]

Input CSV requires: PD_ID, warning_signal_comment
Optional: pd_grade_before, pd_grade_after (migration fields derived if present)
"""
import argparse
import numpy as np
import pandas as pd
from scoring import load_artifacts, assign_topics
from sentiment_engines import score

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="results/predictions.csv")
parser.add_argument("--engine", choices=["finbert", "lexicon"], default="finbert")
args = parser.parse_args()

arts = load_artifacts()
df = pd.read_csv(args.input)
missing = {"PD_ID", "warning_signal_comment"} - set(df.columns)
if missing:
    raise SystemExit(f"Input is missing required columns: {missing}")
docs = df["warning_signal_comment"].astype(str).tolist()
print(f"Scoring {len(docs)} new comments (backend={arts.backend}, engine={args.engine})")

topic_out = assign_topics(docs, arts)
for col, vals in topic_out.items():
    df[col] = vals

s_labels, s_scores = score(docs, args.engine)
df["sentiment"] = s_labels
df["sentiment_score"] = s_scores

if {"pd_grade_before", "pd_grade_after"}.issubset(df.columns):
    df["grade_change"] = df["pd_grade_after"] - df["pd_grade_before"]
    df["migration"] = np.select(
        [df["grade_change"] > 0, df["grade_change"] < 0],
        ["Downgrade", "Upgrade"], default="No Change")

df.to_csv(args.output, index=False)
a = pd.Series(topic_out["topic_assignment"])
print(f"direct: {(a == 'direct').sum()}, nearest_centroid fallback: {(a == 'nearest_centroid').sum()}")
print(df["sentiment"].value_counts().to_string())
print(f"Saved {args.output}")
