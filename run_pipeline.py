"""
run_pipeline.py — orchestrates the four stages end to end.

Usage:
    python run_pipeline.py                       # transformer stack (MiniLM + FinBERT)
    python run_pipeline.py --offline             # no-download fallback (TF-IDF + lexicon)
    python run_pipeline.py --skip-datagen        # use your own data/warning_signals.csv

To run on REAL data: place a CSV at data/warning_signals.csv with columns
    PD_ID, warning_signal_comment, pd_grade_before, pd_grade_after
(grade_change / migration are recomputed if absent) and pass --skip-datagen.
"""
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--offline", action="store_true",
                    help="TF-IDF embeddings + lexicon sentiment (no model downloads)")
parser.add_argument("--skip-datagen", action="store_true")
args = parser.parse_args()

emb = "tfidf" if args.offline else "minilm"
eng = "lexicon" if args.offline else "finbert"

steps = []
if not args.skip_datagen:
    steps.append([sys.executable, "generate_data.py"])
steps += [
    [sys.executable, "topic_model.py", "--embedding", emb],
    [sys.executable, "sentiment.py", "--engine", eng],
    [sys.executable, "analysis.py"],
]
for cmd in steps:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
print("\nPipeline complete. See results/ and results/charts/.")
