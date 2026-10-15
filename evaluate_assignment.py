"""
evaluate_assignment.py
-----------------------
Honest accuracy evaluation of topic ASSIGNMENT (scoring.assign_topics, the
function predict.py/app.py actually use to score new comments) -- not to be
confused with the training-time topic-purity check in analysis.py, which
only measures how well BERTopic's own fit-time clusters line up with
ground truth on the data they were fit on.

Methodology (held-out, not in-sample):
    1. Stratified 80/20 split of data/warning_signals.csv by true_theme.
    2. Fit a SEPARATE eval-only BERTopic model on the 80% train split only
       (models_eval/, results_eval/ -- production models/ and results/ are
       never touched). The held-out 20% is never seen during training.
    3. Build a topic_id -> true_theme mapping from the TRAIN split's own
       fitted assignments (majority vote), so no holdout label leaks into
       the mapping itself.
    4. Score the held-out comments twice through scoring.assign_topics():
       once with use_llm=False (pure statistical baseline: direct HDBSCAN
       cluster, or nearest centroid for outliers) and once with
       use_llm=True (every comment re-checked/fine-tuned by the local LLM,
       per the --llm behavior in scoring.py).
    5. Map each prediction's topic_id through the mapping and compare to
       the comment's actual true_theme.

Metrics reported for BOTH baseline and LLM-fine-tuned, plus the baseline
vs LLM lift:
    accuracy, precision/recall/F1 (macro + weighted) per theme,
    confusion matrix, agreement rate (how often the LLM changed the pick),
    and accuracy specifically on the subset the LLM reassigned (did
    overriding the statistical pick help or hurt, net).

Usage:
    python evaluate_assignment.py                  # full run (trains eval model if needed)
    python evaluate_assignment.py --retrain         # force-refit the eval model
    python evaluate_assignment.py --skip-llm         # baseline-only (fast, no Ollama calls)
    python evaluate_assignment.py --sample 100        # LLM pass on a random 100-row subsample
                                                        # of the holdout (quick iteration)

Note: the LLM pass is sequential (the local Ollama server processes one
request at a time regardless of client concurrency) -- expect roughly
1-2 seconds per comment, i.e. several minutes for a few hundred held-out
rows.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from scoring import load_artifacts, assign_topics

DATA = "data/warning_signals.csv"
TRAIN_SPLIT = "data/_eval_train.csv"
MODELS_EVAL = "models_eval"
RESULTS_EVAL = "results_eval"

parser = argparse.ArgumentParser()
parser.add_argument("--embedding", choices=["minilm", "tfidf"], default="minilm")
parser.add_argument("--llm-model", default=None)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument("--retrain", action="store_true",
                    help="Force-refit the eval-only model even if models_eval/ exists.")
parser.add_argument("--skip-llm", action="store_true",
                    help="Only compute the statistical baseline (no Ollama calls).")
parser.add_argument("--sample", type=int, default=None,
                    help="Evaluate the LLM pass on a random subsample of the holdout "
                         "set (baseline still runs on all of it) for faster iteration.")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# 1-2. Split + train the eval-only model (skip if already there)
# ---------------------------------------------------------------------------
os.makedirs(RESULTS_EVAL, exist_ok=True)
holdout_path = f"{RESULTS_EVAL}/holdout_comments.csv"
need_train = args.retrain or not os.path.exists(f"{MODELS_EVAL}/config.json")

if need_train:
    df = pd.read_csv(DATA)
    train_df, holdout_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state,
        stratify=df["true_theme"])
    train_df.to_csv(TRAIN_SPLIT, index=False)
    holdout_df.to_csv(holdout_path, index=False)
    print(f"Split {len(df)} comments -> {len(train_df)} train / {len(holdout_df)} holdout "
          f"(stratified by true_theme, random_state={args.random_state})")

    cmd = [sys.executable, "topic_model.py", "--embedding", args.embedding,
           "--data", TRAIN_SPLIT, "--models-dir", MODELS_EVAL, "--results-dir", RESULTS_EVAL,
           "--llm"]
    if args.llm_model:
        cmd += ["--llm-model", args.llm_model]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
else:
    print(f"Reusing existing eval model at {MODELS_EVAL}/ (pass --retrain to refit)")

holdout_df = pd.read_csv(holdout_path)

# ---------------------------------------------------------------------------
# 3. topic_id -> true_theme mapping, built ONLY from the train split's own
#    fitted assignments (no holdout leakage)
# ---------------------------------------------------------------------------
train_assigned = pd.read_csv(f"{RESULTS_EVAL}/doc_topics.csv")
topic_to_theme = (train_assigned.groupby("topic_id")["true_theme"]
                   .agg(lambda s: s.value_counts().idxmax())
                   .to_dict())
mapping_confidence = (train_assigned.groupby("topic_id")["true_theme"]
                       .agg(lambda s: s.value_counts(normalize=True).iloc[0])
                       .to_dict())
print(f"\nTopic -> true_theme mapping built from {len(train_assigned)} train rows, "
      f"{len(topic_to_theme)} topics. Mean mapping confidence (majority-vote purity): "
      f"{np.mean(list(mapping_confidence.values())):.1%}")


def predicted_theme(topic_id):
    return topic_to_theme.get(topic_id, "UNMAPPED")


# ---------------------------------------------------------------------------
# 4. Score the holdout set
# ---------------------------------------------------------------------------
arts = load_artifacts(MODELS_EVAL)
docs = holdout_df["warning_signal_comment"].astype(str).tolist()
pd_ids = holdout_df["PD_ID"].tolist()
true_themes = holdout_df["true_theme"].tolist()

print(f"\nScoring {len(docs)} held-out (never-trained-on) comments -- statistical baseline...")
baseline = assign_topics(docs, arts, use_llm=False)

llm = None
llm_idx = list(range(len(docs)))
if not args.skip_llm:
    if args.sample and args.sample < len(docs):
        rng = np.random.RandomState(args.random_state)
        llm_idx = sorted(rng.choice(len(docs), size=args.sample, replace=False).tolist())
        print(f"Scoring a random subsample of {len(llm_idx)}/{len(docs)} holdout comments "
              f"with --llm (every comment fine-tuned by the LLM)...")
    else:
        print(f"Scoring all {len(docs)} holdout comments with --llm "
              f"(every comment fine-tuned by the LLM; this is the slow part)...")
    llm_docs = [docs[i] for i in llm_idx]
    llm = assign_topics(llm_docs, arts, use_llm=True)
    print(pd.Series(llm["topic_assignment"]).value_counts().to_string())

# ---------------------------------------------------------------------------
# 5-6. Metrics
# ---------------------------------------------------------------------------
def theme_metrics(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)))
    print(f"\n=== {label}: accuracy = {acc:.1%} (n={len(y_true)}) ===")
    print(f"macro F1 = {report['macro avg']['f1-score']:.3f}   "
          f"weighted F1 = {report['weighted avg']['f1-score']:.3f}")
    return acc, report, cm


results = {}

base_pred_theme = [predicted_theme(t) for t in baseline["topic_id"]]
base_acc, base_report, base_cm = theme_metrics(true_themes, base_pred_theme, "Baseline (statistical)")
results["baseline"] = {
    "accuracy": base_acc, "report": base_report,
    "confusion_matrix_labels": sorted(set(true_themes) | set(base_pred_theme)),
    "confusion_matrix": base_cm.tolist(),
}

holdout_out = holdout_df.copy()
holdout_out["baseline_topic_id"] = baseline["topic_id"]
holdout_out["baseline_topic_label"] = baseline["topic_label"]
holdout_out["baseline_predicted_theme"] = base_pred_theme
holdout_out["baseline_correct"] = [t == p for t, p in zip(true_themes, base_pred_theme)]

if llm is not None:
    llm_true = [true_themes[i] for i in llm_idx]
    llm_pred_theme = [predicted_theme(t) for t in llm["topic_id"]]
    llm_acc, llm_report, llm_cm = theme_metrics(llm_true, llm_pred_theme, "LLM-fine-tuned (always-on)")
    results["llm"] = {
        "accuracy": llm_acc, "report": llm_report,
        "confusion_matrix_labels": sorted(set(llm_true) | set(llm_pred_theme)),
        "confusion_matrix": llm_cm.tolist(),
        "n_scored": len(llm_idx),
    }

    agreement = [a == b for a, b in zip(llm["baseline_topic_id"], llm["topic_id"])]
    reassigned_mask = [not a for a in agreement]
    n_reassigned = sum(reassigned_mask)
    print(f"\n=== LLM vs. statistical baseline (on the {len(llm_idx)} LLM-scored rows) ===")
    print(f"Agreement rate: {np.mean(agreement):.1%}  "
          f"({n_reassigned}/{len(llm_idx)} comments reassigned by the LLM)")

    if n_reassigned:
        base_on_llm_subset = [base_pred_theme[i] for i in llm_idx]
        reassigned_true = [llm_true[i] for i in range(len(llm_idx)) if reassigned_mask[i]]
        reassigned_base_pred = [base_on_llm_subset[i] for i in range(len(llm_idx)) if reassigned_mask[i]]
        reassigned_llm_pred = [llm_pred_theme[i] for i in range(len(llm_idx)) if reassigned_mask[i]]
        base_acc_on_reassigned = accuracy_score(reassigned_true, reassigned_base_pred)
        llm_acc_on_reassigned = accuracy_score(reassigned_true, reassigned_llm_pred)
        print(f"On just the {n_reassigned} comments the LLM changed: "
              f"baseline was right {base_acc_on_reassigned:.1%} of the time, "
              f"LLM's override was right {llm_acc_on_reassigned:.1%} of the time "
              f"({'net improvement' if llm_acc_on_reassigned > base_acc_on_reassigned else 'net regression' if llm_acc_on_reassigned < base_acc_on_reassigned else 'no net change'}).")
        results["llm_override_effect"] = {
            "n_reassigned": n_reassigned,
            "baseline_accuracy_on_reassigned": base_acc_on_reassigned,
            "llm_accuracy_on_reassigned": llm_acc_on_reassigned,
        }
    results["agreement_rate"] = float(np.mean(agreement))

    full = holdout_out.copy()
    for col in ["llm_topic_id", "llm_topic_label", "llm_topic_assignment",
                "llm_predicted_theme", "llm_correct"]:
        full[col] = np.nan
    for pos, i in enumerate(llm_idx):
        full.loc[full.index[i], "llm_topic_id"] = llm["topic_id"][pos]
        full.loc[full.index[i], "llm_topic_label"] = llm["topic_label"][pos]
        full.loc[full.index[i], "llm_topic_assignment"] = llm["topic_assignment"][pos]
        full.loc[full.index[i], "llm_predicted_theme"] = llm_pred_theme[pos]
        full.loc[full.index[i], "llm_correct"] = (true_themes[i] == llm_pred_theme[pos])
    holdout_out = full

    lift = llm_acc - accuracy_score([true_themes[i] for i in llm_idx],
                                     [base_pred_theme[i] for i in llm_idx])
    print(f"\nAccuracy lift from always-on LLM fine-tuning (same {len(llm_idx)} rows): "
          f"{lift:+.1%}  (baseline {accuracy_score([true_themes[i] for i in llm_idx], [base_pred_theme[i] for i in llm_idx]):.1%} -> LLM {llm_acc:.1%})")
    results["lift"] = float(lift)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
holdout_out.to_csv(f"{RESULTS_EVAL}/assignment_eval_predictions.csv", index=False)
with open(f"{RESULTS_EVAL}/assignment_eval_report.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved {RESULTS_EVAL}/assignment_eval_predictions.csv and assignment_eval_report.json")
