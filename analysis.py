"""
analysis.py
-----------
Stage 4: Tie topics + sentiment back to PD grade migration.

Input : results/doc_topics_sentiment.csv
Output: results/topic_migration_summary.csv   downgrade rate & avg notches by topic
        results/sentiment_migration.csv       sentiment vs migration crosstab
        results/topic_sentiment_matrix.csv    avg grade change: topic x sentiment
        results/topic_vs_true_theme.csv       BERTopic vs ground-truth validation
        results/charts/*.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/charts", exist_ok=True)
df = pd.read_csv("results/doc_topics_sentiment.csv")

# Derive migration fields if the extract only carries the two grades
if "grade_change" not in df.columns:
    df["grade_change"] = df["pd_grade_after"] - df["pd_grade_before"]
if "migration" not in df.columns:
    import numpy as np
    df["migration"] = np.select(
        [df["grade_change"] > 0, df["grade_change"] < 0],
        ["Downgrade", "Upgrade"], default="No Change")

def short(label, n=35):
    return label if len(label) <= n else label[:n - 1] + "…"

# ---- 1. Migration by topic --------------------------------------------------
g = df.groupby(["topic_id", "topic_label"]).agg(
    n=("PD_ID", "count"),
    downgrade_rate=("migration", lambda s: (s == "Downgrade").mean()),
    upgrade_rate=("migration", lambda s: (s == "Upgrade").mean()),
    avg_grade_change=("grade_change", "mean"),
    avg_grade_before=("pd_grade_before", "mean"),
    avg_grade_after=("pd_grade_after", "mean"),
    pct_negative_sentiment=("sentiment", lambda s: (s == "negative").mean()),
).reset_index().sort_values("avg_grade_change", ascending=False)
g = g.round(3)
g.to_csv("results/topic_migration_summary.csv", index=False)
print("=== Migration by topic (positive avg_grade_change = downgrade) ===")
print(g.to_string(index=False))

# ---- 2. Sentiment vs migration ---------------------------------------------
ct = pd.crosstab(df["sentiment"], df["migration"], normalize="index").round(3)
ct.to_csv("results/sentiment_migration.csv")
print("\n=== Sentiment vs migration (row %) ===")
print(ct.to_string())

# ---- 3. Topic x sentiment -> avg grade change ------------------------------
mat = df.pivot_table(index="topic_label", columns="sentiment",
                     values="grade_change", aggfunc="mean").round(2)
mat.to_csv("results/topic_sentiment_matrix.csv")

# ---- 4. Validation: BERTopic topics vs ground-truth themes -----------------
# Only possible on synthetic data; real extracts have no true_theme column.
if "true_theme" in df.columns:
    val = pd.crosstab(df["topic_label"], df["true_theme"])
    val.to_csv("results/topic_vs_true_theme.csv")
    purity = (val.max(axis=1) / val.sum(axis=1)).mean()
    print(f"\nMean topic purity vs ground truth: {purity:.1%}")
else:
    print("\nNo true_theme column (real data) — skipping ground-truth validation.")

# ---- Charts -----------------------------------------------------------------
plt.figure(figsize=(10, 6))
gs = g.sort_values("avg_grade_change")
colors = ["#2e7d32" if v < 0 else "#c62828" for v in gs["avg_grade_change"]]
plt.barh([short(l) for l in gs["topic_label"]], gs["avg_grade_change"], color=colors)
plt.xlabel("Average PD grade change (notches; + = downgrade)")
plt.title("PD grade migration by warning-signal topic")
plt.tight_layout()
plt.savefig("results/charts/avg_grade_change_by_topic.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
gs2 = g.sort_values("downgrade_rate")
plt.barh([short(l) for l in gs2["topic_label"]], gs2["downgrade_rate"] * 100,
         color="#c62828")
plt.xlabel("Downgrade rate (%)")
plt.title("Share of borrowers downgraded, by topic")
plt.tight_layout()
plt.savefig("results/charts/downgrade_rate_by_topic.png", dpi=150)
plt.close()

ax = pd.crosstab(df["sentiment"], df["migration"]).plot(
    kind="bar", stacked=True, figsize=(8, 5),
    color={"Downgrade": "#c62828", "No Change": "#9e9e9e", "Upgrade": "#2e7d32"})
ax.set_ylabel("Number of borrowers")
ax.set_title("PD migration outcome by comment sentiment")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/charts/sentiment_vs_migration.png", dpi=150)
plt.close()

sizes = df["topic_label"].value_counts()
plt.figure(figsize=(10, 6))
plt.barh([short(l) for l in sizes.index[::-1]], sizes.values[::-1], color="#1565c0")
plt.xlabel("Number of comments")
plt.title("Topic sizes (2,000 warning-signal comments)")
plt.tight_layout()
plt.savefig("results/charts/topic_sizes.png", dpi=150)
plt.close()

print("\nSaved 4 charts to results/charts/")
