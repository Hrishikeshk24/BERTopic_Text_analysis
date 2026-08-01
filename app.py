"""
app.py — Gradio UI for scoring warning-signal comments against the frozen models.

Run from the pipeline/ directory (after topic_model.py has written models/):

    python app.py                     # http://127.0.0.1:7860
    APP_ENGINE=lexicon python app.py  # offline sentiment engine default

Tabs:
    Single comment  type/paste a comment, optional PD grades, Submit
    Batch CSV       upload a CSV (PD_ID, warning_signal_comment [, grades])

The app never refits anything — it loads models/ once at startup and scores
with exactly the same code path as predict.py (scoring.py + sentiment_engines).
"""
import os
import numpy as np
import pandas as pd
import gradio as gr
from scoring import load_artifacts, assign_topics
from sentiment_engines import score

DEFAULT_ENGINE = os.environ.get("APP_ENGINE", "finbert")

print("Loading frozen model artifacts from models/ ...")
ARTS = load_artifacts()

# Historical per-topic migration stats (training run) for context in results
try:
    HIST = pd.read_csv("results/topic_migration_summary.csv").set_index("topic_id")
except FileNotFoundError:
    HIST = None

GRADES = [str(i) for i in range(1, 15)]
SENT_COLOR = {"negative": "#c62828", "neutral": "#616161", "positive": "#2e7d32"}


def _topic_history_md(topic_id):
    if HIST is None or topic_id not in HIST.index:
        return "_No historical migration stats available (run analysis.py first)._"
    h = HIST.loc[topic_id]
    return (
        f"**Historical outcomes for this topic** (training data, n={int(h['n'])}):  \n"
        f"Downgrade rate **{h['downgrade_rate']:.0%}** · "
        f"Upgrade rate **{h['upgrade_rate']:.0%}** · "
        f"Avg grade change **{h['avg_grade_change']:+.2f} notches** · "
        f"Avg grade before → after: {h['avg_grade_before']:.1f} → {h['avg_grade_after']:.1f}"
    )


def score_single(comment, grade_before, grade_after, engine):
    comment = (comment or "").strip()
    if not comment:
        return "Please enter a comment.", ""

    t = assign_topics([comment], ARTS)
    s_labels, s_scores = score([comment], engine)
    tid = t["topic_id"][0]
    keywords = ", ".join(ARTS.topic_keywords(tid))
    sent = s_labels[0]
    score_name = "confidence" if engine == "finbert" else "polarity"

    md = [
        f"### Theme: {t['topic_label'][0]}",
        f"**Topic id:** {tid} &nbsp;·&nbsp; **Keywords:** {keywords}",
        f"**Assignment:** {t['topic_assignment'][0]} &nbsp;·&nbsp; "
        f"**Similarity to topic centroid:** {t['topic_similarity'][0]:.2f}"
        + (" ⚠️ _low similarity — review manually_" if t["topic_similarity"][0] < 0.15 else ""),
        f"### Sentiment: <span style='color:{SENT_COLOR[sent]}'>{sent}</span> "
        f"({score_name} {s_scores[0]:+.2f})",
    ]

    if grade_before and grade_after:
        b, a = int(grade_before), int(grade_after)
        chg = a - b
        mig = "Downgrade" if chg > 0 else ("Upgrade" if chg < 0 else "No Change")
        md.append(f"### PD migration: {b} → {a} ({chg:+d} notches, {mig})")
        md.append("_Convention: higher grade = higher risk, so +notches = downgrade._")

    md.append("---")
    md.append(_topic_history_md(tid))
    return "\n\n".join(md), sent


def score_batch(file, engine):
    if file is None:
        return None, None
    df = pd.read_csv(file.name)
    missing = {"PD_ID", "warning_signal_comment"} - set(df.columns)
    if missing:
        raise gr.Error(f"CSV is missing required columns: {missing}")
    docs = df["warning_signal_comment"].astype(str).tolist()

    t = assign_topics(docs, ARTS)
    for col, vals in t.items():
        df[col] = vals
    s_labels, s_scores = score(docs, engine)
    df["sentiment"] = s_labels
    df["sentiment_score"] = s_scores
    if {"pd_grade_before", "pd_grade_after"}.issubset(df.columns):
        df["grade_change"] = df["pd_grade_after"] - df["pd_grade_before"]
        df["migration"] = np.select(
            [df["grade_change"] > 0, df["grade_change"] < 0],
            ["Downgrade", "Upgrade"], default="No Change")

    out_path = "results/app_batch_predictions.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_path, index=False)
    return df, out_path


with gr.Blocks(title="Warning-Signal Comment Scoring") as demo:
    gr.Markdown(
        "# Warning-Signal Comment Scoring\n"
        f"Frozen BERTopic model (`{ARTS.backend}` embeddings, "
        f"{len(ARTS.labels_map)} topics) — no refitting at inference time.")

    engine = gr.Radio(["finbert", "lexicon"], value=DEFAULT_ENGINE,
                      label="Sentiment engine",
                      info="finbert needs model weights; lexicon is fully offline")

    with gr.Tab("Single comment"):
        comment = gr.Textbox(lines=5, label="Underwriter warning-signal comment",
                             placeholder="e.g. Borrower breached the leverage "
                                         "covenant and requested a waiver...")
        with gr.Row():
            g_before = gr.Dropdown([""] + GRADES, value="", label="PD grade before (optional)")
            g_after = gr.Dropdown([""] + GRADES, value="", label="PD grade after (optional)")
        btn = gr.Button("Submit", variant="primary")
        result_md = gr.Markdown()
        sent_state = gr.Textbox(visible=False)
        btn.click(score_single, [comment, g_before, g_after, engine],
                  [result_md, sent_state])

        gr.Examples(
            examples=[
                ["Revolver is 95% drawn and cash fell below $5M; liquidity runway under two quarters.", "7", "10"],
                ["Sponsor injected $50M of equity curing the breach; leverage back under 3x.", "9", "7"],
                ["Audited financials 120 days late; auditor flagged going-concern uncertainty.", "8", ""],
            ],
            inputs=[comment, g_before, g_after])

    with gr.Tab("Batch CSV"):
        gr.Markdown("Upload a CSV with `PD_ID, warning_signal_comment` "
                    "(optional: `pd_grade_before, pd_grade_after`).")
        f_in = gr.File(file_types=[".csv"], label="Input CSV")
        b_btn = gr.Button("Score file", variant="primary")
        table = gr.Dataframe(label="Predictions", interactive=False)
        f_out = gr.File(label="Download scored CSV")
        b_btn.click(score_batch, [f_in, engine], [table, f_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
