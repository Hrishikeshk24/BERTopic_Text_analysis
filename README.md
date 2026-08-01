# Warning-Signal Comment Analytics (BERTopic + FinBERT)

Analyzes unstructured underwriter warning-signal comments: discovers themes with
BERTopic, scores sentiment with FinBERT, and ties both back to PD grade
migration (1 = best, 14 = worst; an increase in grade number is a downgrade).

## Layout

```
pipeline/
  generate_data.py     Stage 1: synthetic dataset (skip for real data)
  topic_model.py       Stage 2: BERTopic (--embedding minilm|tfidf); saves models/
  sentiment.py         Stage 3: sentiment  (--engine finbert|lexicon)
  sentiment_engines.py Shared sentiment functions (training + prediction)
  scoring.py           Shared topic-assignment logic (predict.py + app.py)
  analysis.py          Stage 4: PD migration vs topic/sentiment
  predict.py           Score NEW comments against the frozen models (CLI)
  app.py               Gradio UI for interactive scoring
  run_pipeline.py      Orchestrator (training/batch path)
  data/warning_signals.csv          input (PD_ID key)
  models/                           frozen artifacts for prediction
  results/doc_topics_sentiment.csv  enriched dataset (one row per PD_ID)
  results/predictions.csv           scored new comments
  results/topic_migration_summary.csv, sentiment_migration.csv,
  results/topic_sentiment_matrix.csv, topic_vs_true_theme.csv
  results/charts/*.png
```

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py              # MiniLM + FinBERT (downloads models on first run)
python run_pipeline.py --offline    # no-download fallback (TF-IDF + lexicon)
```

First transformer run downloads ~90 MB (MiniLM) + ~440 MB (FinBERT) from
HuggingFace; point `HF_HOME` at an internal mirror if your network is restricted.

## Predicting on new data

Training (`topic_model.py`) freezes everything prediction needs into `models/`:
the fitted BERTopic model, per-topic centroids, the embedding backend config,
and (tfidf backend) the fitted TF-IDF/SVD pipeline. To score new comments
without refitting:

```bash
python predict.py --input data/new_comments.csv --output results/predictions.csv
# offline environments: add --engine lexicon
```

Input needs `PD_ID, warning_signal_comment`; if `pd_grade_before/after` are
present, migration fields are derived too. Output adds `topic_id, topic_label,
topic_assignment, topic_similarity, sentiment, sentiment_score`.

How assignment works: `BERTopic.transform()` maps each new comment through the
frozen UMAP + HDBSCAN (`approximate_predict`). Comments landing in no cluster
(-1) are assigned to the nearest topic centroid by cosine similarity instead of
being left unthemed; `topic_assignment` records which path was used and
`topic_similarity` gives an auditable confidence — filter low values for
analyst review. Topic IDs stay stable between refits, so scoring runs are
comparable over time.

## Interactive scoring app (Gradio)

```bash
python app.py                     # http://127.0.0.1:7860
APP_ENGINE=lexicon python app.py  # default the engine dropdown to offline
```

Two tabs. "Single comment": paste a comment, optionally pick PD grade
before/after, hit Submit — returns the theme, topic keywords, assignment
method + centroid similarity (low values are flagged for manual review),
sentiment with score, derived migration, and that topic's historical
downgrade rate / average notch change from the training run. "Batch CSV":
upload `PD_ID, warning_signal_comment [, grades]`, get a scored table plus a
downloadable CSV. The app loads `models/` once at startup and scores through
the same `scoring.py` / `sentiment_engines.py` code path as `predict.py`, so
UI and CLI can never disagree. Run `analysis.py` after any retrain so the
historical stats shown in the app match the current model.

## Using real data

Replace `data/warning_signals.csv` with your extract keeping columns
`PD_ID, warning_signal_comment, pd_grade_before, pd_grade_after`, then:

```bash
python run_pipeline.py --skip-datagen
```

## Key design points

- Embeddings are cached (`results/embeddings_*.npy`) so re-runs and outlier
  reassignment are cheap.
- All stochastic components are seeded (UMAP `random_state=42`) — required for
  reproducibility under model governance.
- HDBSCAN outliers (topic -1) are reassigned to the nearest topic by embedding
  similarity rather than dropped, so every PD_ID keeps a theme.
- `min_df` in the topic vectorizer counts *topics*, not comments (BERTopic fits
  it on concatenated per-topic text) — keep it low (2).
