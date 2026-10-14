# Warning-Signal Comment Analytics (BERTopic + FinBERT)

Analyzes unstructured underwriter warning-signal comments: discovers themes with
BERTopic, scores sentiment with FinBERT, and ties both back to PD grade
migration (1 = best, 14 = worst; an increase in grade number is a downgrade).
Optionally, a local LLM (Ollama) can be layered on top to generate better
topic labels and resolve ambiguous topic assignments — see
[BERTopic + LLM hybrid](#bertopic--llm-hybrid-optional) below.

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
  ollama_llm.py        Optional BERTopic+LLM hybrid (local Ollama) --
                       topic labeling + ambiguous-assignment re-check
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
# LLM-assisted re-check of ambiguous assignments (needs Ollama, see below): add --llm
```

Input needs `PD_ID, warning_signal_comment`; if `pd_grade_before/after` are
present, migration fields are derived too. Output adds `topic_id, topic_label,
topic_assignment, topic_similarity, sentiment, sentiment_score`.

How assignment works: `BERTopic.transform()` maps each new comment through the
frozen UMAP + HDBSCAN (`approximate_predict`). Comments landing in no cluster
(-1) are assigned to the nearest topic centroid by cosine similarity instead of
being left unthemed; `topic_assignment` records which path was used (`direct`,
`nearest_centroid`, or — with `--llm` — `llm_assigned`) and `topic_similarity`
gives an auditable confidence — filter low values for analyst review. Topic
IDs stay stable between refits, so scoring runs are comparable over time.

## Interactive scoring app (Gradio)

```bash
python app.py                     # http://127.0.0.1:7860
APP_ENGINE=lexicon python app.py  # default the engine dropdown to offline
APP_LLM=1 python app.py           # LLM-assisted re-check of ambiguous assignments (needs Ollama, see below)
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

## BERTopic + LLM hybrid (optional)

Opt-in use of a local Ollama LLM (https://ollama.com) to improve on the
default c-TF-IDF keywords and cosine-similarity outlier handling. Off by
default everywhere; falls back silently to the existing logic if Ollama
isn't running.

```bash
ollama serve                    # if not already running
ollama pull llama3.2            # or any other local model
```

**Topic creation** — `topic_model.py --llm [--llm-model TAG]` adds an "LLM"
representation aspect: for each topic, the model is prompted with that
topic's keywords + a few representative comments and asked for a short,
human-readable theme name, e.g. `keyword_label` "violation / equity / breach /
operations" becomes `llm_label` "Covenant Breach Indicators". This becomes
`topic_label` everywhere (`results/topic_summary.csv`'s `llm_label` column,
`results/doc_topics.csv`, `models/config.json`). The raw c-TF-IDF keywords
are kept alongside as `keyword_label` and are still what `topic_keywords()` /
the app's "Keywords:" field show — only the headline label changes. Training
on `data/warning_signals.csv` with `--llm` currently produces labels like
"Refinancing Risk Indicators", "Working Capital Squeeze", "Governance Board
Composition" and "Debt Maturity Wall" — see `results/topic_summary.csv` for
the full set.

**Topic assignment** — `predict.py --llm` and, for the Gradio app,
`APP_LLM=1 python app.py` re-check comments that don't land directly in an
HDBSCAN cluster (today resolved by picking the nearest topic centroid by
raw cosine similarity). Instead, the top-3 nearest candidate topics (id,
label, keywords) are shown to the LLM along with the comment, and it picks
the best match. `topic_assignment` records `llm_assigned` when this path
was used (vs. `direct` / `nearest_centroid`).

Both call a local Ollama server over HTTP (`OLLAMA_HOST`, default
`http://localhost:11434`; `OLLAMA_MODEL`, default `llama3.2:latest`) — no
API keys, nothing leaves the machine.

## Key design points

- Embeddings are cached (`results/embeddings_*.npy`) so re-runs and outlier
  reassignment are cheap.
- All stochastic components are seeded (UMAP `random_state=42`) — required for
  reproducibility under model governance.
- HDBSCAN outliers (topic -1) are reassigned to the nearest topic by embedding
  similarity rather than dropped, so every PD_ID keeps a theme.
- `min_df` in the topic vectorizer counts *topics*, not comments (BERTopic fits
  it on concatenated per-topic text) — keep it low (2).
