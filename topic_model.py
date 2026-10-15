"""
topic_model.py
--------------
Stage 2: BERTopic topic modeling on warning-signal comments.

Input : data/warning_signals.csv (--data to point at a different CSV, e.g.
        a train-only split written by evaluate_assignment.py)
Output: results/doc_topics.csv        (PD_ID -> topic assignment + keywords)
        results/topic_summary.csv     (topic id, size, top keywords, auto label)
        results/embeddings.npy        (cached sentence embeddings)
        (--results-dir / --models-dir redirect all of the above, e.g. to
        results_eval/ / models_eval/ for a held-out evaluation model that
        doesn't touch the production artifacts)

Pipeline: embeddings -> UMAP -> HDBSCAN -> c-TF-IDF keywords

Embedding backends (--embedding flag):
    minilm  SentenceTransformer all-MiniLM-L6-v2 (recommended; needs model
            weights from HuggingFace or your internal model registry)
    tfidf   TF-IDF + TruncatedSVD (fully offline fallback; no downloads)
"""
import argparse
import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--embedding", choices=["minilm", "tfidf"], default="minilm")
parser.add_argument("--llm", action="store_true",
                    help="Generate topic labels with a local Ollama LLM "
                         "(BERTopic+LLM hybrid) instead of raw c-TF-IDF keywords. "
                         "Requires `ollama serve` running with the model pulled.")
parser.add_argument("--llm-model", default=None,
                    help="Ollama model tag (default: $OLLAMA_MODEL or llama3.2:latest)")
parser.add_argument("--data", default="data/warning_signals.csv",
                    help="Training CSV (default: data/warning_signals.csv). Used by "
                         "evaluate_assignment.py to fit a held-out-split eval model "
                         "without touching the production training data.")
parser.add_argument("--models-dir", default="models",
                    help="Where to write frozen scoring artifacts (default: models/)")
parser.add_argument("--results-dir", default="results",
                    help="Where to write topic_summary.csv / doc_topics.csv / "
                         "embeddings cache (default: results/)")
args = parser.parse_args()

llm_rep = None
if args.llm:
    from ollama_llm import OllamaRepresentation, is_available, OLLAMA_MODEL
    llm_model_name = args.llm_model or OLLAMA_MODEL
    if is_available():
        llm_rep = OllamaRepresentation(model=llm_model_name)
    else:
        print(f"--llm requested but Ollama is not reachable at {os.environ.get('OLLAMA_HOST', 'http://localhost:11434')} "
              "-- falling back to keyword-only labels.")

df = pd.read_csv(args.data)
docs = df["warning_signal_comment"].tolist()
os.makedirs(args.results_dir, exist_ok=True)

# 1. Embeddings (computed once, cached for reuse)
cache = f"{args.results_dir}/embeddings_{args.embedding}.npy"
if os.path.exists(cache) and os.environ.get("REFRESH") != "1":
    print(f"Loading cached embeddings from {cache}")
    embeddings = np.load(cache)
elif args.embedding == "minilm":
    from sentence_transformers import SentenceTransformer
    print("Embedding documents with all-MiniLM-L6-v2...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(docs, show_progress_bar=False, batch_size=64)
    np.save(cache, embeddings)
else:
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
    print("Embedding documents with TF-IDF + SVD (offline fallback)...")
    # alphabetic tokens only: figures like "12m" / "q2 2024" carry no theme signal
    embed_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=3, max_df=0.9, sublinear_tf=True,
                                  token_pattern=r"(?i)\b[a-z]{2,}\b")),
        ("svd", TruncatedSVD(n_components=150, random_state=SEED)),
        ("norm", Normalizer(copy=False)),
    ])
    embeddings = embed_pipe.fit_transform(docs).astype(np.float32)
    np.save(cache, embeddings)
    # persist the fitted embedding pipeline so NEW comments get embedded
    # in the identical vector space at prediction time
    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(embed_pipe, f"{args.models_dir}/embed_pipeline_tfidf.joblib")

# 2. Sub-models (fixed seeds for reproducibility — required for model governance)
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=SEED)
hdbscan_model = HDBSCAN(min_cluster_size=45, min_samples=15,
                        metric="euclidean", cluster_selection_method="eom",
                        prediction_data=True)
# domain stopwords: boilerplate credit words that appear everywhere
DOMAIN_STOP = ["borrower", "quarter", "quarterly", "company", "year", "management",
               "million", "reported", "prior"]
from sklearn.feature_extraction import text
stop_words = list(text.ENGLISH_STOP_WORDS.union(DOMAIN_STOP))
# NOTE: BERTopic fits this vectorizer on CONCATENATED per-topic documents,
# so min_df counts topics, not comments. Keep it low.
vectorizer_model = CountVectorizer(stop_words=stop_words, ngram_range=(1, 2),
                                   min_df=2, token_pattern=r"(?i)\b[a-z]{2,}\b")

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model={"LLM": llm_rep} if llm_rep else None,
    calculate_probabilities=False,
    verbose=True,
)

print("Fitting BERTopic...")
topics, _ = topic_model.fit_transform(docs, embeddings)

# 3. Reassign outliers (-1) to their nearest topic using embeddings
n_outliers = sum(t == -1 for t in topics)
print(f"Outliers before reduction: {n_outliers}")
if n_outliers:
    topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings",
                                         embeddings=embeddings)
    topic_model.update_topics(docs, topics=topics, vectorizer_model=vectorizer_model)

# 4. Human-readable label = top 4 keywords (always available), optionally
#    replaced by the LLM-generated label when --llm is on and it produced one
info = topic_model.get_topic_info()
def keyword_label_for(tid):
    words = [w for w, _ in topic_model.get_topic(tid)][:4]
    return " / ".join(words)

def llm_label_for(tid):
    if not llm_rep:
        return ""
    aspect = topic_model.topic_aspects_.get("LLM", {}).get(tid)
    return aspect[0][0] if aspect and aspect[0][0] else ""

def label_for(tid):
    return llm_label_for(tid) or keyword_label_for(tid)

info["keyword_label"] = info["Topic"].apply(lambda t: keyword_label_for(t) if t != -1 else "outlier")
info["llm_label"] = info["Topic"].apply(lambda t: llm_label_for(t) if t != -1 else "")
info["auto_label"] = info["Topic"].apply(lambda t: label_for(t) if t != -1 else "outlier")
info.rename(columns={"Topic": "topic_id", "Count": "size"})[
    ["topic_id", "size", "auto_label", "keyword_label", "llm_label", "Representation"]
].to_csv(f"{args.results_dir}/topic_summary.csv", index=False)

df["topic_id"] = topics
df["topic_label"] = df["topic_id"].apply(label_for)
df.to_csv(f"{args.results_dir}/doc_topics.csv", index=False)

print(info[["Topic", "Count", "auto_label"]].to_string(index=False))
print(f"\n{len(info) - (1 if -1 in info.Topic.values else 0)} topics found. Saved {args.results_dir}/doc_topics.csv")

# ---------------------------------------------------------------------------
# 5. Persist artifacts for scoring NEW comments (predict.py)
#    - the fitted BERTopic model (pickle: custom UMAP/HDBSCAN sub-models)
#    - per-topic centroids in ORIGINAL embedding space, for outlier fallback
#    - config recording which embedding backend the model was trained with
# ---------------------------------------------------------------------------
import json
os.makedirs(args.models_dir, exist_ok=True)
topic_model.save(f"{args.models_dir}/bertopic_model", serialization="pickle")

tids = sorted(set(topics))
centroids = np.vstack([embeddings[np.array(topics) == t].mean(axis=0) for t in tids])
np.save(f"{args.models_dir}/topic_centroids.npy", centroids)

labels_map = {int(t): label_for(t) for t in tids}
with open(f"{args.models_dir}/config.json", "w") as f:
    json.dump({"embedding_backend": args.embedding,
               "topic_ids": [int(t) for t in tids],
               "topic_labels": labels_map,
               "llm_enabled": bool(llm_rep),
               "llm_model": (args.llm_model or os.environ.get("OLLAMA_MODEL", "llama3.2:latest")) if llm_rep else None,
               "seed": SEED}, f, indent=2)
print(f"Saved scoring artifacts to {args.models_dir}/ (bertopic_model, topic_centroids.npy, config.json)")
