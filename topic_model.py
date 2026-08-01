"""
topic_model.py
--------------
Stage 2: BERTopic topic modeling on warning-signal comments.

Input : data/warning_signals.csv
Output: results/doc_topics.csv        (PD_ID -> topic assignment + keywords)
        results/topic_summary.csv     (topic id, size, top keywords, auto label)
        results/embeddings.npy        (cached sentence embeddings)

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
args = parser.parse_args()

df = pd.read_csv("data/warning_signals.csv")
docs = df["warning_signal_comment"].tolist()
os.makedirs("results", exist_ok=True)

# 1. Embeddings (computed once, cached for reuse)
cache = f"results/embeddings_{args.embedding}.npy"
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
    os.makedirs("models", exist_ok=True)
    joblib.dump(embed_pipe, "models/embed_pipeline_tfidf.joblib")

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

# 4. Human-readable label = top 4 keywords
info = topic_model.get_topic_info()
def label_for(tid):
    words = [w for w, _ in topic_model.get_topic(tid)][:4]
    return " / ".join(words)
info["auto_label"] = info["Topic"].apply(lambda t: label_for(t) if t != -1 else "outlier")
info.rename(columns={"Topic": "topic_id", "Count": "size"})[
    ["topic_id", "size", "auto_label", "Representation"]
].to_csv("results/topic_summary.csv", index=False)

df["topic_id"] = topics
df["topic_label"] = df["topic_id"].apply(label_for)
df.to_csv("results/doc_topics.csv", index=False)

print(info[["Topic", "Count", "auto_label"]].to_string(index=False))
print(f"\n{len(info) - (1 if -1 in info.Topic.values else 0)} topics found. Saved results/doc_topics.csv")

# ---------------------------------------------------------------------------
# 5. Persist artifacts for scoring NEW comments (predict.py)
#    - the fitted BERTopic model (pickle: custom UMAP/HDBSCAN sub-models)
#    - per-topic centroids in ORIGINAL embedding space, for outlier fallback
#    - config recording which embedding backend the model was trained with
# ---------------------------------------------------------------------------
import json
os.makedirs("models", exist_ok=True)
topic_model.save("models/bertopic_model", serialization="pickle")

tids = sorted(set(topics))
centroids = np.vstack([embeddings[np.array(topics) == t].mean(axis=0) for t in tids])
np.save("models/topic_centroids.npy", centroids)

labels_map = {int(t): label_for(t) for t in tids}
with open("models/config.json", "w") as f:
    json.dump({"embedding_backend": args.embedding,
               "topic_ids": [int(t) for t in tids],
               "topic_labels": labels_map,
               "seed": SEED}, f, indent=2)
print("Saved scoring artifacts to models/ (bertopic_model, topic_centroids.npy, config.json)")
