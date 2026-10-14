"""
scoring.py
----------
Shared new-data scoring logic, used by predict.py (CLI batch) and app.py
(Gradio UI) so all consumers score identically.

    arts = load_artifacts()
    out  = assign_topics(["comment ..."], arts)   # topic fields per doc
"""
import json
import os
import numpy as np
from bertopic import BERTopic

MODELS_DIR = "models"
# Opt-in LLM-assisted re-check for ambiguous/outlier assignments (see
# ollama_llm.py). Mirrors the APP_ENGINE convention in app.py; predict.py
# also exposes this as --llm. Off by default -- pure cosine-similarity
# nearest-centroid behavior is unchanged unless explicitly enabled.
USE_LLM_DEFAULT = os.environ.get("APP_LLM", "0") == "1"
LLM_TOP_K = 3


class Artifacts:
    def __init__(self, models_dir=MODELS_DIR):
        with open(os.path.join(models_dir, "config.json")) as f:
            cfg = json.load(f)
        self.backend = cfg["embedding_backend"]
        self.labels_map = {int(k): v for k, v in cfg["topic_labels"].items()}
        self.centroid_ids = np.array(cfg["topic_ids"])
        self.centroids = np.load(os.path.join(models_dir, "topic_centroids.npy"))
        self.topic_model = BERTopic.load(os.path.join(models_dir, "bertopic_model"))
        self._embedder = None
        self._models_dir = models_dir

    def embed(self, docs):
        if self.backend == "minilm":
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            return self._embedder.encode(docs, show_progress_bar=False,
                                         batch_size=64)
        if self._embedder is None:
            import joblib
            self._embedder = joblib.load(
                os.path.join(self._models_dir, "embed_pipeline_tfidf.joblib"))
        return self._embedder.transform(docs).astype(np.float32)

    def topic_keywords(self, topic_id, n=8):
        return [w for w, _ in self.topic_model.get_topic(topic_id)[:n] if w]


def load_artifacts(models_dir=MODELS_DIR):
    return Artifacts(models_dir)


def assign_topics(docs, arts, use_llm=None):
    """Frozen-model topic assignment with nearest-centroid fallback.

    use_llm: if True (or None and $APP_LLM=1), outlier docs that fall back
    to nearest-centroid are re-checked against their top-K centroid
    candidates by a local Ollama LLM (see ollama_llm.py) instead of taking
    the raw cosine-similarity argmax. Fails soft to the argmax pick if
    Ollama is unreachable.

    Returns dict of lists: topic_id, topic_label, topic_assignment
    (direct | nearest_centroid | llm_assigned), topic_similarity (cosine to
    assigned topic's centroid, 0..1-ish).
    """
    if use_llm is None:
        use_llm = USE_LLM_DEFAULT

    embeddings = arts.embed(docs)
    topics, _ = arts.topic_model.transform(docs, embeddings)
    topics = np.array(topics)

    norm_c = arts.centroids / np.linalg.norm(arts.centroids, axis=1, keepdims=True)
    norm_e = embeddings / np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    sims = norm_e @ norm_c.T

    assignment = np.where(topics == -1, "nearest_centroid", "direct")
    fallback = topics == -1
    if fallback.any():
        topics[fallback] = arts.centroid_ids[sims[fallback].argmax(axis=1)]

    if use_llm and fallback.any():
        from ollama_llm import is_available, llm_pick_topic
        if is_available():
            for i in np.where(fallback)[0]:
                order = np.argsort(-sims[i])[:LLM_TOP_K]
                candidates = [
                    {"id": int(arts.centroid_ids[j]),
                     "label": arts.labels_map[int(arts.centroid_ids[j])],
                     "keywords": arts.topic_keywords(int(arts.centroid_ids[j]))}
                    for j in order]
                picked = llm_pick_topic(docs[i], candidates)
                if picked is not None:
                    topics[i] = picked
                    assignment[i] = "llm_assigned"

    sim_to_assigned = [
        round(float(sims[i, np.where(arts.centroid_ids == t)[0][0]]), 4)
        for i, t in enumerate(topics)]

    return {
        "topic_id": topics.tolist(),
        "topic_label": [arts.labels_map[t] for t in topics],
        "topic_assignment": assignment.tolist(),
        "topic_similarity": sim_to_assigned,
    }
