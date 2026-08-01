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


def assign_topics(docs, arts):
    """Frozen-model topic assignment with nearest-centroid fallback.

    Returns dict of lists: topic_id, topic_label, topic_assignment
    (direct | nearest_centroid), topic_similarity (cosine to assigned
    topic's centroid, 0..1-ish).
    """
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

    sim_to_assigned = [
        round(float(sims[i, np.where(arts.centroid_ids == t)[0][0]]), 4)
        for i, t in enumerate(topics)]

    return {
        "topic_id": topics.tolist(),
        "topic_label": [arts.labels_map[t] for t in topics],
        "topic_assignment": assignment.tolist(),
        "topic_similarity": sim_to_assigned,
    }
