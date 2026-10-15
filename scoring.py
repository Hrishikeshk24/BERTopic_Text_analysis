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
# Opt-in LLM-assisted re-check of topic assignment (see ollama_llm.py).
# Mirrors the APP_ENGINE convention in app.py; predict.py also exposes this
# as --llm. Off by default -- pure cosine-similarity / direct-cluster
# behavior is unchanged unless explicitly enabled.
#
# When enabled, EVERY comment is checked by the LLM (not just HDBSCAN
# outliers) -- but its override is only APPLIED when the statistical pick
# was already low-confidence (topic_similarity below LLM_OVERRIDE_SIM_THRESHOLD).
# A confident direct-cluster hit is never second-guessed. This threshold
# exists because evaluate_assignment.py's held-out evaluation showed letting
# the LLM override *every* pick, confident ones included, net HURT accuracy
# (56.5% -> 50.5% on a 400-row held-out set) -- the small local model
# sometimes confuses adjacent categories (e.g. covenant_breach vs.
# liquidity_stress) that the statistical method already had right. Gating
# by confidence keeps the (occasionally real) benefit on genuinely
# ambiguous comments while not touching picks that were already reliable.
USE_LLM_DEFAULT = os.environ.get("APP_LLM", "0") == "1"
LLM_TOP_K = 3
LLM_OVERRIDE_SIM_THRESHOLD = 0.15  # matches app.py's existing "low similarity --
                                    # review manually" cutoff


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
    """Frozen-model topic assignment with nearest-centroid fallback, and an
    optional LLM fine-tuning pass.

    use_llm: if True (or None and $APP_LLM=1):
        EVERY comment is checked by a local Ollama LLM (see ollama_llm.py)
        against its top candidate topics (the statistical pick plus its
        top-K nearest centroids by cosine similarity). The override is only
        APPLIED when the statistical pick's confidence (topic_similarity)
        is below LLM_OVERRIDE_SIM_THRESHOLD -- a confident direct/centroid
        pick is never second-guessed, even though the LLM is still
        consulted on it. Fails soft to the statistical pick (per-document)
        if Ollama is unreachable or the response can't be parsed.
    If False/unset, assignment is purely statistical: HDBSCAN's direct
    cluster, or (for outliers) the nearest topic centroid by cosine
    similarity.

    Returns dict of lists:
        topic_id            final assigned topic (after any LLM pass)
        topic_label
        topic_assignment    direct | nearest_centroid (no LLM pass, or LLM
                             pass but baseline too confident to override) or
                             llm_confirmed | llm_reassigned (LLM reviewed a
                             LOW-confidence pick and agreed with / overrode
                             it) or llm_skipped_confident (LLM was
                             consulted but the statistical pick was already
                             confident, so its answer -- even if different
                             -- was not applied)
        topic_similarity    cosine of the FINAL topic_id to its centroid
        baseline_topic_id   the statistical (pre-LLM) pick, for comparison
                             -- equals topic_id when use_llm is False

    See evaluate_assignment.py for accuracy metrics (vs. ground-truth
    themes) comparing the statistical baseline to the LLM-fine-tuned pass,
    and why this confidence gate exists.
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

    baseline_topics = topics.copy()
    baseline_sim = [float(sims[i, np.where(arts.centroid_ids == t)[0][0]])
                     for i, t in enumerate(baseline_topics)]

    if use_llm:
        from ollama_llm import is_available, llm_pick_topic
        if is_available():
            for i in range(len(docs)):
                order = np.argsort(-sims[i])[:LLM_TOP_K]
                candidate_ids = [int(arts.centroid_ids[j]) for j in order]
                if int(baseline_topics[i]) not in candidate_ids:
                    candidate_ids = [int(baseline_topics[i])] + candidate_ids
                candidates = [
                    {"id": cid, "label": arts.labels_map[cid],
                     "keywords": arts.topic_keywords(cid)}
                    for cid in candidate_ids]
                picked = llm_pick_topic(docs[i], candidates)  # always consulted
                if picked is None:
                    continue
                if baseline_sim[i] >= LLM_OVERRIDE_SIM_THRESHOLD:
                    # statistical pick already confident -- LLM consulted,
                    # but its answer (even if different) is not applied
                    assignment[i] = "llm_skipped_confident"
                    continue
                topics[i] = picked
                assignment[i] = ("llm_confirmed" if picked == baseline_topics[i]
                                 else "llm_reassigned")

    sim_to_assigned = [
        round(float(sims[i, np.where(arts.centroid_ids == t)[0][0]]), 4)
        for i, t in enumerate(topics)]

    return {
        "topic_id": topics.tolist(),
        "topic_label": [arts.labels_map[t] for t in topics],
        "topic_assignment": assignment.tolist(),
        "topic_similarity": sim_to_assigned,
        "baseline_topic_id": baseline_topics.tolist(),
    }
