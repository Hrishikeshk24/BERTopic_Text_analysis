"""
ollama_llm.py
-------------
BERTopic + LLM hybrid helpers backed by a local Ollama server
(https://ollama.com). Two use sites:

    OllamaRepresentation  topic_model.py --llm
        Generates a short human-readable label per topic from its
        c-TF-IDF keywords + representative documents, as a BERTopic
        "aspect" alongside (not replacing) the default keyword
        representation.

    llm_pick_topic        scoring.py (predict.py --llm / app.py APP_LLM=1)
        For comments that don't land directly in a cluster (HDBSCAN
        outliers resolved today by pure nearest-centroid cosine
        similarity), asks the LLM to choose among the top candidate
        topics instead.

Both fail soft: if Ollama isn't running or a call errors/times out,
callers fall back to the existing c-TF-IDF / cosine-similarity logic.
"""
import os
import re
from typing import List, Mapping, Optional, Tuple

import requests
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document
from tqdm import tqdm

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")

LABEL_PROMPT = """You are labeling themes found in bank underwriter commentary \
about loan risk warning signals.

Keywords for this theme: [KEYWORDS]
Example comments from this theme:
[DOCUMENTS]
Respond with ONLY a short theme name (2-5 words, title case, no punctuation, \
no explanation). Theme name:"""

PICK_PROMPT = """You are classifying a bank underwriter's loan risk warning-signal \
comment into one of the following candidate themes:

[CANDIDATES]
Comment: "[COMMENT]"

Respond with ONLY the id number of the single best-matching theme. No words, \
no punctuation, just the number."""


def is_available(timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        return r.ok
    except requests.RequestException:
        return False


def _generate(prompt: str, model: str = OLLAMA_MODEL, timeout: float = 30.0) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.2, "num_predict": 40}},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


class OllamaRepresentation(BaseRepresentation):
    """BERTopic representation model that labels topics via a local Ollama LLM.

    Intended as an aspect (`representation_model={"LLM": OllamaRepresentation()}`)
    so the default c-TF-IDF keyword representation used elsewhere (e.g.
    `scoring.py`'s `topic_keywords`) is left untouched.
    """

    def __init__(self, model: str = OLLAMA_MODEL, nr_docs: int = 4,
                 doc_length: int = 300, timeout: float = 30.0):
        self.model = model
        self.nr_docs = nr_docs
        self.doc_length = doc_length
        self.timeout = timeout

    def extract_topics(self, topic_model, documents, c_tf_idf, topics
                        ) -> Mapping[str, List[Tuple[str, float]]]:
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, None)

        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            if topic == -1:
                updated_topics[topic] = topics[topic]
                continue
            keywords = ", ".join(next(zip(*topics[topic])))
            truncated_docs = [truncate_document(topic_model, self.doc_length, "char", d)
                               for d in (docs or [])]
            doc_block = "\n".join(f"- {d}" for d in truncated_docs)
            prompt = (LABEL_PROMPT.replace("[KEYWORDS]", keywords)
                                  .replace("[DOCUMENTS]", doc_block))
            try:
                label = _generate(prompt, model=self.model, timeout=self.timeout)
                label = label.strip().strip('"').strip("'").splitlines()[0]
            except Exception:
                label = ""

            padded = [(label, 1.0)] if label else []
            padded += topics[topic][: max(0, 10 - len(padded))]
            updated_topics[topic] = padded

        return updated_topics


def llm_pick_topic(doc_text: str, candidates: List[dict], model: str = OLLAMA_MODEL,
                    timeout: float = 20.0) -> Optional[int]:
    """Ask the LLM to choose the best-fit topic among `candidates`.

    candidates: list of {"id": int, "label": str, "keywords": list[str]}.
    Returns the chosen topic id, or None if Ollama is unreachable or the
    response can't be parsed into one of the candidate ids.
    """
    if not candidates:
        return None
    lines = [f"{c['id']}: {c['label']} (keywords: {', '.join(c['keywords'])})"
             for c in candidates]
    prompt = (PICK_PROMPT.replace("[CANDIDATES]", "\n".join(lines))
                         .replace("[COMMENT]", doc_text[:800]))
    try:
        text = _generate(prompt, model=model, timeout=timeout)
    except Exception:
        return None

    m = re.search(r"-?\d+", text)
    if not m:
        return None
    picked = int(m.group())
    valid_ids = {c["id"] for c in candidates}
    return picked if picked in valid_ids else None
