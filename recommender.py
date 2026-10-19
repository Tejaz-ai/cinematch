"""Steps 3-5 of the pipeline: Vectorize -> Compare -> Recommend.

Two interchangeable text-matching methods:
  * "tfidf"      - keyword matching (TF-IDF + cosine). Fast, explainable baseline.
  * "embeddings" - semantic matching (sentence-transformers + cosine). Understands
                   that "dream-sharing technology" and "simulated reality" are related
                   even when they share no words. Needs `pip install sentence-transformers`.

Genre overlap acts as a *boost* on top of plot similarity (never the main signal):
    score = plot_similarity * (1 + genre_weight * genre_similarity)
"""
import os
from importlib.util import find_spec
from typing import Callable, Dict, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # quiet on Windows

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_pipeline import preprocess

MODEL_NAME = "all-MiniLM-L6-v2"  # ~90 MB, runs fine on CPU
_model = None
_EMB_CACHE: Dict[int, np.ndarray] = {}  # movie id -> embedding, so each plot is encoded once


def embeddings_available() -> bool:
    return find_spec("sentence_transformers") is not None


def default_encoder(texts: List[str]) -> np.ndarray:
    """Encode raw plot text (transformers work best on natural sentences, so no
    stopword removal / lemmatisation here)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return np.asarray(_model.encode(texts, normalize_embeddings=True,
                                    show_progress_bar=False, batch_size=64))


class PlotRecommender:
    def __init__(self, movies: List[dict], backend: str = "tfidf",
                 encoder: Optional[Callable[[List[str]], np.ndarray]] = None):
        if backend not in ("tfidf", "embeddings"):
            raise ValueError("backend must be 'tfidf' or 'embeddings'")
        self.backend = backend
        self.encoder = encoder or default_encoder

        unique: Dict[int, dict] = {}
        for m in movies:
            if m.get("overview"):
                unique[m["id"]] = m
        self.movies = list(unique.values())
        self.index = {m["id"]: i for i, m in enumerate(self.movies)}

        # Step 3: Vectorize (TF-IDF is always built: it powers the "matched on" keywords)
        self.clean_docs = [preprocess(m["overview"]) for m in self.movies]
        # min_df=2 drops terms found in only one plot: they can never create a match.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        self.matrix = self.vectorizer.fit_transform(self.clean_docs)
        self.vocab = np.array(self.vectorizer.get_feature_names_out())

        self.emb = self._embed_movies() if backend == "embeddings" else None

        # Genre vectors (L2-normalised multi-hot)
        all_genres = sorted({g for m in self.movies for g in m.get("genre_ids", [])})
        self.genre_col = {g: i for i, g in enumerate(all_genres)}
        G = np.zeros((len(self.movies), len(all_genres)))
        for r, m in enumerate(self.movies):
            for g in m.get("genre_ids", []):
                G[r, self.genre_col[g]] = 1.0
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.genre_matrix = G / norms

    # ------------------------------------------------------------------
    def _embed_movies(self) -> np.ndarray:
        missing = [m for m in self.movies if m["id"] not in _EMB_CACHE]
        if missing:
            for m, vec in zip(missing, self.encoder([m["overview"] for m in missing])):
                _EMB_CACHE[m["id"]] = vec
        return np.vstack([_EMB_CACHE[m["id"]] for m in self.movies])

    def _genre_sim(self, genre_ids: List[int]) -> np.ndarray:
        q = np.zeros(len(self.genre_col))
        for g in genre_ids:
            if g in self.genre_col:
                q[self.genre_col[g]] = 1.0
        norm = np.linalg.norm(q)
        if norm == 0:
            return np.zeros(len(self.movies))
        return self.genre_matrix @ (q / norm)

    def _text_scores(self, idx: Optional[int], text: Optional[str]):
        """Step 4: Compare - similarity of the query plot to every plot in the pool."""
        if idx is not None:
            tf_q = self.matrix[idx]
            q_emb = self.emb[idx] if self.backend == "embeddings" else None
        else:
            tf_q = self.vectorizer.transform([preprocess(text)])
            q_emb = self.encoder([text])[0] if self.backend == "embeddings" else None
        if self.backend == "embeddings":
            return self.emb @ q_emb, tf_q
        return cosine_similarity(tf_q, self.matrix).ravel(), tf_q

    def _rank(self, idx, text, n, genre_ids=None, genre_weight=0.0) -> List[dict]:
        sim, tf_q = self._text_scores(idx, text)
        use_genre = bool(genre_ids) and genre_weight > 0
        gsim = self._genre_sim(genre_ids) if use_genre else np.zeros(len(sim))
        scores = sim * (1 + genre_weight * gsim) if use_genre else sim.copy()
        # Shown score = ranking score rescaled to 0-1, so the order on screen always
        # matches the number on screen (dividing by a constant never changes the order).
        shown = scores / (1 + genre_weight) if use_genre else scores.copy()
        scores[sim <= 0] = -1.0
        if idx is not None:
            scores[idx] = -1.0
        # Step 5: Recommend - top-N
        results = []
        for i in np.argsort(scores)[::-1][:n]:
            if scores[i] <= 0:
                continue
            rec = dict(self.movies[i])
            rec.update(score=float(shown[i]), text_sim=float(sim[i]), genre_sim=float(gsim[i]),
                       shared_terms=self._shared_terms(tf_q, i) if self.backend == "tfidf" else [])
            results.append(rec)
        return results

    def _shared_terms(self, query_vec, doc_idx: int, k: int = 5) -> List[str]:
        overlap = query_vec.multiply(self.matrix[doc_idx]).toarray().ravel()
        top = np.argsort(overlap)[::-1][:k]
        return [str(self.vocab[j]) for j in top if overlap[j] > 0]

    # ------------------------------------------------------------------
    def recommend_by_id(self, movie_id: int, n: int = 10, genre_weight: float = 0.3) -> List[dict]:
        if movie_id not in self.index:
            raise KeyError(f"Movie {movie_id} is not in the pool.")
        i = self.index[movie_id]
        return self._rank(i, None, n, self.movies[i].get("genre_ids", []), genre_weight)

    def recommend_by_text(self, plot_text: str, n: int = 10) -> List[dict]:
        if not (plot_text or "").strip():
            return []
        if self.backend == "tfidf" and not preprocess(plot_text):
            return []
        return self._rank(None, plot_text, n)
