"""Thin client for the TMDB API (Step 1 of the pipeline: Fetch).

Supports both credential types TMDB issues:
  * v3 API key           (32 hex chars)  -> sent as ?api_key=...
  * v4 read access token (long JWT)      -> sent as Authorization: Bearer ...

Free tier limit is ~40 requests / 10 s, so every call is lightly throttled
and 429 responses are retried after the server-provided delay.
"""
import threading
import time
from typing import Dict, List, Optional

import requests

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w185"


class TMDBError(Exception):
    pass


class TMDBClient:
    # Throttle state is shared by ALL instances and threads, so concurrent visitors
    # on a public site still respect TMDB's ~40 requests / 10 s limit together.
    _lock = threading.Lock()
    _last_call = 0.0

    def __init__(self, credential: str, min_interval: float = 0.15, timeout: int = 10):
        credential = (credential or "").strip()
        if not credential:
            raise TMDBError("No TMDB API key provided.")
        self._secret = credential
        self.session = requests.Session()
        self.timeout = timeout
        self.min_interval = min_interval
        self._params: Dict[str, str] = {"language": "en-US"}
        if len(credential) > 40:  # v4 read-access token (JWT)
            self.session.headers["Authorization"] = f"Bearer {credential}"
        else:  # v3 API key
            self._params["api_key"] = credential

    # ---------------------------------------------------------------- core
    def _redact(self, text: str) -> str:
        """Never let the API key appear in an error message shown to a user."""
        return text.replace(self._secret, "***")

    def _get(self, path: str, **params) -> dict:
        for attempt in range(3):
            with TMDBClient._lock:
                wait = self.min_interval - (time.time() - TMDBClient._last_call)
                if wait > 0:
                    time.sleep(wait)
                TMDBClient._last_call = time.time()
            try:
                resp = self.session.get(
                    f"{BASE_URL}{path}",
                    params={**self._params, **params},
                    timeout=self.timeout,
                )
            except requests.RequestException:
                # requests puts the full URL (including ?api_key=...) in its error text,
                # so we deliberately drop the original message.
                if attempt == 2:
                    raise TMDBError("Could not reach TMDB (network error). Please try again.") from None
                time.sleep(1)
                continue
            if resp.status_code == 429:  # rate limited
                time.sleep(float(resp.headers.get("Retry-After", 2)))
                continue
            if resp.status_code == 401:
                raise TMDBError("TMDB rejected the API key (401). Check that it is correct.")
            if resp.status_code != 200:
                raise TMDBError(f"TMDB error {resp.status_code}: {self._redact(resp.text[:150])}")
            return resp.json()
        raise TMDBError("TMDB rate limit hit repeatedly; try again in a few seconds.")

    @staticmethod
    def _clean(movie: dict) -> Optional[dict]:
        """Keep only the fields we use; drop movies with no plot text."""
        overview = (movie.get("overview") or "").strip()
        if not overview or not movie.get("title"):
            return None
        return {
            "id": movie["id"],
            "title": movie["title"],
            "overview": overview,
            "release_date": movie.get("release_date") or "",
            "vote_average": movie.get("vote_average") or 0.0,
            "poster_path": movie.get("poster_path"),
            "genre_ids": movie.get("genre_ids") or [g["id"] for g in movie.get("genres", [])],
        }

    def _list(self, path: str, pages: int, **params) -> List[dict]:
        out = []
        for page in range(1, pages + 1):
            data = self._get(path, page=page, **params)
            out += [m for m in map(self._clean, data.get("results", [])) if m]
            if page >= data.get("total_pages", 1):
                break
        return out

    # ------------------------------------------------------------ endpoints
    def search_movie(self, query: str) -> List[dict]:
        data = self._get("/search/movie", query=query, include_adult="false")
        return [m for m in map(self._clean, data.get("results", [])) if m]

    def similar(self, movie_id: int) -> List[dict]:
        """TMDB's own 'similar' list - used as an evaluation baseline."""
        return self._list(f"/movie/{movie_id}/similar", pages=1)

    def recommendations(self, movie_id: int) -> List[dict]:
        return self._list(f"/movie/{movie_id}/recommendations", pages=1)

    def keywords(self, movie_id: int) -> List[str]:
        """TMDB's thematic tags for a movie (e.g. 'dream', 'heist'); empty list if none."""
        data = self._get(f"/movie/{movie_id}/keywords")
        return [k["name"] for k in data.get("keywords", []) if k.get("name")]

    def genre_map(self) -> Dict[int, str]:
        data = self._get("/genre/movie/list")
        return {g["id"]: g["name"] for g in data.get("genres", [])}

    def discover(self, genre_ids: List[int], pages: int = 2) -> List[dict]:
        """Well-known movies sharing the seed's genres (extra, relevant candidates)."""
        if not genre_ids:
            return []
        common = {"sort_by": "vote_count.desc", "vote_count.gte": 100}
        out = {}
        # AND of the two main genres (closest), then OR across all genres (broader)
        for with_genres, n in ((",".join(map(str, genre_ids[:2])), pages + 1),
                               ("|".join(map(str, genre_ids)), pages)):
            for m in self._list("/discover/movie", pages=n, with_genres=with_genres, **common):
                out[m["id"]] = m
        return list(out.values())

    # ----------------------------------------------------------------- pool
    def build_base_pool(self, pages: int = 10) -> List[dict]:
        """Live comparison pool: popular + top-rated + now-playing movies."""
        pool = {}
        for endpoint, n in (("popular", pages), ("top_rated", pages), ("now_playing", max(1, pages // 3))):
            for m in self._list(f"/movie/{endpoint}", pages=n):
                pool[m["id"]] = m
        return list(pool.values())
