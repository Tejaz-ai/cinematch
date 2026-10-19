"""CineMatch - plot-based movie recommender powered live by the TMDB API.

Run locally:  streamlit run app.py

Key handling
  * If TMDB_API_KEY is set on the server (environment variable or Streamlit secrets),
    the app uses it and NEVER shows a key field - visitors don't see or enter anything.
  * If no server key exists (typical local run), a key box appears in the sidebar.
"""
import os

import streamlit as st

from evaluation import lift_over_random
from recommender import PlotRecommender, embeddings_available
from tmdb_client import IMAGE_BASE, TMDBClient, TMDBError

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

MAX_QUERY_CHARS = 100      # protects the shared API key from abuse
MAX_PLOT_CHARS = 3000
PUBLIC_POOL_PAGES = 10     # fixed pool size on the public site (~25 API calls, cached)


# ------------------------------------------------------------- credentials
def server_credential() -> str:
    if os.environ.get("TMDB_API_KEY"):
        return os.environ["TMDB_API_KEY"].strip()
    try:
        return str(st.secrets.get("TMDB_API_KEY", "")).strip()
    except Exception:  # no secrets.toml present
        return ""


SERVER_KEY = server_credential()


# ----------------------------------------------------------------- caching
@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def load_pool(credential: str, pages: int) -> list:
    return TMDBClient(credential).build_base_pool(pages)


@st.cache_data(ttl=3600, max_entries=100, show_spinner=False)
def search(credential: str, query: str) -> list:
    return TMDBClient(credential).search_movie(query)


@st.cache_data(ttl=86400, show_spinner=False)
def genre_names(credential: str) -> dict:
    try:
        return TMDBClient(credential).genre_map()
    except TMDBError:
        return {}


@st.cache_data(ttl=3600, max_entries=100, show_spinner=False)
def tmdb_related(credential: str, movie_id: int) -> list:
    c = TMDBClient(credential)
    return c.similar(movie_id) + c.recommendations(movie_id)


@st.cache_data(ttl=3600, max_entries=100, show_spinner=False)
def same_genre_movies(credential: str, genre_ids: tuple) -> list:
    return TMDBClient(credential).discover(list(genre_ids))


@st.cache_resource(max_entries=15, show_spinner=False)
def get_recommender(credential: str, pages: int, seed_id: int = 0, seed_title: str = "",
                    seed_overview: str = "", seed_genres: tuple = (),
                    backend: str = "tfidf") -> PlotRecommender:
    """Pool = live popular/top-rated list + (optionally) the searched movie, its TMDB
    neighbours and well-known movies from the same genres, so even niche or
    brand-new titles have relevant candidates."""
    movies = list(load_pool(credential, pages))
    if seed_id:
        movies.append({"id": seed_id, "title": seed_title, "overview": seed_overview,
                       "release_date": "", "vote_average": 0.0, "poster_path": None,
                       "genre_ids": list(seed_genres)})
        movies += tmdb_related(credential, seed_id)
        movies += same_genre_movies(credential, seed_genres)
    return PlotRecommender(movies, backend=backend)


def build(credential: str, pages: int, backend: str, seed: dict = None) -> PlotRecommender:
    """Build the recommender; fall back to TF-IDF if the embedding model can't load."""
    seed_args = ((seed["id"], seed["title"], seed["overview"], tuple(seed.get("genre_ids", [])))
                 if seed else (0, "", "", ()))
    label = ("Loading language model (first run downloads ~90 MB)…" if backend == "embeddings"
             else "Fetching live data and building the model…")
    try:
        with st.spinner(label):
            return get_recommender(credential, pages, *seed_args, backend=backend)
    except TMDBError:
        raise
    except Exception as exc:  # missing package, model download failed, etc.
        if backend == "tfidf":
            raise
        st.warning(f"Semantic model unavailable ({exc}). Using keyword matching instead.")
        return get_recommender(credential, pages, *seed_args, backend="tfidf")


# ---------------------------------------------------------------- rendering
def year(m: dict) -> str:
    return (m.get("release_date") or "")[:4] or "n/a"


def show_results(results: list, genres: dict) -> None:
    if not results:
        st.warning("No similar movies found. Try a longer plot or a different title.")
        return
    for rec in results:
        c1, c2 = st.columns([1, 5])
        with c1:
            if rec.get("poster_path"):
                st.image(IMAGE_BASE + rec["poster_path"], use_container_width=True)
        with c2:
            st.markdown(f"**{rec['title']}** ({year(rec)})  ·  ⭐ {rec['vote_average']:.1f}")
            label = f"Match score: {rec['score']:.0%}  (plot {rec['text_sim']:.0%}"
            label += f" · genre {rec['genre_sim']:.0%})" if rec["genre_sim"] else ")"
            st.progress(min(max(rec["score"], 0.0), 1.0), text=label)
            tags = [genres[g] for g in rec.get("genre_ids", []) if g in genres]
            if tags:
                st.caption("🎭 " + " · ".join(tags))
            if rec["shared_terms"]:
                st.caption("Matched on: " + ", ".join(rec["shared_terms"]))
            with st.expander("Plot"):
                st.write(rec["overview"])


def show_evaluation(credential: str, seed: dict, rec: PlotRecommender, results: list) -> None:
    theirs = {m["id"] for m in tmdb_related(credential, seed["id"])}
    ev = lift_over_random(results, theirs, rec.index.keys(), seed["id"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Agree with TMDB", f"{ev['hits']} of {ev['n']}")
    c2.metric("Expected by chance", f"{ev['expected_by_chance']:.1f}")
    c3.metric("Lift over random", f"{ev['lift']:.0f}×" if ev["lift"] else "n/a")
    st.caption("Our candidate pool includes TMDB's own related movies, so raw agreement is "
               "partly built in. Comparing against random picks from the same pool removes that "
               "bias. TMDB's list is itself loose, so treat this as a sanity check, not accuracy.")
    if ev["hit_titles"]:
        st.caption("Agreement: " + ", ".join(ev["hit_titles"]))


# ---------------------------------------------------------------------- UI
st.title("🎬 CineMatch")
st.caption("Find movies by what they're about, not by who rated them. "
           "Live plot data from TMDB → NLTK preprocessing → text similarity → ranked matches.")

with st.sidebar:
    st.header("Settings")
    if SERVER_KEY:                      # public deployment: key stays on the server
        credential, pages = SERVER_KEY, PUBLIC_POOL_PAGES
    else:                               # local run: ask for a key
        credential = st.text_input("TMDB API key / token", type="password",
                                   help="Free at themoviedb.org → Settings → API")
        pages = st.slider("Comparison pool size (pages of 20 movies)", 3, 25, 10)
    top_n = st.slider("Recommendations to show", 3, 15, 8)
    if embeddings_available():
        method = st.radio("Matching method", ["Semantic (sentence embeddings)", "Keyword (TF-IDF)"],
                          help="Semantic understands meaning; keyword only matches shared words.")
    else:
        method = "Keyword (TF-IDF)"
        if not SERVER_KEY:
            st.caption("Semantic matching is off. Enable it with: "
                       "`pip install -r requirements-embeddings.txt`")
    backend = "embeddings" if method.startswith("Semantic") else "tfidf"
    genre_w = st.slider("Genre weight", 0.0, 0.6, 0.3, 0.05,
                        help="0 = plot only. Higher = boost movies from the same genres "
                             "(never the main signal).")

if not credential:
    st.info("Enter your free TMDB API key in the sidebar to start. "
            "(Get one at themoviedb.org → Settings → API.)")
    st.stop()

tab_title, tab_plot = st.tabs(["🔎 Search by title", "📝 Paste a plot"])

try:
    with tab_title:
        query = st.text_input("Enter a movie you like", placeholder="e.g. Inception",
                              max_chars=MAX_QUERY_CHARS)
        if query.strip():
            matches = search(credential, query.strip())
            if not matches:
                st.warning("No movie with a plot summary found for that title.")
            else:
                labels = [f"{m['title']} ({year(m)})" for m in matches[:8]]
                choice = st.selectbox("Did you mean:", range(len(labels)),
                                      format_func=lambda i: labels[i])
                seed = matches[choice]
                genres = genre_names(credential)
                rec = build(credential, pages, backend, seed)
                results = rec.recommend_by_id(seed["id"], top_n, genre_w)
                st.subheader(f"Because you liked {seed['title']}")
                seed_tags = [genres[g] for g in seed.get("genre_ids", []) if g in genres]
                if seed_tags:
                    st.caption("🎭 " + " · ".join(seed_tags))
                show_results(results, genres)
                with st.expander("📊 Evaluation: how do we compare with TMDB's own model?"):
                    show_evaluation(credential, seed, rec, results)

    with tab_plot:
        st.write("Got a brand-new movie that isn't indexed yet? Paste its plot and get matches.")
        plot = st.text_area("Plot / synopsis", height=160, max_chars=MAX_PLOT_CHARS)
        if st.button("Recommend from this plot") and plot.strip():
            results = build(credential, pages, backend).recommend_by_text(plot, top_n)
            show_results(results, genre_names(credential))

except TMDBError as err:
    st.error(str(err))

with st.expander("ℹ️ How CineMatch works"):
    st.markdown(
        "1. **Fetch** - live movie plots from the TMDB API (no static dataset)\n"
        "2. **Preprocess** - lowercase, tokenize, remove stopwords, lemmatize (NLTK)\n"
        "3. **Vectorize** - TF-IDF keyword vectors, or sentence embeddings when available\n"
        "4. **Compare** - cosine similarity between your movie's plot and every candidate\n"
        "5. **Recommend** - top matches, lightly boosted when genres overlap\n\n"
        "No ratings are used, so brand-new releases work immediately."
    )

st.divider()
st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")
