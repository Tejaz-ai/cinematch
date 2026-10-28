"""CineMatch - plot-based movie recommender powered live by the TMDB API.

Run locally:  streamlit run app.py

Key handling
  * If TMDB_API_KEY is set on the server (environment variable or Streamlit secrets),
    the app uses it and NEVER shows a key field - visitors don't see or enter anything.
  * If no server key exists (typical local run), a key box appears in the sidebar.
"""
import os
from html import escape as esc

import streamlit as st

from evaluation import lift_over_random
from explain import build_explanation
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


@st.cache_data(ttl=86400, max_entries=500, show_spinner=False)
def movie_keywords(credential: str, movie_id: int) -> list:
    return TMDBClient(credential).keywords(movie_id)


def safe_keywords(credential: str, movie_id: int) -> list:
    """Themes are a bonus layer: if TMDB fails here, explain with fewer layers
    instead of breaking the results page (and don't cache the failure)."""
    try:
        return movie_keywords(credential, movie_id)
    except TMDBError:
        return []


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


# ------------------------------------------------------------------ styling
STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, .stApp button, .stApp input, .stApp textarea {font-family: Inter, system-ui, sans-serif;}
.block-container {max-width: 1100px; padding-top: 1.2rem; padding-bottom: 3rem;}
footer, #MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.cm-hero {position: relative; overflow: hidden; border-radius: 22px; padding: 34px 38px 38px; margin-bottom: 18px;
  background: linear-gradient(135deg, #0D1829 0%, #1B2E4B 58%, #15493D 100%); color: #fff;}
.cm-hero:before {content: ""; position: absolute; width: 480px; height: 480px; right: -130px; top: -190px;
  background: radial-gradient(circle, rgba(63,191,147,.38), transparent 68%);}
.cm-logo {position: relative; display: flex; align-items: center; gap: 10px; margin-bottom: 30px;
  font-family: Fraunces, Georgia, serif; font-size: 22px; font-weight: 600;}
.cm-mark {width: 34px; height: 34px; border-radius: 10px; background: #3FBF93; display: grid; place-items: center; font-size: 17px;}
.cm-hero h1 {position: relative; font-family: Fraunces, Georgia, serif; font-weight: 600; line-height: 1.1;
  font-size: clamp(30px, 4.6vw, 48px); margin: 0 0 12px; padding: 0; max-width: 680px; color: #fff;}
.cm-hero h1 em {font-style: normal; color: #74E3BC;}
.cm-hero .cm-sub {position: relative; color: #C0CCDD; font-size: 16px; max-width: 560px;}
div[data-testid="stTextInput"] input {height: 54px; font-size: 17px; border-radius: 14px;}
div[data-testid="stTextArea"] textarea {border-radius: 14px; font-size: 15px;}
.stButton > button {border-radius: 999px;}
button[data-baseweb="tab"] {font-size: 15px;}
.cm-head {margin: 22px 0 14px;}
.cm-head h2 {font-family: Fraunces, Georgia, serif; font-size: 26px; font-weight: 600; margin: 0 0 8px; padding: 0;}
.cm-head h2 span {color: #74E3BC;}
.cm-chip {display: inline-block; font-size: 12px; padding: 3px 10px; border-radius: 999px; background: #173B34; color: #74E3BC; margin: 0 6px 6px 0; font-weight: 500;}
.cm-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px;}
.cm-card {position: relative; display: flex; gap: 14px; background: #142034; border: 1px solid #263650; border-radius: 16px; padding: 14px;
  transition: transform .18s, box-shadow .18s, border-color .18s;}
.cm-card:hover {transform: translateY(-4px); box-shadow: 0 12px 30px rgba(0,0,0,.5); border-color: #3FBF93;}
.cm-poster {flex: 0 0 92px; height: 138px; border-radius: 10px; overflow: hidden; display: flex; align-items: flex-end; background: #1B2E4B;}
.cm-poster img {width: 100%; height: 100%; object-fit: cover;}
.cm-ini {display: block; padding: 8px; color: #fff; font-family: Fraunces, Georgia, serif; font-size: 26px; line-height: 1;}
.cm-info {min-width: 0; flex: 1; display: flex; flex-direction: column;}
.cm-t {font-weight: 600; font-size: 16px; margin: 0 56px 2px 0; color: #E8EDF5;}
.cm-m {font-size: 12.5px; color: #93A1B8; margin-bottom: 8px;}
.cm-why {margin-top: auto; font-size: 12.5px; color: #93A1B8;}
.cm-why b {color: #E8EDF5; font-weight: 500;}
.cm-card details {margin-top: 6px; font-size: 12.5px; color: #93A1B8;}
.cm-card summary {cursor: pointer; color: #74E3BC;}
.cm-card details p {margin: 6px 0 0; line-height: 1.5;}
.cm-ring {position: absolute; right: 14px; top: 14px; width: 46px; height: 46px; border-radius: 50%;
  background: conic-gradient(#3FBF93 calc(var(--p) * 1%), #1A2A42 0); display: grid; place-items: center;}
.cm-ring span {width: 36px; height: 36px; border-radius: 50%; background: #142034; display: grid; place-items: center; font-size: 12px; font-weight: 600; color: #74E3BC;}
.cm-eval {margin: 34px 0 10px; background: #142034; border: 1px solid #263650; border-radius: 16px; padding: 18px 20px;}
.cm-eval h3 {margin: 0 0 12px; padding: 0; font-size: 14px; font-weight: 500; color: #93A1B8;}
.cm-tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;}
.cm-tile {background: #1A2A42; border-radius: 12px; padding: 12px 14px;}
.cm-tile small {display: block; color: #93A1B8; font-size: 12px;}
.cm-tile strong {font-size: 24px; font-weight: 600; color: #E8EDF5;}
.cm-tile strong.hi {color: #74E3BC;}
.cm-note {margin-top: 10px; font-size: 12px; color: #93A1B8; line-height: 1.5;}
.cm-steps {display: flex; flex-wrap: wrap; gap: 10px 22px; margin: 22px 0 6px; font-size: 13.5px; color: #93A1B8;}
.cm-steps i {font-style: normal; display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%;
  background: #173B34; color: #74E3BC; font-size: 12px; font-weight: 600; margin-right: 7px;}
.cm-foot {margin-top: 22px; padding-top: 16px; border-top: 1px solid #263650; font-size: 12.5px; color: #93A1B8;}
@media (max-width: 640px) {.cm-hero {padding: 24px 20px 26px;} .cm-grid {grid-template-columns: 1fr;}}
</style>"""

HERO = ('<div class="cm-hero"><div class="cm-logo"><span class="cm-mark">🎬</span>CineMatch</div>'
        '<h1>Find movies by what they’re <em>about</em></h1>'
        '<div class="cm-sub">Not by who rated them. Live plot data from TMDB, so brand-new releases work on day one.</div></div>')

GRADS = ["#12304F,#2F8F6F", "#0F3B2E,#1B2E4B", "#3A2F5B,#1B2E4B",
         "#1B2E4B,#4A6FA5", "#14473B,#0D1829", "#2B3A67,#2F8F6F"]


# ---------------------------------------------------------------- rendering
def year(m: dict) -> str:
    return (m.get("release_date") or "")[:4] or "n/a"


def txt(value) -> str:
    """Escape for HTML and collapse all whitespace. A blank line inside an HTML block would
    make Streamlit's markdown parser end the block and break the card layout."""
    return esc(" ".join(str(value).split()))


def chips(names: list) -> str:
    return "".join(f'<span class="cm-chip">{txt(n)}</span>' for n in names)


def why_html(ex: dict) -> str:
    """The 'Why this movie?' panel. Every piece of text is escaped and whitespace-collapsed."""
    if not ex:
        return ""
    rows = []
    if ex["summary"]:
        rows.append(f"<p>{txt(ex['summary'])}</p>")
    if ex["terms"]:
        rows.append(f"<p><b>Shared plot words:</b> {txt(', '.join(ex['terms']))}</p>")
    if ex["pair"]:
        rows.append(f"<p><b>Closest matching parts</b><br>Your movie: “{txt(ex['pair']['seed'])}”"
                    f"<br>This movie: “{txt(ex['pair']['other'])}”</p>"
                    "<div class='cm-note'>These show where the two plots are most alike, "
                    "not proof of why the model chose this movie.</div>")
    if ex["empty"]:
        rows.append("<p>No strong overlap beyond the overall text similarity.</p>")
    return "<details><summary>Why this movie?</summary>" + "".join(rows) + "</details>"


def card_html(rec: dict, genres: dict, ex: dict = None) -> str:
    """One movie card. Everything that comes from TMDB or the user is HTML-escaped."""
    pct = round(min(max(rec["score"], 0.0), 1.0) * 100)
    if rec.get("poster_path"):
        poster = f'<div class="cm-poster"><img src="{esc(IMAGE_BASE + rec["poster_path"])}" alt="" loading="lazy"></div>'
    else:
        a, b = GRADS[rec["id"] % len(GRADS)].split(",")
        initials = "".join(w[0] for w in rec["title"].split()[:2]).upper() or "?"
        poster = (f'<div class="cm-poster" style="background:linear-gradient(160deg,{a},{b})">'
                  f'<span class="cm-ini">{txt(initials)}</span></div>')
    rating = f"★ {rec['vote_average']:.1f}" if rec["vote_average"] > 0 else "Not yet rated"
    tags = chips([genres[g] for g in rec.get("genre_ids", []) if g in genres][:3])
    why = (f'<div class="cm-why"><b>Matched on:</b> {txt(", ".join(rec["shared_terms"]))}</div>'
           if rec["shared_terms"] and not ex else "")
    tip = f"Plot {rec['text_sim']:.0%}" + (f" · genre {rec['genre_sim']:.0%}" if rec["genre_sim"] else "")
    return ('<div class="cm-card">' + poster +
            '<div class="cm-info">'
            f'<div class="cm-t">{txt(rec["title"])}</div>'
            f'<div class="cm-m">{txt(year(rec))} · {rating}</div>'
            f'<div>{tags}</div>{why}{why_html(ex)}'
            f'<details><summary>Plot</summary><p>{txt(rec["overview"])}</p></details></div>'
            f'<div class="cm-ring" style="--p:{pct}" title="{esc(tip)}"><span>{pct}%</span></div></div>')


def show_results(results: list, genres: dict, heading_html: str, explanations: dict = None) -> None:
    st.markdown(heading_html, unsafe_allow_html=True)
    if not results:
        st.warning("No similar movies found. Try a longer plot or a different title.")
        return
    ex = explanations or {}
    st.markdown('<div class="cm-grid">' + "".join(card_html(r, genres, ex.get(r["id"])) for r in results) + "</div>",
                unsafe_allow_html=True)


def show_evaluation(credential: str, seed: dict, rec: PlotRecommender, results: list) -> None:
    theirs = {m["id"] for m in tmdb_related(credential, seed["id"])}
    ev = lift_over_random(results, theirs, rec.index.keys(), seed["id"])
    lift = f"{ev['lift']:.0f}×" if ev["lift"] else "n/a"
    agree = f"Agreement: {txt(', '.join(ev['hit_titles']))}. " if ev["hit_titles"] else ""
    st.markdown(
        '<div class="cm-eval"><h3>How do we compare with TMDB’s own model?</h3><div class="cm-tiles">'
        f'<div class="cm-tile"><small>Agree with TMDB</small><strong>{ev["hits"]} of {ev["n"]}</strong></div>'
        f'<div class="cm-tile"><small>Expected by chance</small><strong>{ev["expected_by_chance"]:.1f}</strong></div>'
        f'<div class="cm-tile"><small>Lift over random</small><strong class="hi">{lift}</strong></div></div>'
        f'<div class="cm-note">{agree}Our candidate pool includes TMDB’s own related movies, so raw agreement is '
        'partly built in. Comparing against random picks from the same pool removes that bias. '
        'TMDB’s list is itself loose, so treat this as a sanity check, not accuracy.</div></div>',
        unsafe_allow_html=True)


def explain_all(credential: str, seed: dict, results: list, genres: dict, rec: PlotRecommender) -> dict:
    """One 'Why this movie?' explanation per result, keyed by movie id."""
    seed_kw = safe_keywords(credential, seed["id"]) if seed.get("id") else []
    encode = rec.encoder if rec.backend == "embeddings" else None   # sentence pairs need the semantic model
    return {r["id"]: build_explanation(seed, r, genres, seed_kw,
                                       safe_keywords(credential, r["id"]) if seed_kw else [], encode)
            for r in results}


def set_query(name: str) -> None:
    st.session_state["query"] = name


# ---------------------------------------------------------------------- UI
st.markdown(STYLE, unsafe_allow_html=True)

if SERVER_KEY:                          # public deployment: key stays on the server, no sidebar at all
    credential, pages = SERVER_KEY, PUBLIC_POOL_PAGES
else:                                   # local run: ask for a key in the sidebar
    with st.sidebar:
        st.header("Setup")
        credential = st.text_input("TMDB API key / token", type="password",
                                   help="Free at themoviedb.org → Settings → API")
        pages = st.slider("Comparison pool size (pages of 20 movies)", 3, 25, 10)

st.markdown(HERO, unsafe_allow_html=True)

if not credential:
    st.info("Enter your free TMDB API key in the sidebar to start. "
            "(Get one at themoviedb.org → Settings → API.)")
    st.stop()

with st.expander("Options"):
    oc1, oc2, oc3 = st.columns(3)
    top_n = oc1.slider("Recommendations", 3, 15, 8)
    genre_w = oc2.slider("Genre boost", 0.0, 0.6, 0.3, 0.05,
                         help="0 = plot only. Higher = boost movies from the same genres (never the main signal).")
    if embeddings_available():
        method = oc3.radio("Matching method", ["Semantic (sentence embeddings)", "Keyword (TF-IDF)"],
                           help="Semantic understands meaning; keyword only matches shared words.")
    else:
        method = "Keyword (TF-IDF)"
        if not SERVER_KEY:
            oc3.caption("Semantic matching is off. Enable it with: `pip install -r requirements-embeddings.txt`")
backend = "embeddings" if method.startswith("Semantic") else "tfidf"

tab_title, tab_plot = st.tabs(["Search by title", "Paste a plot"])

try:
    with tab_title:
        query = st.text_input("Movie title", key="query", label_visibility="collapsed",
                              placeholder="Type a movie you like, e.g. Inception", max_chars=MAX_QUERY_CHARS)
        tc = st.columns([0.5, 1.4, 1.4, 1.4, 6])
        tc[0].markdown("<div style='padding-top:.5rem;color:#93A1B8;font-size:13px'>Try</div>", unsafe_allow_html=True)
        for col, name in zip(tc[1:4], ("Interstellar", "Parasite", "Dune")):
            col.button(name, key=f"ex_{name}", on_click=set_query, args=(name,))
        if query.strip():
            matches = search(credential, query.strip())
            if not matches:
                st.warning("No movie with a plot summary found for that title.")
            else:
                labels = [f"{m['title']} ({year(m)})" for m in matches[:8]]
                choice = st.selectbox("Did you mean", range(len(labels)), format_func=lambda i: labels[i])
                seed = matches[choice]
                genres = genre_names(credential)
                rec = build(credential, pages, backend, seed)
                results = rec.recommend_by_id(seed["id"], top_n, genre_w)
                seed_tags = chips([genres[g] for g in seed.get("genre_ids", []) if g in genres])
                with st.spinner("Preparing explanations…"):
                    explanations = explain_all(credential, seed, results, genres, rec)
                show_results(results, genres,
                             f'<div class="cm-head"><h2>Because you liked <span>{txt(seed["title"])}</span></h2>{seed_tags}</div>',
                             explanations)
                show_evaluation(credential, seed, rec, results)

    with tab_plot:
        st.write("Got a brand-new movie that isn’t indexed yet? Paste its plot and get matches.")
        plot = st.text_area("Plot / synopsis", height=160, max_chars=MAX_PLOT_CHARS, label_visibility="collapsed",
                            placeholder="Paste the plot of a brand-new movie…")
        if st.button("Match this plot") and plot.strip():
            rec = build(credential, pages, backend)
            results = rec.recommend_by_text(plot, top_n)
            genres = genre_names(credential)
            explanations = explain_all(credential, {"overview": plot, "genre_ids": []}, results, genres, rec)
            show_results(results, genres,
                         '<div class="cm-head"><h2>Closest <span>matches</span> to your plot</h2></div>',
                         explanations)

except TMDBError as err:
    st.error(str(err))

st.markdown(
    '<div class="cm-steps"><span><i>1</i>Fetch live plots</span><span><i>2</i>Preprocess</span>'
    '<span><i>3</i>Vectorize</span><span><i>4</i>Compare</span><span><i>5</i>Recommend</span></div>'
    '<div class="cm-foot">This product uses the TMDB API but is not endorsed or certified by TMDB.</div>',
    unsafe_allow_html=True)
