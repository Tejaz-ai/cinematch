"""Headless UI test: runs app.py with a fake TMDB client (no network/API key needed)."""
from streamlit.testing.v1 import AppTest
import tmdb_client
from test_pipeline import movies  # reuse mock plots

def fake_search(self, q):
    return [m for m in movies if q.lower() in m["title"].lower()] or []
tmdb_client.TMDBClient.search_movie = fake_search
tmdb_client.TMDBClient.build_base_pool = lambda self, pages=10: movies
tmdb_client.TMDBClient.genre_map = lambda self: {1: "Science Fiction", 2: "Thriller", 3: "Horror", 4: "Romance"}
tmdb_client.TMDBClient.discover = lambda self, genre_ids, pages=2: []
tmdb_client.TMDBClient.similar = lambda self, mid: []
tmdb_client.TMDBClient.recommendations = lambda self, mid: []

# 1) no key -> friendly prompt, no crash
at = AppTest.from_file("app.py", default_timeout=30).run()
assert not at.exception, at.exception
assert any("API key" in i.value for i in at.info), "expected key prompt"
print("no-key state OK")

# 2) with key, search by title
at.sidebar.text_input[0].set_value("fake-key-123").run()
at.tabs[0].text_input[0].set_value("Dream").run()
assert not at.exception, at.exception
print("title search OK ->", [m.value for m in at.markdown if "**" in m.value][:3])

# 3) paste-a-plot tab
at.tabs[1].text_area[0].set_value("A hacker finds out reality is a dream simulation.").run()
at.tabs[1].button[0].click().run()
assert not at.exception, at.exception
print("paste-plot OK")
# 4) embeddings branch of the UI (stand-in encoder) + graceful fallback when the model fails
import recommender
from test_pipeline import fake_encoder
recommender.embeddings_available = lambda: True
recommender.default_encoder = fake_encoder
at = AppTest.from_file("app.py", default_timeout=30).run()
at.sidebar.text_input[0].set_value("fake-key-123").run()
assert at.sidebar.radio[0].options[0].startswith("Semantic")
at.tabs[0].text_input[0].set_value("Dream").run()
assert not at.exception, at.exception
print("embeddings UI OK")

def broken(texts): raise OSError("no internet to download model")
recommender.default_encoder = broken
recommender._EMB_CACHE.clear()
import streamlit as st; st.cache_resource.clear()
at = AppTest.from_file("app.py", default_timeout=30).run()
at.sidebar.text_input[0].set_value("fake-key-456").run()
at.tabs[0].text_input[0].set_value("Dream").run()
assert not at.exception, at.exception
assert any("unavailable" in w.value for w in at.warning), "expected fallback warning"
print("fallback to TF-IDF OK")

# 5) PUBLIC MODE: key set on the server -> no key field is ever rendered to the visitor
import os
recommender.embeddings_available = lambda: False
recommender.default_encoder = fake_encoder
tmdb_client.TMDBClient.similar = lambda self, mid: [movies[2]]   # give the metric something to compare
SECRET = "serversecretkey0123456789abcdef"
os.environ["TMDB_API_KEY"] = SECRET
at = AppTest.from_file("app.py", default_timeout=30).run()
assert not at.exception, at.exception
assert len(at.sidebar.text_input) == 0, "key field must not exist in public mode"
assert not any(SECRET in str(e.value) for e in list(at.markdown) + list(at.caption) + list(at.info)), "key leaked on page"
at.tabs[0].text_input[0].set_value("Dream").run()
assert not at.exception, at.exception
assert len(at.metric) == 3, "evaluation metrics should render"
print("public mode OK (no key field); metrics:", [(m.label, m.value) for m in at.metric])
del os.environ["TMDB_API_KEY"]

print("\nApp UI checks passed.")
