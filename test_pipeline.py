"""Offline sanity test: runs the NLP pipeline on hand-written plots (no API needed)."""
from recommender import PlotRecommender

MOVIES = [
    (1, "Dream Heist", "A thief who steals corporate secrets through dream-sharing technology is given the task of planting an idea in a mind."),
    (2, "Island of Memory", "A detective investigates a disappearance at a psychiatric hospital and begins questioning his own memory and reality."),
    (3, "Digital Reality", "A hacker discovers reality is a simulated dream world controlled by machines and joins a rebellion."),
    (4, "Shark Beach", "A giant shark terrorizes a small beach town during the summer and a sheriff must hunt it down."),
    (5, "Love in Paris", "Two strangers fall in love during a romantic week in Paris and must choose between career and each other."),
    (6, "Mind Loop", "A soldier relives the same day repeatedly in a simulation to stop a bomb, unsure what is real."),
    (7, "Ocean Predator", "A great white shark attacks swimmers off a seaside resort and a marine biologist tries to stop it."),
    (8, "Paris Romance", "A young woman moves to Paris, meets a charming painter and falls in love."),
]
GENRES = {1: [1, 2], 2: [2], 3: [1], 4: [3], 5: [4], 6: [1, 2], 7: [3], 8: [4]}
movies = [{"id": i, "title": t, "overview": o, "release_date": "2020-01-01",
           "vote_average": 7.0, "poster_path": None, "genre_ids": GENRES[i]} for i, t, o in MOVIES]

rec = PlotRecommender(movies)

print("== Because you liked 'Dream Heist' ==")
for r in rec.recommend_by_id(1, 3):
    print(f"  {r['title']:<18} {r['score']:.2f}  terms={r['shared_terms']}")
top = [r["title"] for r in rec.recommend_by_id(1, 3)]
assert top[0] in {"Digital Reality", "Mind Loop", "Island of Memory"}, top

print("== Because you liked 'Shark Beach' ==")
top = rec.recommend_by_id(4, 1)
print(f"  {top[0]['title']}  {top[0]['score']:.2f}")
assert top[0]["title"] == "Ocean Predator"

print("== Pasted plot (new release) ==")
top = rec.recommend_by_text("A couple meets in Paris and falls in love over a romantic weekend.", 2)
for r in top:
    print(f"  {r['title']:<18} {r['score']:.2f}")
assert top[0]["title"] in {"Love in Paris", "Paris Romance"}
print("== Genre signal ==")
# Dream Heist (SciFi+Thriller) has plot overlap with Island of Memory (Thriller) and
# Digital Reality (SciFi). Genre must be able to change the ordering.
by_text = [r["title"] for r in rec.recommend_by_id(1, 5, genre_weight=0.0)]
by_both = rec.recommend_by_id(1, 5, genre_weight=0.5)
assert all("genre_sim" in r and "text_sim" in r for r in by_both)
assert by_both[0]["genre_sim"] > 0
# genre alone must never recommend an unrelated story (Love in Paris shares no plot terms)
assert "Love in Paris" not in [r["title"] for r in by_both], by_both
print("  text-only:", by_text)
print("  blended  :", [(r["title"], round(r["genre_sim"], 2)) for r in by_both])

print("== Displayed score matches ranking order ==")
many = rec.recommend_by_id(1, 8, genre_weight=0.5)
sc = [r["score"] for r in many]
assert sc == sorted(sc, reverse=True) and all(0 < s <= 1 for s in sc), sc
assert rec.recommend_by_id(1, 8, genre_weight=0.0)[0]["score"] == rec.recommend_by_id(1, 8, 0.0)[0]["text_sim"]
print("  scores:", [round(s, 2) for s in sc])

import random
rnd = random.Random(0)
words = "dream city war love ship robot island king spy storm forest murder music school desert ghost".split()
synthetic = [{"id": 1000 + i, "title": f"S{i}", "overview": " ".join(rnd.choices(words, k=14)),
              "release_date": "", "vote_average": 0, "poster_path": None,
              "genre_ids": rnd.sample([1, 2, 3, 4, 5], k=rnd.randint(1, 3))} for i in range(60)]
big = PlotRecommender(synthetic)
for w in (0.0, 0.3, 0.6):
    out = big.recommend_by_id(1000, 10, genre_weight=w)
    s = [r["score"] for r in out]
    assert len(out) >= 5 and s == sorted(s, reverse=True) and all(0 < x <= 1 for x in s), (w, s)
print("  60-movie synthetic corpus: order == score order for genre weights 0, 0.3, 0.6")

print("== Lift over random ==")
from evaluation import lift_over_random
res = [{"id": i, "title": f"M{i}"} for i in (1, 2, 3, 4)]
ev = lift_over_random(res, theirs_ids=[2, 3, 50, 51], pool_ids=range(0, 100), seed_id=0)
# pool 99 candidates, 4 are TMDB's -> base rate 4/99; 4 picks -> expected ~0.16; hits 2 -> lift ~12.4
assert ev["hits"] == 2 and abs(ev["expected_by_chance"] - 4 * 4 / 99) < 1e-9 and 12 < ev["lift"] < 13, ev
assert lift_over_random(res, [], range(10), 0)["lift"] is None  # nothing to compare -> n/a, no crash
print("  lift:", round(ev["lift"], 1))

print("== Embeddings backend (stand-in encoder; real model not downloadable here) ==")
import numpy as np, zlib, recommender
from nlp_pipeline import preprocess

calls = []
def fake_encoder(texts):
    """Deterministic bag-of-words hashing -> unit vectors. Stands in for the real model."""
    calls.append(len(texts))
    out = np.zeros((len(texts), 64))
    for r, tx in enumerate(texts):
        for tok in preprocess(tx).split():
            out[r, zlib.crc32(tok.encode()) % 64] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True); n[n == 0] = 1
    return out / n

recommender._EMB_CACHE.clear()
erec = PlotRecommender(movies, backend="embeddings", encoder=fake_encoder)
assert erec.emb.shape == (8, 64) and calls == [8], calls
top = erec.recommend_by_id(4, 1)
assert top[0]["title"] == "Ocean Predator" and top[0]["shared_terms"] == [], top
# a second recommender over the same movies must reuse cached embeddings (no re-encoding)
PlotRecommender(movies + [{"id": 99, "title": "New", "overview": "A shark hunts divers.",
                           "release_date": "", "vote_average": 0, "poster_path": None, "genre_ids": [3]}],
                backend="embeddings", encoder=fake_encoder)
assert calls == [8, 1], f"expected only the new movie to be encoded, got {calls}"
assert erec.recommend_by_text("A shark attacks a beach town", 1)[0]["title"] in {"Shark Beach", "Ocean Predator"}
assert erec.recommend_by_text("   ", 3) == []
print("  caching + ranking OK:", calls)

print("\nAll pipeline checks passed.")
