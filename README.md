# CineMatch — Plot-Based Movie Recommender

Content-based recommender that finds movies by **what they're about** (plot text), using
live data from the TMDB API. No ratings, no static dataset — new releases work immediately.

## Pipeline
| Step | What happens | File |
|------|--------------|------|
| 1. Fetch | Search + build a live pool (popular / top-rated / now-playing) via TMDB | `tmdb_client.py` |
| 2. Preprocess | lowercase → tokenize → stopwords (+ plot-filler words) → lemmatize (NLTK) | `nlp_pipeline.py` |
| 3. Vectorize | **Keyword:** TF-IDF unigrams + bigrams (scikit-learn). **Semantic (optional):** sentence embeddings, `all-MiniLM-L6-v2` | `recommender.py` |
| 4. Compare | Cosine similarity against every plot vector | `recommender.py` |
| 5. Recommend | Top-N ranking; plot similarity boosted by genre overlap (slider); "matched on" keywords in keyword mode | `recommender.py` |
| UI | Streamlit app: search by title, or paste a plot | `app.py` |

## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

1. Create a free account at https://www.themoviedb.org → Settings → API → request a key
   (choose "Developer", personal/educational use).
2. Provide the key — either way works:
   - paste it into the app sidebar, **or**
   - `export TMDB_API_KEY=your_key` (Windows: `set TMDB_API_KEY=your_key`)

## Optional: semantic matching (recommended)
```bash
python -m pip install -r requirements-embeddings.txt
```
Large install (pulls PyTorch, a few hundred MB) and the first search downloads a ~90 MB model.
Once installed, a "Matching method" switch appears in the sidebar. If the model can't load,
the app falls back to keyword matching automatically.

## Deploy publicly (key hidden from visitors)
See **DEPLOY.md** - free hosting on Streamlit Community Cloud, with the TMDB key stored in
server-side Secrets so visitors never see or enter it.

## Run
```bash
streamlit run app.py
```
First run downloads two small NLTK corpora (stopwords, wordnet) automatically.

## Test without an API key
```bash
python test_pipeline.py   # NLP pipeline on hand-written plots
python test_app.py        # headless UI test with a fake TMDB client (incl. public/hidden-key mode)
python test_client.py     # checks the API key can never leak into error messages
```

## Notes for the report
- **Cold-start:** only plot text is used, so a movie released today can be recommended and queried.
- **Evaluation:** the app compares our top-N against TMDB's own `/similar` + `/recommendations`
  lists. Because our pool includes those movies, raw agreement is partly built in, so we also report
  the *lift over random picks from the same pool* — a fairer sanity check (TMDB's list is itself loose).
- **Rate limit:** TMDB allows ~40 requests / 10 s; the client throttles and retries on 429.
  The base pool is cached for 1 hour, so only the first search is slow.
- **Why genres were added:** pure TF-IDF matches exact words, so generic phrases ("last year") caused
  poor matches (an epic action film matched a teen romance). Genre similarity re-ranks candidates,
  and the pool is topped up with well-known movies from the same genres. A movie must still share
  plot vocabulary to be recommended, so genre can never pick an unrelated story by itself.
- **TF-IDF vs embeddings (good report section):** TF-IDF only matches shared words, so "Inception"
  (dreams, subconscious) can't connect to "The Matrix" (simulated reality). Embeddings compare meaning.
  Run the same movie with both methods and compare each against TMDB's own list in the evaluation panel.
- **Lesson from testing:** blending genre as an equal signal backfired - candidates fetched by genre all
  scored 100% on genre, so it stopped discriminating. Genre is now only a boost on plot similarity.
- **Design choices worth explaining:** `min_df=2` (terms in a single plot can't create a match,
  they only dilute scores) and domain stopwords ("must", "find", "young"…).
