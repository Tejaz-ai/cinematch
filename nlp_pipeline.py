"""Text preprocessing (Step 2 of the pipeline: Preprocess).

lowercase -> tokenize -> drop stopwords / short tokens -> lemmatize
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

_NLTK_RESOURCES = [("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet")]


def ensure_nltk_data() -> None:
    """Download the small NLTK corpora on first run."""
    for path, name in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_data()

_tokenizer = RegexpTokenizer(r"[a-z]+")  # letters only: drops digits & punctuation
_lemmatizer = WordNetLemmatizer()
# Domain-specific stopwords: filler that appears in almost every movie synopsis
# and says nothing about what the story is actually about.
_PLOT_FILLER = {
    "must", "find", "finds", "found", "one", "two", "new", "will", "get", "gets",
    "take", "takes", "make", "makes", "young", "man", "woman", "life", "world",
    "become", "becomes", "back", "after", "film", "movie", "story", "tries", "try",
    "last", "year", "years", "set", "left", "held", "meet", "meets", "soon", "also",
    "day", "time", "old", "first", "still", "around", "along", "begin", "begins",
    "want", "wants", "return", "returns", "help", "helps", "goes", "three", "four",
    "small", "beautiful", "gift", "possesses", "group", "named", "called", "decides",
}
_stop = set(stopwords.words("english")) | _PLOT_FILLER


def preprocess(text: str) -> str:
    """Return the cleaned, lemmatized version of a plot summary."""
    tokens = _tokenizer.tokenize((text or "").lower())
    tokens = [t for t in tokens if t not in _stop and len(t) > 2]
    return " ".join(_lemmatizer.lemmatize(t) for t in tokens)
