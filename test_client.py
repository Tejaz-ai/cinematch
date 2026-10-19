"""Security + rate-limit tests for the TMDB client (no network needed)."""
import requests
from tmdb_client import TMDBClient, TMDBError

KEY = "abcd1234abcd1234abcd1234abcd1234"


class Boom:
    """Stands in for requests.Session: fails the way real network errors do - with the
    full URL (including the api_key) inside the exception message."""
    def get(self, url, params=None, timeout=None):
        raise requests.ConnectionError(f"Max retries exceeded with url: /3/search/movie?api_key={KEY}&query=x")


c = TMDBClient(KEY, min_interval=0)
c.session = Boom()
try:
    c.search_movie("x")
    raise AssertionError("expected TMDBError")
except TMDBError as e:
    assert KEY not in str(e) and KEY not in repr(e.__cause__), "API key leaked in error text!"
    print("network error redacted OK:", e)


class Bad:
    status_code = 500
    text = f'{{"error": "bad key {KEY}"}}'
    headers = {}


class Sess500:
    def get(self, *a, **k):
        return Bad()


c2 = TMDBClient(KEY, min_interval=0)
c2.session = Sess500()
try:
    c2.search_movie("x")
    raise AssertionError("expected TMDBError")
except TMDBError as e:
    assert KEY not in str(e), "API key leaked in HTTP error text!"
    print("HTTP error redacted OK:", e)

# throttle state is shared across instances (important for many simultaneous visitors)
assert TMDBClient._lock is c._lock is c2._lock
print("shared throttle OK")
print("\nClient security checks passed.")
