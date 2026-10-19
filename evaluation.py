"""Sanity-check metric: how much better than random do we agree with TMDB?

Our candidate pool deliberately includes TMDB's own similar/recommended movies, so raw
overlap is partly built in. To remove that bias we compare our hits against what RANDOM
picks from the same pool would score (the "base rate"), and report the lift.
"""
from typing import Iterable, List, Optional


def lift_over_random(results: List[dict], theirs_ids: Iterable[int],
                     pool_ids: Iterable[int], seed_id: int) -> dict:
    candidates = set(pool_ids) - {seed_id}
    theirs = set(theirs_ids) & candidates
    n = len(results)
    hits = [r["title"] for r in results if r["id"] in theirs]
    base_rate = len(theirs) / len(candidates) if candidates else 0.0
    expected = base_rate * n
    lift: Optional[float] = (len(hits) / expected) if expected > 0 else None
    return {"n": n, "hits": len(hits), "hit_titles": hits, "pool_size": len(candidates),
            "theirs_in_pool": len(theirs), "expected_by_chance": expected, "lift": lift}
