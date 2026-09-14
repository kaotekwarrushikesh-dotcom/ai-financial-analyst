"""Phase 2: one fetch per company per window, rather than one fetch per tool call.

The problem this exists to solve is stated in the README's limitations: every tool call
refetches, so a broad question touching six tools pays the network six times for the same
company. The agent that Phase 6 will add makes this worse, not better, because answering one
question well means calling several tools about one company in quick succession.

The fix is deliberately placed at the four functions that actually reach the network, rather
than at the tool layer above them. Those four are the only seams where data enters this
module:

    module1._analysis   filings, ratios and the health score
    module2._run        the valuation pipeline
    module3._prices     price history
    module3._returns    price history plus its log returns

Caching there means every tool benefits without any tool knowing about it, and a tool cannot
accidentally bypass the cache by taking a different route to the same data.

ON CHOOSING A TTL. The tempting move is to cache filings for a day, since a 10-K does not
change between quarters. That would be wrong here: Module 1's `Analysis` carries ten years of
filed statements *and* a live share price and market capitalisation in the same object. A
cache window is governed by the freshest field in a payload, never the stalest, so the whole
object inherits the share price's shelf life. 900 seconds is the default, matching the
`st.cache_data(ttl=900)` the deployed Streamlit apps already use, so a number does not have
one staleness in the app and a different one here.

Failures are never cached. A cached exception would turn a momentary network blip into a
fifteen-minute outage, which is precisely the wrong trade for data this cheap to refetch.
"""

from __future__ import annotations

import functools
import inspect
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_TTL_S = 900

_REGISTRY: dict[str, "TTLCache"] = {}


@dataclass
class CacheStats:
    """Enough to answer "is the cache actually doing anything", which is the only reason to
    add one. A cache nobody can measure is a cache nobody can justify keeping."""

    hits: int = 0
    misses: int = 0
    expirations: int = 0

    @property
    def calls(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.calls if self.calls else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "expirations": self.expirations,
                "calls": self.calls, "hit_rate": round(self.hit_rate, 3)}


def _normalise(value: Any) -> Any:
    """Make keys insensitive to the incidental shape of a caller's argument.

    Tickers arrive as 'aapl', 'AAPL' and ' AAPL ' from different call sites, and all three
    should share one cache entry. This only touches the *key*; the original argument is what
    actually gets passed to the wrapped function, so nothing downstream sees a rewritten
    value. A company name like 'Apple' normalises to 'APPLE', which is harmless for the same
    reason: the key never has to round-trip back to anything.
    """
    if isinstance(value, str):
        return value.strip().upper()
    return value


def _make_key(name: str, signature, args: tuple, kwargs: dict) -> tuple:
    """Bind the call against the function's real signature before keying on it.

    Without this, `_prices("AAPL")` and `_prices("AAPL", period="10y")` are different keys even
    when `10y` is the parameter's default, and `f("AAPL")` and `f(query="AAPL")` are different
    keys for an identical call. Both happen here for real: the risk tools variously pass
    `period` explicitly or leave it out, which showed up as two cache misses for one piece of
    data. Binding and applying defaults collapses every spelling of the same call onto one key.
    """
    try:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return (name, tuple(sorted((k, _normalise(v)) for k, v in bound.arguments.items())))
    except TypeError:
        # Arguments that do not match the signature will fail in the function itself, which is
        # where that error belongs. Fall back to a positional key so caching never turns a
        # TypeError into a confusing cache error.
        return (name,
                tuple(_normalise(a) for a in args),
                tuple(sorted((k, _normalise(v)) for k, v in kwargs.items())))


class TTLCache:
    """A small time-expiring cache. Not an LRU: the working set here is the handful of
    companies touched while answering one question, so entries age out on time rather than
    being evicted on size, and bounding it would add a knob with nothing to tune."""

    def __init__(self, ttl_s: int = DEFAULT_TTL_S):
        self.ttl_s = ttl_s
        self._entries: dict[tuple, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get_or_call(self, key: tuple, produce: Callable[[], Any]) -> Any:
        now = time.monotonic()

        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                value, expires_at = entry
                if expires_at > now:
                    self.stats.hits += 1
                    return value
                del self._entries[key]
                self.stats.expirations += 1
            self.stats.misses += 1

        # Produced outside the lock on purpose. Holding it across a multi-second network call
        # would serialise every fetch in the process behind one mutex, including fetches for
        # unrelated companies. The cost is that two threads racing on the same cold key can
        # both fetch; duplicated work in a rare race is the cheaper side of that trade, and
        # the sequential tool calls this module actually makes never hit it.
        value = produce()

        with self._lock:
            self._entries[key] = (value, time.monotonic() + self.ttl_s)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def cached(name: str, ttl_s: int = DEFAULT_TTL_S):
    """Wrap one of the fetch seams. `name` is what shows up in `stats()`."""

    def decorator(fn):
        cache = TTLCache(ttl_s)
        _REGISTRY[name] = cache
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return cache.get_or_call(_make_key(name, signature, args, kwargs),
                                     lambda: fn(*args, **kwargs))

        wrapper.cache = cache
        wrapper.uncached = fn  # so a caller that genuinely needs a fresh read can force one
        return wrapper

    return decorator


def stats() -> dict[str, dict[str, Any]]:
    """Per-seam hit and miss counts, for tests and for anyone asking whether this earns its
    place."""
    return {name: cache.stats.as_dict() for name, cache in sorted(_REGISTRY.items())}


def total_stats() -> dict[str, Any]:
    combined = CacheStats()
    for cache in _REGISTRY.values():
        combined.hits += cache.stats.hits
        combined.misses += cache.stats.misses
        combined.expirations += cache.stats.expirations
    return combined.as_dict()


def clear_all() -> None:
    """Drop every cached entry. Tests need this to stay independent of each other, and a long
    running session needs it to force a genuinely fresh read."""
    for cache in _REGISTRY.values():
        cache.clear()


def reset_stats() -> None:
    for cache in _REGISTRY.values():
        cache.stats = CacheStats()
