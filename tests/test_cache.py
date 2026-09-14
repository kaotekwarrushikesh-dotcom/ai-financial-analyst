"""Tests for Phase 2: the fetch cache.

All offline. The cache is deliberately tested against a counter function rather than the real
engines, because what is under test is the caching behaviour itself, and a test that needs the
network to prove a cache hit is testing two things and diagnosing neither.

The most valuable tests here are the negative ones: that a failure is not cached, and that
different data does not collide onto one key. A cache that silently serves the wrong company's
numbers, or that pins a transient network error in place for fifteen minutes, is far worse
than no cache at all.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import cache
from src.data.cache import TTLCache, cached


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test gets its own view. The registry is module-level, so without this a stats
    assertion in one test would be reading another test's counters."""
    saved = dict(cache._REGISTRY)
    cache._REGISTRY.clear()
    yield
    cache._REGISTRY.clear()
    cache._REGISTRY.update(saved)


def counter_fn(ttl_s=900, name="test.fn"):
    """A cached function that records how many times the real body actually ran."""
    calls = {"n": 0}

    @cached(name, ttl_s=ttl_s)
    def fetch(ticker: str, period: str = "5y"):
        calls["n"] += 1
        return f"{ticker}:{period}:{calls['n']}"

    return fetch, calls


# --- the basic bargain ------------------------------------------------------------------

def test_a_repeated_call_does_not_re_run_the_function():
    fetch, calls = counter_fn()
    first = fetch("AAPL")
    second = fetch("AAPL")
    assert calls["n"] == 1
    assert first == second


def test_different_arguments_are_different_entries():
    """The failure this guards against is the worst one a cache can have: serving one
    company's data for another."""
    fetch, calls = counter_fn()
    assert fetch("AAPL") != fetch("MSFT")
    assert calls["n"] == 2


def test_a_different_period_is_a_different_entry():
    fetch, calls = counter_fn()
    fetch("AAPL", period="5y")
    fetch("AAPL", period="10y")
    assert calls["n"] == 2


# --- key normalisation -------------------------------------------------------------------

def test_case_and_whitespace_do_not_split_an_entry():
    fetch, calls = counter_fn()
    fetch("AAPL")
    fetch("aapl")
    fetch("  AAPL  ")
    assert calls["n"] == 1


def test_positional_and_keyword_spellings_share_an_entry():
    fetch, calls = counter_fn()
    fetch("AAPL")
    fetch(ticker="AAPL")
    assert calls["n"] == 1


def test_an_explicitly_passed_default_shares_the_entry_with_an_omitted_one():
    """`_prices(t)` and `_prices(t, period="5y")` are the same request. Different tools spell
    it both ways, so without signature binding this cost a duplicate fetch of identical data."""
    fetch, calls = counter_fn()
    fetch("AAPL")
    fetch("AAPL", period="5y")
    fetch("AAPL", "5y")
    assert calls["n"] == 1


# --- expiry ------------------------------------------------------------------------------

def test_an_expired_entry_is_refetched():
    fetch, calls = counter_fn(ttl_s=0)
    fetch("AAPL")
    time.sleep(0.01)
    fetch("AAPL")
    assert calls["n"] == 2


def test_an_expiry_is_counted_separately_from_a_plain_miss():
    fetch, calls = counter_fn(ttl_s=0)
    fetch("AAPL")
    time.sleep(0.01)
    fetch("AAPL")
    stats = fetch.cache.stats
    assert stats.expirations == 1
    assert stats.misses == 2


# --- failures ------------------------------------------------------------------------------

def test_a_failure_is_not_cached():
    """A cached exception turns a momentary network blip into a full TTL of downtime. The
    engines are cheap to retry and an error is not a value worth keeping."""
    calls = {"n": 0}

    @cached("test.flaky")
    def flaky(ticker: str):
        calls["n"] += 1
        raise RuntimeError("network hiccup")

    with pytest.raises(RuntimeError):
        flaky("AAPL")
    with pytest.raises(RuntimeError):
        flaky("AAPL")
    assert calls["n"] == 2


def test_a_success_after_a_failure_is_cached_normally():
    calls = {"n": 0}

    @cached("test.recovering")
    def recovering(ticker: str):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first attempt fails")
        return "value"

    with pytest.raises(RuntimeError):
        recovering("AAPL")
    assert recovering("AAPL") == "value"
    assert recovering("AAPL") == "value"
    assert calls["n"] == 2


# --- observability --------------------------------------------------------------------------

def test_stats_count_hits_and_misses():
    fetch, _ = counter_fn(name="test.stats")
    fetch("AAPL")
    fetch("AAPL")
    fetch("MSFT")
    reported = cache.stats()["test.stats"]
    assert reported["misses"] == 2
    assert reported["hits"] == 1
    assert reported["calls"] == 3
    assert reported["hit_rate"] == pytest.approx(1 / 3, abs=0.01)


def test_hit_rate_is_zero_rather_than_a_crash_when_nothing_has_been_called():
    TTLCache()
    fetch, _ = counter_fn(name="test.untouched")
    assert cache.stats()["test.untouched"]["hit_rate"] == 0.0


def test_total_stats_add_up_across_seams():
    a, _ = counter_fn(name="test.a")
    b, _ = counter_fn(name="test.b")
    a("AAPL"); a("AAPL")
    b("MSFT"); b("MSFT")
    total = cache.total_stats()
    assert total["hits"] == 2
    assert total["misses"] == 2


def test_clear_all_forces_a_refetch():
    fetch, calls = counter_fn()
    fetch("AAPL")
    cache.clear_all()
    fetch("AAPL")
    assert calls["n"] == 2


def test_the_undecorated_function_stays_reachable():
    """An escape hatch for a caller that genuinely needs a fresh read, without having to
    reach into cache internals to get one."""
    fetch, calls = counter_fn()
    fetch("AAPL")
    fetch.uncached("AAPL")
    assert calls["n"] == 2


# --- concurrency ------------------------------------------------------------------------------

def test_concurrent_readers_all_get_the_same_value():
    """The agent will eventually fan tools out across threads. Entries must not be corrupted
    by that, and every caller must see one consistent answer."""
    fetch, _ = counter_fn(name="test.threads")
    results = []
    lock = threading.Lock()

    def worker():
        value = fetch("AAPL")
        with lock:
            results.append(value)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 12
    assert len(set(results)) == 1


# --- the real seams --------------------------------------------------------------------------

def test_every_fetch_seam_is_actually_wrapped():
    """The point of the whole phase. If an adapter grows a new way to reach the network and
    forgets to cache it, this is the test that should start failing."""
    cache._REGISTRY.clear()
    import importlib

    from src.financial import module1, module2, module3
    for module in (module1, module2, module3):
        importlib.reload(module)

    assert set(cache._REGISTRY) == {
        "module1.analysis", "module2.pipeline", "module3.prices", "module3.returns",
    }
