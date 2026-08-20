"""Adapter for Module 1: Financial Statement Intelligence.

Calls `fsi` and reshapes what it returns into the provenance-tagged form the tool layer
speaks. It deliberately holds no financial logic of its own: every ratio, trend and score
here is computed by Module 1, and if a number looks wrong the fix belongs in that repo, not
this file. That constraint is what makes the AI analyst a consumer of the engines rather
than a fourth, quietly divergent implementation of them.

The one thing the adapter does add is classification. Module 1 returns filed statement
values and derived scores in the same object; a filed revenue figure is a SOURCE_FACT while
a health score is a MODEL_CALCULATION, and that distinction has to be applied here because
Module 1 has no reason to care about it.
"""

from typing import Any

from src.financial.provenance import DataStatus, Provenance, ToolResult

# Metrics reported as filed values rather than derived ones, so they carry SOURCE_FACT.
FILED_COLUMNS = {
    "revenue", "net_income", "ebitda", "ebit", "cfo", "capex",
    "total_debt", "cash", "equity", "total_assets",
}


def _analysis(query: str):
    from fsi.company import analyse
    return analyse(query)


def get_company_profile(query: str) -> ToolResult:
    """Identity, listing and data provenance for one company."""
    result = ToolResult(tool="get_company_profile")
    try:
        a = _analysis(query)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model, not raised at it
        return ToolResult.failure("get_company_profile", f"Could not resolve '{query}': {exc}")

    result.ok = True
    result.status = DataStatus.DELAYED
    result.sources = [a.source]
    result.add("name", a.name, Provenance.SOURCE_FACT, a.source)
    result.add("ticker", a.ticker, Provenance.SOURCE_FACT, a.source)
    result.add("region", a.region, Provenance.SOURCE_FACT, a.source)
    result.add("currency", a.currency, Provenance.SOURCE_FACT, a.source)
    result.add("years_of_history", a.years, Provenance.SOURCE_FACT, a.source)
    result.add("first_year", a.first_year, Provenance.SOURCE_FACT, a.source)
    result.add("last_year", a.last_year, Provenance.SOURCE_FACT, a.source)
    result.add("share_price", a.share_price, Provenance.SOURCE_FACT, "market quote",
               unit=a.currency)
    result.add("market_cap", a.market_cap, Provenance.SOURCE_FACT, "market quote",
               unit=a.currency)
    result.warnings = list(a.notes)
    result.data = {"statement_source": a.source}
    return result


def get_financial_statements(query: str, years: int = 10) -> ToolResult:
    """Filed statement lines, most recent `years` fiscal years."""
    result = ToolResult(tool="get_financial_statements")
    try:
        a = _analysis(query)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_financial_statements", f"Could not resolve '{query}': {exc}")

    frame = a.ratios.tail(years)
    available = [c for c in FILED_COLUMNS if c in frame.columns]
    if not available:
        return ToolResult.failure(
            "get_financial_statements",
            f"{a.ticker}: no filed statement columns available from {a.source}.")

    result.ok = True
    result.status = DataStatus.DELAYED
    result.sources = [a.source]
    result.data = {
        "currency": a.currency,
        "fiscal_years": [int(y) for y in frame["fiscal_year"]],
        "statements": {c: [None if _isnan(v) else float(v) for v in frame[c]] for c in available},
    }
    latest = frame.iloc[-1]
    for column in sorted(available):
        value = latest[column]
        if not _isnan(value):
            result.add(f"latest_{column}", float(value), Provenance.SOURCE_FACT, a.source,
                       unit=a.currency, as_of=f"FY{int(latest['fiscal_year'])}")
    return result


def get_financial_ratios(query: str) -> ToolResult:
    """Module 1's computed ratios. Every value here is derived, never filed."""
    result = ToolResult(tool="get_financial_ratios")
    try:
        a = _analysis(query)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_financial_ratios", f"Could not resolve '{query}': {exc}")

    frame = a.ratios
    latest = frame.iloc[-1]
    derived = [c for c in frame.columns
               if c not in FILED_COLUMNS and c != "fiscal_year" and not _isnan(latest[c])]

    result.ok = True
    result.status = DataStatus.DELAYED
    result.sources = [a.source, "Module 1 ratio calculations"]
    for column in sorted(derived):
        result.add(column, round(float(latest[column]), 6), Provenance.MODEL_CALCULATION,
                   "Module 1 (fsi.ratios)", as_of=f"FY{int(latest['fiscal_year'])}")
    result.data = {
        "currency": a.currency,
        "fiscal_years": [int(y) for y in frame["fiscal_year"]],
        "ratios": {c: [None if _isnan(v) else float(v) for v in frame[c]] for c in derived},
    }
    return result


def get_financial_trends(query: str) -> ToolResult:
    """Module 1's narrated trend readings."""
    result = ToolResult(tool="get_financial_trends")
    try:
        a = _analysis(query)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_financial_trends", f"Could not resolve '{query}': {exc}")

    result.ok = True
    result.status = DataStatus.DELAYED
    result.sources = ["Module 1 (fsi.trends)"]
    result.data = {"trends": list(a.trends), "years_covered": a.years}
    for i, line in enumerate(a.trends, start=1):
        result.add(f"trend_{i}", line, Provenance.MODEL_CALCULATION, "Module 1 (fsi.trends)")
    return result


def get_financial_health(query: str) -> ToolResult:
    """Module 1's Financial Health Score, with its pillars and its exclusions.

    `unavailable_metrics` is passed through deliberately. Module 1 drops metrics it cannot
    compute and renormalises the remaining weights rather than scoring a gap as zero, so a
    score of 72 built from six metrics is a different statement from the same 72 built from
    nine, and the analyst should be able to say which it has.
    """
    result = ToolResult(tool="get_financial_health")
    try:
        a = _analysis(query)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_financial_health", f"Could not resolve '{query}': {exc}")

    health = a.health
    result.ok = True
    result.status = DataStatus.DELAYED
    result.sources = ["Module 1 (fsi.financial_health)"]
    result.add("overall_score", round(float(health["overall"]), 2), Provenance.MODEL_CALCULATION,
               "Module 1 health score", unit="0-100")
    result.add("rating", health["rating"], Provenance.MODEL_CALCULATION, "Module 1 health score")
    for pillar, score in health.get("pillars", {}).items():
        result.add(f"pillar_{pillar}", round(float(score), 2), Provenance.MODEL_CALCULATION,
                   "Module 1 health score", unit="0-100")

    excluded = health.get("unavailable_metrics", [])
    if excluded:
        result.warnings.append(
            f"{len(excluded)} metric(s) unavailable and excluded from the score, with the "
            f"remaining weights renormalised rather than scored as zero: {', '.join(excluded)}."
        )
    result.data = {"health": _plain(health)}
    return result


def _isnan(value: Any) -> bool:
    try:
        return value != value  # NaN is the only value unequal to itself
    except Exception:  # noqa: BLE001
        return False


def _plain(obj: Any) -> Any:
    """Coerce numpy scalars so the payload is JSON-serialisable for the model."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj
