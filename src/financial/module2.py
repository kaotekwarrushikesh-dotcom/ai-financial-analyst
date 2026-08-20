"""Adapter for Module 2: the Institutional-Style Valuation Engine.

Two entry points exist upstream and the distinction matters enough to preserve rather than
paper over. `run_pipeline` is the full curated workflow for the 33 Nifty companies, where
hand-picked sector peers make comparables a genuine cross-check on the DCF. It reads CSVs
checked into that repo, so it is unavailable to an installed copy and this adapter says so
plainly instead of failing obscurely. `run_quick_pipeline` fetches any listed company live
and returns a DCF, scenarios and Monte Carlo, with no comparables.

The engine's own Calibration finding travels with every valuation this adapter returns. The
DCF reads below market across that universe, comparables do not, and the engine's rule is
that a DCF is never quoted on its own. An AI analyst is exactly the consumer most likely to
quote one number in isolation, so the warning is attached to the result rather than left in
a README the model will never read.
"""

from typing import Any

from src.financial.provenance import DataStatus, Provenance, ToolResult

CALIBRATION_WARNING = (
    "This engine's DCF reads systematically below market across its tested universe, a "
    "documented limitation of its Gordon-growth terminal value rather than a data bug (see "
    "the engine's Calibration section). Never present the DCF fair value on its own: pair it "
    "with the comparable valuation where one exists, and where none does, say so."
)

NO_COMPARABLES = (
    "No comparable-company valuation is available for this company. Real sector peers only "
    "exist for the curated Nifty universe; an auto-discovered peer group would be lower "
    "quality without saying so, which would defeat the purpose of comparables as a "
    "cross-check. Treat the DCF here as one method unsupported by a second."
)


def _run(query: str, horizon: int = 5):
    """Curated workflow where the company is in it, live fetch otherwise."""
    from valuation_engine import pipeline
    from valuation_engine.universe import NIFTY_UNIVERSE

    upper = query.upper().strip()
    candidates = {upper, f"{upper}.NS"}
    for ticker in NIFTY_UNIVERSE:
        if ticker in candidates:
            try:
                return pipeline.run_pipeline(ticker, horizon), "curated"
            except FileNotFoundError:
                # The curated CSVs ship with a checkout, not with the installed package.
                break
    return pipeline.run_quick_pipeline(query, horizon), "quick"


def get_valuation(query: str, horizon: int = 5) -> ToolResult:
    """The headline valuation: DCF, comparables where they exist, and the assumptions behind
    both. This is the tool to call for "what is it worth"."""
    result = ToolResult(tool="get_valuation")
    try:
        r, mode = _run(query, horizon)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_valuation", f"Could not value '{query}': {exc}")

    currency = r["currency"]
    result.ok = True
    result.status = DataStatus.DELAYED
    result.sources = ["Module 2 valuation engine", "Yahoo Finance", "FRED"]
    result.warnings.append(CALIBRATION_WARNING)

    result.add("name", r["name"], Provenance.SOURCE_FACT, "Yahoo Finance")
    result.add("ticker", r["ticker"], Provenance.SOURCE_FACT, "Yahoo Finance")
    result.add("mode", mode, Provenance.MODEL_ASSUMPTION, "Module 2",
               note="curated = full workflow with comparables; quick = live DCF only")
    result.add("share_price", round(float(r["share_price"]), 2), Provenance.SOURCE_FACT,
               "market quote", unit=currency)
    result.add("market_cap", float(r["market_cap"]), Provenance.SOURCE_FACT,
               "market quote", unit=currency)

    dcf = r.get("dcf")
    if dcf is None:
        result.warnings.append(
            f"No DCF was produced: {r.get('dcf_error') or 'the WACC failed validation'}.")
    else:
        result.add("dcf_implied_share_price", round(float(dcf.implied_share_price), 2),
                   Provenance.MODEL_CALCULATION, "Module 2 DCF", unit=currency)
        result.add("dcf_upside_vs_price", round(float(dcf.upside), 4),
                   Provenance.MODEL_CALCULATION, "Module 2 DCF", unit="fraction")
        result.add("terminal_value_share_of_ev", round(float(dcf.terminal_share), 4),
                   Provenance.MODEL_CALCULATION, "Module 2 DCF", unit="fraction",
                   note="how much of the valuation rests on assumptions beyond the forecast")

    wacc = r.get("wacc")
    if wacc is not None:
        result.add("wacc", round(float(wacc.wacc), 5), Provenance.MODEL_CALCULATION,
                   "Module 2 WACC (CAPM)")
        result.add("cost_of_equity", round(float(wacc.cost_of_equity), 5),
                   Provenance.MODEL_CALCULATION, "Module 2 WACC")

    assumptions = r.get("assumptions")
    if assumptions is not None:
        result.add("terminal_growth", round(float(assumptions.terminal_growth), 5),
                   Provenance.MODEL_ASSUMPTION, "Module 2 assumptions")
        result.add("tax_rate", round(float(assumptions.tax_rate), 5),
                   Provenance.MODEL_ASSUMPTION, "Module 2 assumptions")

    beta = r.get("beta")
    if beta is not None:
        result.add("beta_adjusted", round(float(beta.adjusted), 3),
                   Provenance.MODEL_CALCULATION, "Module 2 beta regression")

    comparables = r.get("comparables")
    if comparables is None:
        result.warnings.append(NO_COMPARABLES)
    else:
        implied = getattr(comparables, "blended_value_per_share", None)
        if implied is not None:
            result.add("comparables_implied_share_price", round(float(implied), 2),
                       Provenance.MODEL_CALCULATION, "Module 2 comparables", unit=currency)

    if r.get("implied_wacc") is not None:
        result.add("market_implied_wacc", round(float(r["implied_wacc"]), 5),
                   Provenance.MODEL_CALCULATION, "Module 2 reverse DCF",
                   note="the discount rate at which the model would agree with today's price")

    if not r.get("market_calibrated", True):
        result.warnings.append(
            f"{currency} is not a calibrated market in this engine: the risk-free rate, "
            "equity risk premium and terminal-growth ceiling use a generic developed-market "
            "fallback. Treat this valuation as indicative only."
        )
    result.warnings.extend(r.get("fetch_notes", []))
    result.data = {"currency": currency, "mode": mode, "horizon_years": horizon}
    return result


def get_dcf_valuation(query: str, horizon: int = 5) -> ToolResult:
    """The DCF alone, with its cross-checks. Still carries the calibration warning."""
    result = get_valuation(query, horizon)
    if not result.ok:
        return result
    result.tool = "get_dcf_valuation"
    result.facts = [f for f in result.facts if not f.name.startswith("comparables_")]
    return result


def get_scenario_analysis(query: str, horizon: int = 5) -> ToolResult:
    """Bull, base and bear, each independently recomputed rather than a haircut on the base."""
    result = ToolResult(tool="get_scenario_analysis")
    try:
        r, mode = _run(query, horizon)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_scenario_analysis", f"Could not value '{query}': {exc}")

    scenarios = r.get("scenarios")
    if not scenarios:
        return ToolResult.failure(
            "get_scenario_analysis",
            f"No scenarios available for {r.get('ticker', query)}: the base valuation did not "
            "complete, so there is nothing to flex.")

    currency = r["currency"]
    result.ok = True
    result.status = DataStatus.DELAYED
    result.sources = ["Module 2 scenario engine"]
    result.warnings.append(CALIBRATION_WARNING)
    result.add("share_price", round(float(r["share_price"]), 2), Provenance.SOURCE_FACT,
               "market quote", unit=currency)

    payload: dict[str, Any] = {}
    for scenario in scenarios:
        label = getattr(scenario, "name", None) or getattr(scenario, "label", "scenario")
        price = getattr(scenario, "implied_share_price", None)
        if price is None:
            continue
        result.add(f"{label.lower()}_implied_share_price", round(float(price), 2),
                   Provenance.MODEL_CALCULATION, "Module 2 scenarios", unit=currency)
        payload[label] = {"implied_share_price": round(float(price), 2)}

    result.warnings.append(
        "Each scenario is a full independent revaluation, not a percentage adjustment to the "
        "base case, so a downside compounds margin, growth, discount-rate and terminal effects "
        "together."
    )
    result.data = {"currency": currency, "mode": mode, "scenarios": payload}
    return result


def get_valuation_sensitivity(query: str, horizon: int = 5) -> ToolResult:
    """What the valuation is most sensitive to, which is usually the honest headline."""
    result = ToolResult(tool="get_valuation_sensitivity")
    try:
        r, mode = _run(query, horizon)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_valuation_sensitivity", f"Could not value '{query}': {exc}")

    dcf = r.get("dcf")
    if dcf is None:
        return ToolResult.failure(
            "get_valuation_sensitivity",
            "No DCF was produced, so there is no sensitivity to report.")

    result.ok = True
    result.status = DataStatus.DELAYED
    result.sources = ["Module 2 DCF"]
    result.add("terminal_value_share_of_ev", round(float(dcf.terminal_share), 4),
               Provenance.MODEL_CALCULATION, "Module 2 DCF", unit="fraction",
               note="the single most useful sensitivity read: the share of value resting on "
                    "assumptions beyond the explicit forecast")
    if r.get("implied_wacc") is not None:
        result.add("market_implied_wacc", round(float(r["implied_wacc"]), 5),
                   Provenance.MODEL_CALCULATION, "Module 2 reverse DCF")
    if r.get("wacc") is not None:
        result.add("model_wacc", round(float(r["wacc"].wacc), 5),
                   Provenance.MODEL_CALCULATION, "Module 2 WACC")
    for check in r.get("dcf_checks", []):
        result.warnings.append(str(check))
    result.data = {"mode": mode}
    return result
