"""Adapter for Module 3: the Live Institutional Risk Engine.

Module 3 is deliberately fine-grained: it exposes returns, volatility, beta, drawdown, three
VaR methods, Expected Shortfall, GARCH and backtesting as separate functions rather than one
"analyse risk" call. That is right for a risk library and wrong for an LLM tool surface,
where each extra tool is another routing decision the model can get wrong. This adapter
composes them into a handful of tools that answer questions an analyst actually asks.

Three of Module 3's findings are carried through as warnings rather than left in its README,
because they change how a number should be read and the model will not know otherwise:

  - Historical VaR cannot exceed the worst loss already observed, so a quiet sample produces
    a reassuring figure for a structural reason rather than an empirical one.
  - VaR at 95% and 99% are wrong in *opposite* directions under a normal assumption, so
    "the normal model understates tail risk" is only half true.
  - Every VaR figure except the GARCH one is unconditional: it describes the whole sample
    rather than today's volatility regime.
"""

from src.data.cache import cached
from src.financial.provenance import DataStatus, Provenance, ToolResult

DEFAULT_PERIOD = "5y"
DEFAULT_BENCHMARK = "^GSPC"

UNCONDITIONAL_WARNING = (
    "This VaR is unconditional: it describes the return distribution across the whole sample "
    "window, not the distribution given today's volatility. Use the GARCH conditional figure "
    "when the question is about risk right now rather than risk in general."
)


@cached("module3.prices")
def _prices(ticker: str, period: str = DEFAULT_PERIOD):
    """Every risk tool reads price history through here, and there are seven of them. This is
    the seam where the "a sweep pays six times" problem was most expensive."""
    from risk_engine.data_loader import clean_prices, fetch_prices
    frame, status = fetch_prices(ticker, period=period)
    return clean_prices(frame), status


@cached("module3.returns")
def _returns(ticker: str, period: str = DEFAULT_PERIOD):
    """Cached separately from `_prices` even though it calls it. The fetch is already saved by
    the cache below it; what this additionally saves is recomputing the log-return series for
    each of the seven risk tools, which is pure arithmetic but not free on ten years of daily
    data."""
    from risk_engine.returns import log_returns
    frame, status = _prices(ticker, period)
    return log_returns(frame["adj_close"]), frame, status


def _status(raw) -> DataStatus:
    label = getattr(raw, "classification", "UNKNOWN")
    return DataStatus.STALE if label == "STALE" else DataStatus.DELAYED


def get_market_data(ticker: str, period: str = DEFAULT_PERIOD) -> ToolResult:
    """Current price and recent history, with an honest freshness classification.

    Module 3 never reports LIVE regardless of how fresh a pull happens to be, because
    yfinance carries no real-time guarantee. That classification is passed straight through.
    """
    result = ToolResult(tool="get_market_data")
    try:
        frame, raw_status = _prices(ticker, period)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_market_data", f"No market data for '{ticker}': {exc}")

    result.ok = True
    result.status = _status(raw_status)
    result.sources = ["Yahoo Finance via Module 3"]
    latest = frame.iloc[-1]
    result.add("last_close", round(float(latest["adj_close"]), 4), Provenance.SOURCE_FACT,
               "Yahoo Finance", as_of=str(frame.index[-1].date()))
    result.add("observations", len(frame), Provenance.SOURCE_FACT, "Yahoo Finance")
    result.add("first_date", str(frame.index[0].date()), Provenance.SOURCE_FACT, "Yahoo Finance")
    result.add("last_date", str(frame.index[-1].date()), Provenance.SOURCE_FACT, "Yahoo Finance")
    result.warnings.append(
        "Data status is reported as DELAYED or STALE, never LIVE: the provider offers no "
        "real-time service guarantee, so a real-time claim would not be supportable."
    )
    result.data = {"period": period}
    return result


def get_volatility(ticker: str, period: str = DEFAULT_PERIOD, window: int = 60) -> ToolResult:
    """Realised volatility at several horizons, plus the current regime."""
    from risk_engine.volatility import historical_volatility, rolling_volatility, volatility_regime

    result = ToolResult(tool="get_volatility")
    try:
        returns, _, raw_status = _returns(ticker, period)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_volatility", f"No data for '{ticker}': {exc}")

    summary = historical_volatility(returns)
    rolling = rolling_volatility(returns, window=window).dropna()
    current = float(rolling.iloc[-1]) if len(rolling) else float("nan")
    average = float(rolling.mean()) if len(rolling) else float("nan")

    result.ok = True
    result.status = _status(raw_status)
    result.sources = ["Module 3 volatility"]
    result.add("annualised_volatility", round(summary.annualised, 5),
               Provenance.MODEL_CALCULATION, "Module 3", unit="fraction")
    result.add("daily_volatility", round(summary.daily, 6), Provenance.MODEL_CALCULATION, "Module 3")
    result.add("current_rolling_volatility", round(current, 5), Provenance.MODEL_CALCULATION,
               "Module 3", unit="fraction", note=f"{window}-day window, annualised")
    result.add("regime", volatility_regime(current, average), Provenance.MODEL_CALCULATION,
               "Module 3", note="measured against this security's own history, not an absolute level")
    result.data = {"window": window, "observations": summary.n_observations}
    return result


def get_beta(ticker: str, benchmark: str = DEFAULT_BENCHMARK,
             period: str = DEFAULT_PERIOD) -> ToolResult:
    """Beta against a benchmark, with the confidence field that says whether to believe it."""
    from risk_engine.beta import compute_beta

    result = ToolResult(tool="get_beta")
    try:
        asset_returns, _, raw_status = _returns(ticker, period)
        benchmark_returns, _, _ = _returns(benchmark, period)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_beta", f"Could not compute beta for '{ticker}': {exc}")

    try:
        beta = compute_beta(asset_returns, benchmark_returns)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_beta", f"Beta estimation failed for '{ticker}': {exc}")

    result.ok = True
    result.status = _status(raw_status)
    result.sources = ["Module 3 beta"]
    # Module 3 computes beta twice (covariance formula and OLS) and asserts them equal at
    # runtime, so either field is the beta; the regression one is used here.
    result.add("beta", round(float(beta.beta_regression), 4), Provenance.MODEL_CALCULATION,
               "Module 3 (covariance and OLS, cross-checked)")
    result.add("r_squared", round(float(beta.r_squared), 4), Provenance.MODEL_CALCULATION, "Module 3")
    result.add("correlation", round(float(beta.correlation), 4), Provenance.MODEL_CALCULATION, "Module 3")
    result.add("confidence", beta.confidence, Provenance.MODEL_CALCULATION, "Module 3")
    result.add("observations", int(beta.n_observations), Provenance.SOURCE_FACT, "Module 3")
    result.add("benchmark", benchmark, Provenance.MODEL_ASSUMPTION, "caller choice")

    if beta.confidence == "low":
        result.warnings.append(
            f"R-squared is {beta.r_squared:.3f}, so this benchmark explains very little of the "
            "stock's movement and the beta is a number that looks meaningful and largely is "
            "not. Report it as low-confidence rather than at face value."
        )
    return result


def get_drawdown(ticker: str, period: str = DEFAULT_PERIOD) -> ToolResult:
    """Current and maximum drawdown, plus the deepest episodes."""
    from risk_engine.drawdown import drawdown_summary, identify_drawdown_episodes

    result = ToolResult(tool="get_drawdown")
    try:
        frame, raw_status = _prices(ticker, period)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_drawdown", f"No data for '{ticker}': {exc}")

    prices = frame["adj_close"]
    summary = drawdown_summary(prices)
    episodes = identify_drawdown_episodes(prices, min_depth=0.10)

    result.ok = True
    result.status = _status(raw_status)
    result.sources = ["Module 3 drawdown"]
    result.add("current_drawdown", round(float(summary.current_drawdown), 5),
               Provenance.MODEL_CALCULATION, "Module 3", unit="fraction")
    result.add("max_drawdown", round(float(summary.max_drawdown), 5),
               Provenance.MODEL_CALCULATION, "Module 3", unit="fraction")
    result.add("max_drawdown_date", str(summary.max_drawdown_date), Provenance.SOURCE_FACT,
               "Module 3")
    result.add("currently_in_drawdown", bool(summary.is_in_drawdown),
               Provenance.MODEL_CALCULATION, "Module 3")
    result.add("episodes_over_10pct", len(episodes), Provenance.MODEL_CALCULATION, "Module 3")
    result.data = {
        "episodes": [
            {"peak": str(e.peak_date), "trough": str(e.trough_date),
             "depth_pct": round(float(e.trough_value_pct), 5),
             "recovered": e.recovery_date is not None}
            for e in episodes[:5]
        ]
    }
    result.warnings.append(
        "Episode durations are in trading days, so they understate elapsed calendar time and "
        "should not be read as days out of the market."
    )
    return result


def get_var(ticker: str, confidence: float = 0.99, horizon_days: int = 1,
            period: str = DEFAULT_PERIOD, portfolio_value: float | None = None) -> ToolResult:
    """Value at Risk by all three methods at once.

    Returned together deliberately. A single VaR number is one method's opinion, and the
    engine's own finding is that the methods disagree in a structured way: the normal model
    is more conservative than the empirical figure at 95% and understates it at 99%. Handing
    the model one number invites it to state that number as the risk.
    """
    from risk_engine.var_historical import historical_var
    from risk_engine.var_monte_carlo import monte_carlo_var
    from risk_engine.var_parametric import fit_distribution, parametric_var

    result = ToolResult(tool="get_var")
    try:
        returns, _, raw_status = _returns(ticker, period)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_var", f"No data for '{ticker}': {exc}")

    try:
        hist = historical_var(returns, confidence, horizon_days, portfolio_value)
        normal = parametric_var(returns, confidence, horizon_days, portfolio_value, "normal")
        student = parametric_var(returns, confidence, horizon_days, portfolio_value, "t")
        simulated = monte_carlo_var(returns, confidence, horizon_days, portfolio_value,
                                    paths=20_000, draw_method="bootstrap")
        fit = fit_distribution(returns)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_var", f"VaR calculation failed for '{ticker}': {exc}")

    result.ok = True
    result.status = _status(raw_status)
    result.sources = ["Module 3 VaR"]
    result.add("confidence_level", confidence, Provenance.MODEL_ASSUMPTION, "caller choice")
    result.add("horizon_days", horizon_days, Provenance.MODEL_ASSUMPTION, "caller choice")
    result.add("var_historical", round(hist.var_return, 5), Provenance.MODEL_CALCULATION,
               "Module 3 historical VaR", unit="fraction loss")
    result.add("var_parametric_normal", round(normal.var_return, 5), Provenance.MODEL_CALCULATION,
               "Module 3 parametric VaR", unit="fraction loss")
    result.add("var_parametric_student_t", round(student.var_return, 5),
               Provenance.MODEL_CALCULATION, "Module 3 parametric VaR", unit="fraction loss")
    result.add("var_monte_carlo_bootstrap", round(simulated.var_return, 5),
               Provenance.MODEL_CALCULATION, "Module 3 Monte Carlo VaR", unit="fraction loss")
    result.add("worst_observed_loss", round(hist.worst_observed_loss, 5),
               Provenance.SOURCE_FACT, "observed history", unit="fraction loss")
    result.add("excess_kurtosis", round(fit.excess_kurtosis, 3), Provenance.MODEL_CALCULATION,
               "Module 3", note="0 for a normal distribution; equity returns are typically 3 to 20")

    if portfolio_value is not None:
        result.add("var_historical_value", round(hist.var_value, 2),
                   Provenance.MODEL_CALCULATION, "Module 3", unit="currency")

    result.warnings.append(UNCONDITIONAL_WARNING)
    result.warnings.extend(hist.warnings)
    if not fit.is_normal:
        result.warnings.append(
            f"Normality is rejected for this series (Jarque-Bera p = {fit.jarque_bera_p:.2g}, "
            f"excess kurtosis {fit.excess_kurtosis:.2f}). The normal figure is not simply "
            "optimistic: it tends to be the more conservative of the methods at 95% and to "
            "understate the tail at 99%, so which way it errs depends on the confidence level."
        )
    if hist.at_the_edge_of_the_data:
        result.warnings.append(
            "The historical VaR sits within 90% of the worst loss in the entire sample. "
            "Historical VaR cannot produce a number worse than something already observed, so "
            "here it is reporting the edge of its data rather than measuring a tail."
        )
    return result


def get_expected_shortfall(ticker: str, confidence: float = 0.99,
                           period: str = DEFAULT_PERIOD) -> ToolResult:
    """Expected Shortfall: the average loss *given* the bad case, which VaR cannot say."""
    from risk_engine.expected_shortfall import (
        historical_expected_shortfall,
        parametric_expected_shortfall,
    )

    result = ToolResult(tool="get_expected_shortfall")
    try:
        returns, _, raw_status = _returns(ticker, period)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_expected_shortfall", f"No data for '{ticker}': {exc}")

    try:
        empirical = historical_expected_shortfall(returns, confidence)
        normal = parametric_expected_shortfall(returns, confidence, distribution="normal")
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_expected_shortfall", f"ES failed for '{ticker}': {exc}")

    result.ok = True
    result.status = _status(raw_status)
    result.sources = ["Module 3 Expected Shortfall"]
    result.add("expected_shortfall_historical", round(empirical.es_return, 5),
               Provenance.MODEL_CALCULATION, "Module 3", unit="fraction loss")
    result.add("expected_shortfall_normal", round(normal.es_return, 5),
               Provenance.MODEL_CALCULATION, "Module 3", unit="fraction loss")
    result.add("var_threshold", round(empirical.var_return, 5), Provenance.MODEL_CALCULATION,
               "Module 3", unit="fraction loss")
    result.add("tail_severity", round(empirical.tail_severity, 3), Provenance.MODEL_CALCULATION,
               "Module 3", unit="ratio",
               note="ES divided by VaR: how much worse the average bad day is than the threshold")
    result.warnings.append(
        "Expected Shortfall is the coherent measure and VaR is not: VaR can report that "
        "diversification increased risk, which is a defect of the measure. This is why Basel "
        "moved market-risk capital from 99% VaR to 97.5% ES."
    )
    result.warnings.extend(empirical.warnings)
    return result


def get_garch_forecast(ticker: str, confidence: float = 0.99, horizon_days: int = 10,
                       period: str = "10y") -> ToolResult:
    """Conditional volatility and conditional VaR: risk given today, not risk on average.

    This is the only risk tool here whose answer changes with the current regime, which makes
    it the right one when the question is about now.
    """
    from risk_engine.garch import conditional_var, fit_garch, forecast_volatility

    result = ToolResult(tool="get_garch_forecast")
    try:
        returns, _, raw_status = _returns(ticker, period)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_garch_forecast", f"No data for '{ticker}': {exc}")

    try:
        fit = fit_garch(returns)
        forecast = forecast_volatility(fit, horizon_days)
        conditional = conditional_var(fit, confidence, horizon_days=1)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure("get_garch_forecast", f"GARCH failed for '{ticker}': {exc}")

    result.ok = True
    result.status = _status(raw_status)
    result.sources = ["Module 3 GARCH(1,1)"]
    result.add("current_annualised_volatility", round(fit.current_annualised_volatility, 5),
               Provenance.MODEL_CALCULATION, "Module 3 GARCH", unit="fraction")
    result.add("realised_annualised_volatility", round(fit.realised_annualised_volatility, 5),
               Provenance.MODEL_CALCULATION, "Module 3", unit="fraction",
               note="the sample benchmark the regime is judged against")
    result.add("regime", conditional["regime"], Provenance.MODEL_CALCULATION, "Module 3 GARCH")
    result.add("conditional_var_1day", round(conditional["var_return"], 5),
               Provenance.MODEL_CALCULATION, "Module 3 GARCH", unit="fraction loss",
               note="VaR given today's volatility, unlike the unconditional figures")
    result.add("persistence", round(fit.persistence, 4), Provenance.MODEL_CALCULATION,
               "Module 3 GARCH", note="alpha + beta: how slowly a volatility shock decays")
    result.add("half_life_days", round(fit.half_life_days, 1), Provenance.MODEL_CALCULATION,
               "Module 3 GARCH", unit="trading days")
    result.add("forecast_volatility_annualised",
               round(float(forecast.annualised_volatility.iloc[-1]), 5),
               Provenance.MODEL_CALCULATION, "Module 3 GARCH", unit="fraction",
               note=f"{horizon_days} trading days ahead")

    result.warnings.extend(fit.warnings)
    if not fit.residuals_are_clean:
        result.warnings.append(
            "ARCH effects remain in the standardised residuals, so this GARCH is misspecified "
            "for this series and its conditional figures should be treated with caution."
        )
    result.data = {"horizon_days": horizon_days, "distribution": fit.distribution}
    return result


def get_risk_metrics(ticker: str, benchmark: str = DEFAULT_BENCHMARK,
                     period: str = DEFAULT_PERIOD) -> ToolResult:
    """The consolidated risk picture, for "what are the risks" rather than one metric.

    Composes volatility, beta, drawdown, VaR, ES and the GARCH conditional read into one
    call. Individual tools remain available for narrower questions; this exists so the common
    case does not require the model to orchestrate six calls correctly.
    """
    result = ToolResult(tool="get_risk_metrics")
    parts = {
        "volatility": get_volatility(ticker, period),
        "beta": get_beta(ticker, benchmark, period),
        "drawdown": get_drawdown(ticker, period),
        "var": get_var(ticker, 0.99, 1, period),
        "expected_shortfall": get_expected_shortfall(ticker, 0.99, period),
        "garch": get_garch_forecast(ticker),
    }

    succeeded = {name: part for name, part in parts.items() if part.ok}
    if not succeeded:
        return ToolResult.failure(
            "get_risk_metrics",
            f"No risk metrics could be computed for '{ticker}': " +
            "; ".join(p.error for p in parts.values() if p.error))

    result.ok = True
    result.status = next(iter(succeeded.values())).status
    result.sources = ["Module 3 risk engine"]
    for name, part in succeeded.items():
        result.facts.extend(part.facts)
        result.warnings.extend(part.warnings)
    for name, part in parts.items():
        if not part.ok:
            result.warnings.append(f"{name} unavailable: {part.error}")
    result.data = {"components": sorted(succeeded), "benchmark": benchmark, "period": period}
    return result
