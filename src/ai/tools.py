"""The unified tool interface: one registry the AI calls, three engines behind it.

This is the seam the whole module is built around. The LLM sees a flat list of financial
tools with JSON schemas and never learns that ratios come from one repository, valuation from
a second and risk from a third. Each tool is a real Python function that ends in a
deterministic calculation, so the model chooses *what* to compute and never computes it.

Two properties are enforced here rather than requested in a prompt, because a prompt is a
request and this is a guarantee:

**A tool never raises at the model.** Every call returns a ToolResult, and a failure is a
result with `ok=False` and a readable reason. An exception gives a model nothing to say and
invites it to invent something; "no financial statements are available for this ticker" gives
it something true to say. Section 22's preference for abstaining over hallucinating only
works if abstention is expressible.

**Every value carries its provenance.** Tools may only emit source facts, model calculations
and model assumptions (see provenance.py). Interpretation and judgement are the agent's to
add and are labelled there, so a model-authored sentence can never reach a reader wearing the
same label as a filed figure.

The schemas are written in the shape tool-calling APIs expect, so `specifications()` can be
passed to a model as the tool list without further translation.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.financial import module1, module2, module3
from src.financial.provenance import ToolResult


@dataclass(frozen=True)
class Tool:
    """One callable financial capability, with the schema the model sees."""

    name: str
    description: str
    parameters: dict
    function: Callable[..., ToolResult]
    engine: str

    def specification(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


COMPANY = {"type": "string", "description": "Company name or ticker, e.g. 'AAPL' or 'Apple'."}
TICKER = {"type": "string", "description": "Ticker symbol, e.g. 'AAPL'."}
PERIOD = {"type": "string",
          "description": "History window: '1y', '3y', '5y' or '10y'. Defaults to '5y'."}
CONFIDENCE = {"type": "number",
              "description": "Confidence level between 0.5 and 1, e.g. 0.99. Defaults to 0.99."}


# Registration is explicit rather than derived by introspection: the description a model reads
# is the single biggest determinant of whether it picks the right tool, so each is written
# deliberately and says when *not* to use the tool as well as when to.
TOOLS: list[Tool] = [
    Tool(
        name="get_company_profile",
        description=(
            "Identity, listing region, reporting currency, share price, market cap and how "
            "many years of filed history are available. Call this first for an unfamiliar "
            "company: it confirms the ticker resolved to the right business and reveals "
            "whether the data came from SEC filings or a market data provider."
        ),
        parameters=_schema({"query": COMPANY}, ["query"]),
        function=module1.get_company_profile,
        engine="Module 1",
    ),
    Tool(
        name="get_financial_statements",
        description=(
            "Filed statement lines by fiscal year: revenue, net income, EBITDA, EBIT, "
            "operating cash flow, capex, debt, cash and equity. Use for questions about "
            "reported figures. For margins, returns or growth rates use get_financial_ratios "
            "instead, which computes them rather than making you divide."
        ),
        parameters=_schema({"query": COMPANY,
                            "years": {"type": "integer",
                                      "description": "How many recent years. Defaults to 10."}},
                           ["query"]),
        function=module1.get_financial_statements,
        engine="Module 1",
    ),
    Tool(
        name="get_financial_ratios",
        description=(
            "Computed ratios: margins, returns on capital and equity, leverage, interest "
            "cover, cash conversion and growth. Every value is calculated in Python from "
            "filed data. Use this rather than deriving a ratio yourself from statement lines."
        ),
        parameters=_schema({"query": COMPANY}, ["query"]),
        function=module1.get_financial_ratios,
        engine="Module 1",
    ),
    Tool(
        name="get_financial_trends",
        description=(
            "Narrated multi-year trend readings: which metrics are improving, which are "
            "deteriorating, and over what period. Use for 'how has this changed' questions "
            "rather than fetching several years and comparing them yourself."
        ),
        parameters=_schema({"query": COMPANY}, ["query"]),
        function=module1.get_financial_trends,
        engine="Module 1",
    ),
    Tool(
        name="get_financial_health",
        description=(
            "The Financial Health Score, 0 to 100, with its pillar scores and rating. Also "
            "reports which metrics were unavailable and excluded, because a score built from "
            "six metrics is a weaker claim than the same score built from nine, and the "
            "difference must be stated when quoting it."
        ),
        parameters=_schema({"query": COMPANY}, ["query"]),
        function=module1.get_financial_health,
        engine="Module 1",
    ),
    Tool(
        name="get_valuation",
        description=(
            "Intrinsic valuation: DCF fair value, comparables where real sector peers exist, "
            "WACC, beta, terminal growth and the reverse-DCF market-implied discount rate. "
            "The primary tool for 'what is it worth' or 'is it undervalued'. The result "
            "carries a calibration warning that must be honoured: this DCF reads "
            "systematically below market, so never quote the fair value on its own."
        ),
        parameters=_schema({"query": COMPANY,
                            "horizon": {"type": "integer",
                                        "description": "Forecast years. Defaults to 5."}},
                           ["query"]),
        function=module2.get_valuation,
        engine="Module 2",
    ),
    Tool(
        name="get_valuation_sensitivity",
        description=(
            "What the valuation actually depends on, including the share of enterprise value "
            "sitting in the terminal value and the gap between the model's WACC and the one "
            "the market price implies. Use for 'how sensitive is this' or 'what would have to "
            "be true' questions."
        ),
        parameters=_schema({"query": COMPANY}, ["query"]),
        function=module2.get_valuation_sensitivity,
        engine="Module 2",
    ),
    Tool(
        name="get_scenario_analysis",
        description=(
            "Bull, base and bear valuations, each independently recomputed rather than a "
            "percentage haircut on the base case. Use for 'what if' and downside questions."
        ),
        parameters=_schema({"query": COMPANY}, ["query"]),
        function=module2.get_scenario_analysis,
        engine="Module 2",
    ),
    Tool(
        name="get_market_data",
        description=(
            "Latest close, date range and observation count, with a freshness classification "
            "that is DELAYED or STALE and never LIVE. Use when the question is about the "
            "current price or how current the data is."
        ),
        parameters=_schema({"ticker": TICKER, "period": PERIOD}, ["ticker"]),
        function=module3.get_market_data,
        engine="Module 3",
    ),
    Tool(
        name="get_volatility",
        description=(
            "Realised volatility, daily and annualised, plus whether the current level is "
            "elevated or subdued against this security's own history rather than an absolute "
            "threshold."
        ),
        parameters=_schema({"ticker": TICKER, "period": PERIOD,
                            "window": {"type": "integer",
                                       "description": "Rolling window in trading days. Defaults to 60."}},
                           ["ticker"]),
        function=module3.get_volatility,
        engine="Module 3",
    ),
    Tool(
        name="get_beta",
        description=(
            "Beta against a benchmark, with R-squared and an explicit confidence field. A low "
            "R-squared beta looks meaningful and largely is not, so the confidence must be "
            "reported alongside the number."
        ),
        parameters=_schema({"ticker": TICKER,
                            "benchmark": {"type": "string",
                                          "description": "Benchmark ticker. Defaults to '^GSPC'."},
                            "period": PERIOD},
                           ["ticker"]),
        function=module3.get_beta,
        engine="Module 3",
    ),
    Tool(
        name="get_drawdown",
        description=(
            "Current and maximum peak-to-trough decline, when the worst occurred, whether the "
            "security is currently in drawdown, and the deepest episodes with recovery dates."
        ),
        parameters=_schema({"ticker": TICKER, "period": PERIOD}, ["ticker"]),
        function=module3.get_drawdown,
        engine="Module 3",
    ),
    Tool(
        name="get_var",
        description=(
            "Value at Risk by all three methods at once: historical, parametric (normal and "
            "Student-t) and Monte Carlo. They are returned together deliberately, because "
            "they disagree in a structured way and quoting one as 'the' VaR misrepresents "
            "that. These figures are unconditional; for risk given today's volatility use "
            "get_garch_forecast."
        ),
        parameters=_schema({"ticker": TICKER, "confidence": CONFIDENCE,
                            "horizon_days": {"type": "integer",
                                             "description": "Holding period in trading days. Defaults to 1."},
                            "period": PERIOD,
                            "portfolio_value": {"type": "number",
                                                "description": "Optional position size, to express VaR in currency."}},
                           ["ticker"]),
        function=module3.get_var,
        engine="Module 3",
    ),
    Tool(
        name="get_expected_shortfall",
        description=(
            "Expected Shortfall: the average loss given that the bad case happens, which VaR "
            "structurally cannot say. Use whenever the question is about the severity of the "
            "tail rather than the threshold."
        ),
        parameters=_schema({"ticker": TICKER, "confidence": CONFIDENCE, "period": PERIOD},
                           ["ticker"]),
        function=module3.get_expected_shortfall,
        engine="Module 3",
    ),
    Tool(
        name="get_garch_forecast",
        description=(
            "Conditional volatility and conditional VaR from a fitted GARCH(1,1): risk given "
            "today's regime rather than averaged across the sample. The right tool when the "
            "question is about risk *now*, or about whether volatility is rising or falling."
        ),
        parameters=_schema({"ticker": TICKER, "confidence": CONFIDENCE,
                            "horizon_days": {"type": "integer",
                                             "description": "Forecast horizon in trading days. Defaults to 10."}},
                           ["ticker"]),
        function=module3.get_garch_forecast,
        engine="Module 3",
    ),
    Tool(
        name="get_risk_metrics",
        description=(
            "The consolidated risk picture in one call: volatility, beta, drawdown, VaR, "
            "Expected Shortfall and the GARCH conditional read. Use for broad questions like "
            "'what are the risks'. For a single specific metric, call that metric's own tool "
            "instead, which is faster."
        ),
        parameters=_schema({"ticker": TICKER,
                            "benchmark": {"type": "string",
                                          "description": "Benchmark ticker. Defaults to '^GSPC'."},
                            "period": PERIOD},
                           ["ticker"]),
        function=module3.get_risk_metrics,
        engine="Module 3",
    ),
]

REGISTRY: dict[str, Tool] = {t.name: t for t in TOOLS}


def specifications() -> list[dict]:
    """The tool list in the shape a tool-calling API expects."""
    return [t.specification() for t in TOOLS]


def names() -> list[str]:
    return [t.name for t in TOOLS]


def by_engine() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for tool in TOOLS:
        grouped.setdefault(tool.engine, []).append(tool.name)
    return grouped


def call(name: str, **kwargs: Any) -> ToolResult:
    """Invoke a tool by name. Never raises; failures come back as results.

    An unknown tool name is itself a failure result rather than a KeyError, because models do
    occasionally invent a plausible-sounding tool, and the useful response is to tell it what
    actually exists rather than to crash the turn.
    """
    tool = REGISTRY.get(name)
    if tool is None:
        return ToolResult.failure(
            name, f"No tool named '{name}'. Available tools: {', '.join(sorted(REGISTRY))}.")

    allowed = set(tool.parameters.get("properties", {}))
    unexpected = set(kwargs) - allowed
    if unexpected:
        return ToolResult.failure(
            name,
            f"Unexpected argument(s) {sorted(unexpected)} for '{name}'. "
            f"It accepts: {sorted(allowed)}.")

    missing = set(tool.parameters.get("required", [])) - set(kwargs)
    if missing:
        return ToolResult.failure(name, f"Missing required argument(s) {sorted(missing)} for '{name}'.")

    try:
        return tool.function(**kwargs)
    except Exception as exc:  # noqa: BLE001 - the whole point: nothing escapes to the model
        return ToolResult.failure(name, f"{name} failed: {type(exc).__name__}: {exc}")
