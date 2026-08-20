"""Tests for Phase 1: the unified tool interface over Modules 1, 2 and 3.

Two kinds of test live here and they are worth separating. Most are offline and pin the
contract: schema shape, provenance rules, and the guarantee that a tool never raises at the
model. A handful marked `live` actually hit the engines and the network, because a tool layer
that has only ever been tested against mocks proves the mocks work.

The most important property under test is negative: no matter what is thrown at `call`, it
returns a ToolResult. An LLM handed an exception has nothing true to say and will fill the
silence, which is precisely the hallucination this architecture exists to prevent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai import tools
from src.financial.provenance import (
    TOOL_PERMITTED,
    DataStatus,
    Fact,
    Provenance,
    ToolResult,
)

live = pytest.mark.live


# --- Provenance rules -----------------------------------------------------------------------

def test_a_tool_cannot_emit_an_ai_authored_label():
    """The central guarantee: a model-written sentence can never wear the same label as a
    filed figure, because the label is applied by the function that produced the value."""
    for banned in (Provenance.AI_INTERPRETATION, Provenance.AI_JUDGEMENT):
        with pytest.raises(ValueError, match="cannot be produced by a tool"):
            Fact(name="verdict", value="looks attractive", provenance=banned, source="the model")


def test_the_three_permitted_labels_are_accepted():
    for allowed in TOOL_PERMITTED:
        fact = Fact(name="x", value=1, provenance=allowed, source="test")
        assert fact.provenance is allowed


def test_a_fact_renders_its_own_label_and_source():
    fact = Fact(name="revenue", value=391_035, provenance=Provenance.SOURCE_FACT,
                source="SEC EDGAR", unit="USDm", as_of="FY2024")
    rendered = fact.render()
    assert "[SOURCE_FACT]" in rendered
    assert "SEC EDGAR" in rendered and "FY2024" in rendered and "USDm" in rendered


def test_results_default_to_not_ok_so_success_must_be_stated():
    """A tool that returns early, or raises partway through building its result, must never be
    mistaken for one that succeeded."""
    assert ToolResult(tool="anything").ok is False


def test_the_model_payload_carries_provenance_into_the_prompt():
    """The label has to travel with the value into the model's context, not sit in a side
    structure the model would have to be trusted to consult."""
    result = ToolResult(tool="t", ok=True, status=DataStatus.DELAYED)
    result.add("wacc", 0.1108, Provenance.MODEL_CALCULATION, "Module 2")
    payload = result.to_model_payload()
    assert payload["data_status"] == "DELAYED"
    assert any("[MODEL_CALCULATION]" in f for f in payload["facts"])


def test_a_failed_result_payload_is_short_and_says_why():
    payload = ToolResult.failure("get_valuation", "no statements available").to_model_payload()
    assert payload == {"tool": "get_valuation", "ok": False, "error": "no statements available"}


# --- Registry and schemas ---------------------------------------------------------------------

def test_every_engine_is_represented():
    grouped = tools.by_engine()
    assert set(grouped) == {"Module 1", "Module 2", "Module 3"}
    assert all(len(names) >= 3 for names in grouped.values())


def test_tool_names_are_unique():
    assert len(tools.names()) == len(set(tools.names()))


def test_every_schema_is_well_formed_for_a_tool_calling_api():
    for spec in tools.specifications():
        assert set(spec) == {"name", "description", "input_schema"}
        schema = spec["input_schema"]
        assert schema["type"] == "object"
        assert schema["properties"]
        # Everything named as required must actually be declared.
        assert set(schema["required"]) <= set(schema["properties"])


def test_every_tool_description_is_substantial():
    """The description is the single biggest determinant of whether a model picks the right
    tool, so a one-line stub is a routing bug waiting to happen."""
    for tool in tools.TOOLS:
        assert len(tool.description) > 120, f"{tool.name} needs a fuller description"


def test_the_registry_and_the_list_agree():
    assert set(tools.REGISTRY) == set(tools.names())
    for name, tool in tools.REGISTRY.items():
        assert tool.name == name


# --- The never-raise guarantee -------------------------------------------------------------------

def test_an_unknown_tool_name_returns_a_result_listing_the_real_ones():
    """Models do invent plausible-sounding tools. The useful answer is what exists, not a
    KeyError that ends the turn."""
    result = tools.call("get_insider_sentiment", query="AAPL")
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert "No tool named" in result.error
    assert "get_valuation" in result.error


def test_an_unexpected_argument_is_refused_with_the_accepted_list():
    result = tools.call("get_beta", ticker="AAPL", timeframe="weekly")
    assert result.ok is False
    assert "timeframe" in result.error
    assert "benchmark" in result.error


def test_a_missing_required_argument_is_named():
    result = tools.call("get_var")
    assert result.ok is False
    assert "ticker" in result.error


def test_an_exception_inside_a_tool_becomes_a_result_not_a_traceback(monkeypatch):
    def explode(**_kwargs):
        raise RuntimeError("upstream engine fell over")

    monkeypatch.setitem(
        tools.REGISTRY, "get_valuation",
        tools.Tool(name="get_valuation", description="x" * 130,
                   parameters={"type": "object", "properties": {"query": {"type": "string"}},
                               "required": ["query"]},
                   function=explode, engine="Module 2"))

    result = tools.call("get_valuation", query="AAPL")
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert "upstream engine fell over" in result.error
    assert "RuntimeError" in result.error


def test_a_nonsense_ticker_abstains_rather_than_inventing(monkeypatch):
    result = tools.call("get_company_profile", query="NOT_A_REAL_COMPANY_XYZQ")
    assert result.ok is False
    assert result.facts == []


# --- Live integration, which is the point of the phase ----------------------------------------

@live
def test_all_three_engines_run_in_one_process():
    """The blocker Phase 1 existed to remove: before the packaging work, importing more than
    one engine was impossible because all three shipped a package named `src`."""
    import fsi.company            # noqa: F401
    import risk_engine.garch      # noqa: F401
    import valuation_engine.pipeline  # noqa: F401


@live
def test_module_1_tools_return_tagged_facts():
    result = tools.call("get_financial_health", query="AAPL")
    assert result.ok
    assert isinstance(result.get("overall_score"), float)
    assert all(f.provenance in TOOL_PERMITTED for f in result.facts)


@live
def test_module_2_valuation_always_carries_the_calibration_warning():
    """The engine's own rule is that its DCF is never quoted alone. An AI analyst is exactly
    the consumer most likely to break that, so the warning rides on the result."""
    result = tools.call("get_valuation", query="AAPL")
    assert result.ok
    assert any("never present the dcf fair value on its own" in w.lower() for w in result.warnings)


@live
def test_module_3_var_returns_all_three_methods_together():
    """Returned together deliberately: they disagree in a structured way, and handing a model
    one number invites it to report that number as the risk."""
    result = tools.call("get_var", ticker="AAPL")
    assert result.ok
    for method in ("var_historical", "var_parametric_normal", "var_parametric_student_t",
                   "var_monte_carlo_bootstrap"):
        assert isinstance(result.get(method), float)


@live
def test_risk_data_is_never_reported_as_live():
    """yfinance carries no real-time guarantee, so DELAYED is the strongest honest claim."""
    result = tools.call("get_market_data", ticker="AAPL")
    assert result.ok
    assert result.status in (DataStatus.DELAYED, DataStatus.STALE)
    assert result.status.value != "LIVE"


@live
def test_the_same_tool_works_for_more_than_one_company():
    """Section 3's requirement: nothing may be hard-coded to a single company."""
    for ticker in ("AAPL", "MSFT"):
        result = tools.call("get_financial_ratios", query=ticker)
        assert result.ok, f"{ticker} failed: {result.error}"
        assert result.facts
