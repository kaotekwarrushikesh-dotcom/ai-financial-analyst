# Module 4: AI Financial Analyst

> An AI-powered financial analyst that researches companies, interprets financial data,
> evaluates valuation and risk, and produces evidence-backed investment insights.

**Status: Phase 1 of 16 built and tested.** Modules 1, 2 and 3 are integrated behind a single
tool interface: 16 financial tools, every one of them a real function ending in a
deterministic calculation, each result tagged with where its numbers came from. The LLM agent,
RAG, query router, report generation and dashboard are **not built yet**. This README says so
rather than describing the finished system in the present tense. See [Roadmap](#roadmap).

The architecture this module exists to demonstrate:

```text
                    LLM: chooses WHAT to compute, never computes it
                                      |
                        unified tool interface (this repo)
                                      |
        +---------------------+-------+-------+---------------------+
        |                     |               |                     |
    Module 1              Module 2        Module 3            RAG (Phase 4)
  statements,            DCF, WACC,      VaR, ES, GARCH,       filings and
  ratios, health         comparables,    beta, drawdown,       transcripts
  score                  scenarios       backtesting
```

## What Phase 1 actually delivers

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q -m "not live"   # contract tests, offline
.venv/bin/python -m pytest tests/ -q -m live         # hits the real engines
```

```python
from src.ai import tools

tools.call("get_financial_health", query="AAPL")
tools.call("get_valuation", query="MSFT")
tools.call("get_var", ticker="AAPL", confidence=0.99)
```

| Engine | Tools |
|---|---|
| Module 1 | `get_company_profile`, `get_financial_statements`, `get_financial_ratios`, `get_financial_trends`, `get_financial_health` |
| Module 2 | `get_valuation`, `get_valuation_sensitivity`, `get_scenario_analysis` |
| Module 3 | `get_market_data`, `get_volatility`, `get_beta`, `get_drawdown`, `get_var`, `get_expected_shortfall`, `get_garch_forecast`, `get_risk_metrics` |

`tools.specifications()` returns these as JSON schemas in the shape a tool-calling API expects,
so Phase 6 can hand the list straight to a model.

## The blocker that had to be removed first

The three engines could not be imported into the same Python process. Each shipped a
top-level package literally named `src`, so Python resolves whichever appears first on the
path and the other two become unreachable. No amount of adapter code fixes that; it is a
namespace collision, not an integration problem.

The alternative was vendoring copies of each engine into this repo, which is how the analyst
quietly becomes a fourth, divergent implementation of the same finance. So each engine was
renamed to its own namespace and given packaging metadata instead:

| Repo | Was | Now | Installs as |
|---|---|---|---|
| Module 1 | `src/` | `fsi/` | `financial-statement-intelligence` |
| Module 2 | `src/valuation/` | `valuation_engine/` | `valuation-engine` |
| Module 3 | `src/risk/` | `risk_engine/` | `risk-engine` |

They are now pinned in `requirements.txt` as git dependencies, so **a fix in an engine reaches
this analyst without anything being copied between repositories**. Two supporting changes were
needed to make that real rather than nominal:

- Module 3's `data_loader` reached out to a sibling `config/` directory through a
  `sys.path.insert`. That works from a checkout and breaks the moment the package is
  installed, so the settings moved inside the package.
- Module 2's entire valuation pipeline lived in `app.py` behind Streamlit's `@st.cache_data`.
  Importing it ran `st.set_page_config` and tied a library's caching policy to the lifetime of
  a browser tab. It moved to `valuation_engine/pipeline.py`, and the app now imports it and
  applies its own caching, which is where a caching policy belongs.

Both deployed Streamlit apps were booted and verified after the change, and all three engines'
test suites (26, 100 and 181) pass unchanged.

## Provenance: the rule the tool layer enforces

An AI analyst mixes four different kinds of statement in one paragraph, and a reader who
cannot tell them apart will treat the weakest as the strongest. Every value a tool returns is
therefore tagged at the point it is produced:

```text
[SOURCE_FACT]        latest_revenue: 391035000000 USD (source: SEC EDGAR, as of FY2024)
[MODEL_CALCULATION]  wacc: 0.09299 (source: Module 2 WACC (CAPM))
[MODEL_ASSUMPTION]   terminal_growth: 0.02 (source: Module 2 assumptions)
```

**Tools can only ever emit those three.** `AI_INTERPRETATION` and `AI_JUDGEMENT` raise a
`ValueError` if a tool tries to produce one, because interpretation and opinion belong to the
agent layer and must be labelled there. A model-authored sentence therefore cannot reach a
reader wearing the same label as a filed figure, and that is enforced in code rather than
requested in a prompt.

## Two guarantees, and why they are code rather than prompt instructions

**A tool never raises at the model.** Every call returns a `ToolResult`; failure is
`ok=False` with a readable reason. An exception gives a model nothing true to say and invites
it to fill the silence, which is the exact hallucination this architecture exists to prevent.
An invented tool name returns the list of real ones rather than a `KeyError`.

**Each engine's own caveats travel with its numbers.** The engines documented their
limitations in their READMEs, which the model will never read, so the ones that change how a
figure should be interpreted are attached to the results:

- Module 2's valuation carries its **calibration warning**: that DCF reads systematically
  below market, so the fair value must never be quoted on its own. A test asserts the warning
  is present on every valuation.
- `get_var` returns **all three VaR methods together**, because they disagree in a structured
  way (the normal model is the more conservative at 95% and understates at 99%), and handing a
  model one number invites it to report that number as the risk.
- Every VaR figure except the GARCH one is flagged **unconditional**: it describes the whole
  sample, not today's volatility regime.
- Historical VaR warns when it sits within 90% of the worst loss ever observed, because it
  cannot produce a number worse than something that has already happened.
- Market data is classified **DELAYED or STALE, never LIVE**.

## Tests

22 tests: 16 offline contract tests and 6 live integration tests, split by a `live` marker so
the contract can be checked without the network.

The offline set pins the provenance rules and the never-raise guarantee. The live set is
deliberately not mocked, because a tool layer tested only against mocks proves the mocks work:
it imports all three engines in one process, runs real valuations and VaR, and checks the same
tool works for more than one company.

## Roadmap

Phase 1 is done. The remaining phases follow the build order, each tested before the next
begins.

- **Phase 2** Live company and market data access
- **Phase 3** Document ingestion (filings, transcripts, presentations)
- **Phase 4** RAG retrieval with citations
- **Phase 5** Remaining financial tools (peers, stress testing)
- **Phase 6** LLM tool calling
- **Phase 7** Query router
- **Phases 8-11** Financial, valuation, risk and investment-synthesis workflows
- **Phase 12** Citation and evidence layer
- **Phase 13** Report generation
- **Phase 14** Streamlit interface
- **Phase 15** AI evaluation (numerical accuracy, retrieval, citation, hallucination rate)
- **Phase 16** Deployment

## Known limitations so far

1. **There is no AI in this module yet.** Phase 1 is the tool surface an agent will call. The
   tools work and are tested; nothing chooses between them yet.
2. **Module 2's curated workflow is unavailable to an installed copy.** The full Nifty
   workflow reads CSVs checked into that repo, so an installed package gets the live "quick"
   valuation path instead: a DCF and scenarios with no comparables. The adapter falls back
   automatically and the result says which mode produced it, but it means comparables are
   currently unavailable through this interface for every company.
3. **No caching layer.** Each tool call refetches, so a broad question that touches six tools
   pays six times. Phase 2 addresses this; until then, latency is real and a full risk sweep
   takes several seconds.
4. **Peer comparison, stress testing and document search are not built.** They appear in the
   target tool list in the brief and are Phases 3 to 5.
5. **The `live` tests depend on Yahoo Finance being reachable** and on specific tickers still
   resolving, so they are excluded from the default run rather than being made to pass by
   mocking the thing they exist to check.
