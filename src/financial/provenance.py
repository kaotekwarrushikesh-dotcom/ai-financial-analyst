"""Provenance: where every number came from, carried with the number itself.

Section 32 of the build brief asks that every important output be classified. That is not
presentation polish. An AI analyst mixes four genuinely different kinds of statement in the
same paragraph, and a reader who cannot tell them apart will treat the weakest as if it were
the strongest:

    SOURCE_FACT         A figure as filed or quoted. "Revenue was $391.0bn in FY2024."
    MODEL_CALCULATION   Deterministic Python output. "WACC is 11.08%."
    MODEL_ASSUMPTION    An input someone chose. "Terminal growth is capped at 2.5%."
    AI_INTERPRETATION   The LLM reading the numbers. "Margin expansion looks structural."
    AI_JUDGEMENT        The LLM's opinion. "The valuation appears attractive."

The rule this file exists to enforce is that the tool layer can only ever produce the first
three. An LLM cannot label its own output as a source fact, because it never touches this
code path: facts and calculations are stamped at the point they are produced, by the
function that produced them, before any model sees them. Interpretation and judgement are
added later, by the agent, and are the only two categories it is allowed to mint.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Provenance(str, Enum):
    SOURCE_FACT = "SOURCE_FACT"
    MODEL_CALCULATION = "MODEL_CALCULATION"
    MODEL_ASSUMPTION = "MODEL_ASSUMPTION"
    AI_INTERPRETATION = "AI_INTERPRETATION"
    AI_JUDGEMENT = "AI_JUDGEMENT"


# The three a tool is permitted to emit. Anything else has to come from the agent layer,
# which is what stops a model-authored sentence from being presented as filed data.
TOOL_PERMITTED = frozenset({
    Provenance.SOURCE_FACT,
    Provenance.MODEL_CALCULATION,
    Provenance.MODEL_ASSUMPTION,
})


class DataStatus(str, Enum):
    """How fresh the underlying data is.

    Deliberately borrowed from Module 3's vocabulary, which never claims LIVE: yfinance is an
    unofficial interface with no real-time guarantee, so the strongest honest claim is
    DELAYED. Section 3 of the brief requires this distinction be shown rather than implied.
    """

    DELAYED = "DELAYED"
    STALE = "STALE"
    CACHED = "CACHED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Fact:
    """One value, with everything needed to say where it came from and how much to trust it."""

    name: str
    value: Any
    provenance: Provenance
    source: str
    unit: str | None = None
    as_of: str | None = None
    note: str | None = None

    def __post_init__(self):
        if self.provenance not in TOOL_PERMITTED:
            raise ValueError(
                f"{self.provenance.value} cannot be produced by a tool. Tools emit source "
                "facts, model calculations and model assumptions; interpretation and "
                "judgement belong to the agent layer and must be labelled there."
            )

    def render(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        line = f"[{self.provenance.value}] {self.name}: {self.value}{unit} (source: {self.source}"
        if self.as_of:
            line += f", as of {self.as_of}"
        line += ")"
        return line + (f" — {self.note}" if self.note else "")


@dataclass
class ToolResult:
    """What every financial tool returns.

    `ok=False` with a populated `error` is a first-class outcome rather than an exception,
    because the consumer is an LLM: a tool that raises gives the model nothing to reason
    about and invites it to fill the silence, while a tool that returns "no statements
    available for this ticker" gives it something true to say. Section 22's rule that the
    system should prefer abstaining over hallucinating needs the abstention to be expressible.
    """

    tool: str
    # Defaults to False so a tool that returns early, or raises partway through building its
    # result, can never be mistaken for one that succeeded. Success has to be stated.
    ok: bool = False
    facts: list[Fact] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    status: DataStatus = DataStatus.UNKNOWN
    retrieved_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def failure(cls, tool: str, error: str) -> "ToolResult":
        return cls(tool=tool, ok=False, error=error)

    def add(self, name: str, value: Any, provenance: Provenance, source: str,
            unit: str | None = None, as_of: str | None = None, note: str | None = None) -> "Fact":
        fact = Fact(name=name, value=value, provenance=provenance, source=source,
                    unit=unit, as_of=as_of, note=note)
        self.facts.append(fact)
        return fact

    def get(self, name: str) -> Any:
        for fact in self.facts:
            if fact.name == name:
                return fact.value
        return None

    def to_model_payload(self) -> dict:
        """The shape handed to the LLM.

        Facts are flattened to strings carrying their own provenance tag, so the label
        travels with the value into the model's context rather than sitting in a separate
        structure the model has to be trusted to consult.
        """
        if not self.ok:
            return {"tool": self.tool, "ok": False, "error": self.error}
        return {
            "tool": self.tool,
            "ok": True,
            "data_status": self.status.value,
            "retrieved_at": self.retrieved_at,
            "sources": self.sources,
            "facts": [f.render() for f in self.facts],
            "data": self.data,
            "warnings": self.warnings,
        }
