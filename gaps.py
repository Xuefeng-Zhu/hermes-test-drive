"""Cheap, local capability-gap detection.

A *gap* is a turn where Hermes hit a wall: the reply says it cannot do the thing, a tool failed as
unknown/unavailable, or the user asked for a capability and got a refusal-shaped answer. Signals
are scored; one strong signal, or two weak ones, makes a gap. Everything here is regex and
arithmetic: no network, no model. (An optional model check sits in core and is off by default.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# The assistant admits a missing capability.
_ADJ = r"(?:[\w/-]+\s+){0,4}?"
_THING = r"(?:tools?|integrations?|plugins?|access|ability|way|means|capability|connection|api)"
_CANT = re.compile(
    r"\b(?:i(?:\s+am|'m)\s+(?:not\s+able|unable)|i\s+(?:can(?:no|')t|cannot|do(?:n'?t|\s+not)\s+"
    r"(?:currently\s+)?have\s+(?:the\s+|a\s+|an\s+|any\s+|such\s+an?\s+)?" + _ADJ + _THING +
    r"|(?:have|'ve\s+got)\s+no\s+" + _ADJ + _THING + r"|lack\s+(?:the\s+|a\s+|an\s+|any\s+)?" + _ADJ + _THING + r")"
    r"|(?:there(?:'s|\s+is)\s+no|no)\s+" + _ADJ + r"(?:tool|integration|plugin)s?\s+(?:available|for|to|that|here)"
    r"|(?:isn'?t|is\s+not|aren'?t|are\s+not)\s+(?:available|supported|something\s+i\s+can)"
    r"|(?:beyond|outside)\s+(?:my|the)\s+(?:current\s+)?(?:capabilities|tools|abilities)"
    r"|not\s+(?:something\s+i(?:'m|\s+am)\s+able|equipped|connected\s+to))",
    re.IGNORECASE)
# Words that make a "can't" about capability rather than policy/ethics/knowledge.
_CAPABILITY_CONTEXT = re.compile(
    r"\b(?:tool|tools|access|integration|plugin|connect|api|directly|real[- ]time|live|browse|send|post|"
    r"install(?:ed)?|account|device|fetch|retrieve|read|open|control|execute|run|available|ability|built-in)\b",
    re.IGNORECASE)
# Refusals for reasons a plugin will not fix.
_POLICY = re.compile(
    r"\b(?:against\s+(?:my|the)\s+(?:policy|guidelines)|harmful|illegal|unethical|not\s+appropriate|"
    r"i\s+won'?t\s+help|violat\w+|privacy\s+reasons|i\s+can'?t\s+(?:say|know|predict|guarantee|be\s+sure|recall))\b",
    re.IGNORECASE)
# The user asks for a capability.
_ASK = re.compile(
    r"^\s*(?:can|could|would|will)\s+you\b|\b(?:please\s+)?(?:send|post|check|track|fetch|book|order|"
    r"convert|transcribe|translate|generate|create|monitor|watch|schedule|remind|search|look\s+up|download|"
    r"upload|read|summari[sz]e|connect|sync|control|turn\s+on|turn\s+off|deploy|publish|calculate|compute|"
    r"analy[sz]e|compare)\b", re.IGNORECASE)
# A tool result that means "the capability is not there", not "the input was wrong".
_TOOL_MISSING = re.compile(
    r"\b(?:unknown\s+tool|no\s+such\s+tool|tool\s+(?:not\s+found|unavailable|is\s+not\s+available|not\s+enabled)|"
    r"not\s+(?:available|enabled|installed|configured|supported)|is\s+disabled|requires?\s+(?:the\s+)?\w+\s+"
    r"(?:plugin|integration|toolset)|missing\s+(?:api\s+key|credentials?)|unsupported)\b", re.IGNORECASE)

STRONG = 2.0
WEAK = 1.0
THRESHOLD = 2.0


@dataclass
class ToolFailure:
    tool: str
    message: str


@dataclass
class Gap:
    score: float
    reasons: list[str] = field(default_factory=list)
    need: str = ""
    failure: str = ""

    @property
    def is_gap(self) -> bool:
        return self.score >= THRESHOLD


def _sentence_around(text: str, start: int, end: int, limit: int = 220) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    stops = [i for i in (text.find(".", end), text.find("\n", end)) if i != -1]
    right = min(stops) + 1 if stops else len(text)
    snippet = " ".join(text[left:right].split())
    return snippet if len(snippet) <= limit else snippet[: limit - 1] + "…"


def tool_failure(tool_name: str, result: object, status: Optional[str] = None,
                 error_message: Optional[str] = None) -> Optional[ToolFailure]:
    """A post_tool_call result that says a capability is missing (None for ordinary errors)."""
    text = " ".join(str(x) for x in (error_message, result) if x)[:2000]
    if status not in (None, "", "error") and not error_message:
        return None
    m = _TOOL_MISSING.search(text)
    if not m:
        return None
    return ToolFailure(tool_name, _sentence_around(text, m.start(), m.end()))


def detect(user_message: str, response: str, tool_failures: Optional[list[ToolFailure]] = None) -> Gap:
    user_message = user_message or ""
    response = response or ""
    gap = Gap(score=0.0, need=need_from(user_message))
    head = response[:1500]  # the admission is near the top; long answers that mention "can't" late are fine
    m = _CANT.search(head)
    if m:
        window = head[max(0, m.start() - 120): m.end() + 160]
        if _POLICY.search(window):
            gap.reasons.append("refusal looks like policy/knowledge, not a missing tool")
        elif _CAPABILITY_CONTEXT.search(window):
            gap.score += STRONG
            gap.reasons.append("assistant said it lacks the capability")
            gap.failure = _sentence_around(head, m.start(), m.end())
        else:
            gap.score += WEAK
            gap.reasons.append("assistant said it can't")
            gap.failure = _sentence_around(head, m.start(), m.end())
    for f in tool_failures or []:
        gap.score += STRONG
        gap.reasons.append(f"tool {f.tool} failed as unavailable")
        gap.failure = gap.failure or f"{f.tool}: {f.message}"
        break
    if gap.score >= STRONG and _ASK.search(user_message):
        gap.score += WEAK
        gap.reasons.append("user asked for a capability")
    if len(gap.need) < 6:
        gap.score = min(gap.score, THRESHOLD - 0.5)  # nothing to search for
    return gap


def need_from(user_message: str, limit: int = 300) -> str:
    text = " ".join((user_message or "").split())
    # Drop a leading slash command and pleasantries.
    text = re.sub(r"^/\S+\s*", "", text)
    text = re.sub(r"^(?:hey|hi|hello|ok(?:ay)?|so|please)[,!.\s]+", "", text, flags=re.IGNORECASE)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def is_ambiguous(gap: Gap) -> bool:
    return 0 < gap.score < THRESHOLD
