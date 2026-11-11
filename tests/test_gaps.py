import pytest

from test_drive import gaps

ASK = "Can you check my Gmail inbox for the invoice from Acme?"


@pytest.mark.parametrize("reply", [
    "I don't have access to your Gmail inbox, so I can't check it.",
    "I have no such tool.",
    "I have no cultural-dimensions dataset tool available — stopping here.",
    "I don't have a tool to read your email.",
    "I'm unable to send messages to Slack directly from here.",
    "Unfortunately I can't browse the web in real time.",
    "There is no Spotify integration available in this session.",
    "I lack a PDF parsing tool here, so I can't extract the tables.",
    "That's beyond my current capabilities: I can't control smart-home devices.",
])
def test_capability_walls_are_gaps(reply):
    gap = gaps.detect(ASK, reply)
    assert gap.is_gap, (reply, gap)
    assert gap.failure and len(gap.failure) <= 220


@pytest.mark.parametrize("reply", [
    "Here is the invoice summary: Acme billed $1,200 on Nov 3.",
    "I can't help with that request because it would be harmful.",
    "I can't predict next week's stock price, but here are the drivers.",
    "I don't have enough information to answer: which Acme do you mean?",
    "I can see three invoices. I can't find one from Acme, though; did it come from another address?",
    "",
])
def test_ordinary_answers_and_policy_refusals_are_not_gaps(reply):
    assert not gaps.detect(ASK, reply).is_gap


def test_late_mention_in_long_answer_is_ignored():
    reply = "Here's the full plan.\n" + ("Step details. " * 150) + "I can't access your calendar though."
    assert not gaps.detect(ASK, reply).is_gap


def test_tool_failure_alone_is_a_gap():
    f = gaps.tool_failure("browser_navigate", '{"error": "Tool browser_navigate is not available: install the browser toolset"}')
    assert f and f.tool == "browser_navigate"
    gap = gaps.detect("open example.com and screenshot it", "Here is what I would do instead.", [f])
    assert gap.is_gap and "tool browser_navigate failed as unavailable" in gap.reasons
    assert gap.failure.startswith("browser_navigate:")


def test_ordinary_tool_errors_are_not_capability_failures():
    assert gaps.tool_failure("read_file", '{"error": "File not found: notes.txt"}') is None
    assert gaps.tool_failure("terminal", "exit code 1: syntax error near token") is None
    assert gaps.tool_failure("web_search", '{"results": []}', status="ok") is None


def test_weak_signal_alone_is_ambiguous_not_a_gap():
    gap = gaps.detect("tell me a joke", "I can't think of a good one right now.")
    assert not gap.is_gap


def test_need_needs_words():
    gap = gaps.detect("hi", "I don't have access to that tool.")
    assert not gap.is_gap


def test_need_from_strips_command_and_pleasantries():
    assert gaps.need_from("/btw  hey, can you post this to Bluesky?") == "can you post this to Bluesky?"
    assert len(gaps.need_from("x" * 1000)) == 300
