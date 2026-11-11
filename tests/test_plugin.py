import json
from pathlib import Path

import pytest

from conftest import PLUGIN_DIR, FakeLLM
import test_drive
from test_drive import core, sandbox
from test_drive.core import TestDrive

GAP_ASK = "Compute the exact culture gap analysis between Germany and Japan for my team"
GAP_REPLY = "I have no cultural-dimensions dataset tool, so I can't compute exact gaps."


def _manifest_list(key):
    lines = (PLUGIN_DIR / "plugin.yaml").read_text().splitlines()
    out, on = [], False
    for ln in lines:
        if ln.startswith(f"{key}:"):
            on = True
            continue
        if on and ln.startswith("  - "):
            out.append(ln[4:].strip())
        elif on:
            break
    return out


class Drives:
    """Records drive() calls and returns a scripted outcome."""

    def __init__(self, outcome=None):
        self.calls = []
        self.outcome = outcome

    def __call__(self, **kw):
        self.calls.append(kw)
        if self.outcome is not None:
            return self.outcome
        res = sandbox.RunResult(True, "Evaluating 68, Communicating 65, Disagreeing 58.",
                                tools_called=["culture_gap_analysis"], seconds=12.0, cost_usd=0.0004, evidence_available=True)
        return sandbox.DriveOutcome(plugin=kw["plugin"], sandbox="/tmp/hermes-testdrive-x", installed=True,
                                    with_plugin=res, cleanup_ok=True)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OTARI_API_KEY", "sk-test")
    monkeypatch.delenv("ABSTRACT_API_KEY", raising=False)
    monkeypatch.setattr(TestDrive, "_user_model", lambda self: (
        {"default": "m", "provider": "custom", "base_url": "https://api.otari.ai/api/v1", "key_env": "OTARI_API_KEY"},
        None))
    monkeypatch.setattr(TestDrive, "_secret", staticmethod(lambda n: __import__("os").environ.get(n)))


@pytest.fixture
def wired(ctx, store, env):
    drives = Drives()
    td = TestDrive(ctx, store=store, drive_fn=drives)
    ctx.td, ctx.drives = td, drives
    return ctx


def _register(ctx, store, drives):
    orig = core.TestDrive.__init__

    def init(self, c, **kw):
        orig(self, c, store=store, drive_fn=drives)
    core.TestDrive.__init__ = init
    try:
        test_drive.register(ctx)
    finally:
        core.TestDrive.__init__ = orig


@pytest.fixture
def plugin(ctx, store, env):
    drives = Drives()
    _register(ctx, store, drives)
    ctx.drives = drives
    return ctx


def cmd(ctx, text, session_id="s1"):
    return ctx.commands["testdrive"]["handler"](text, session_id=session_id)


# --- registration -----------------------------------------------------------------------------------

def test_registrations_match_manifest(plugin):
    assert sorted(plugin.tools) == sorted(_manifest_list("provides_tools"))
    assert sorted(plugin.hooks) == sorted(_manifest_list("provides_hooks"))
    assert "testdrive" in plugin.commands
    for t in plugin.tools.values():
        assert t["toolset"] == "test_drive" and t["schema"]["name"] == t["name"]


def test_register_touches_no_network(ctx, env):
    def boom(url):
        raise AssertionError("network during register")
    _register(ctx, core.CatalogStore(lambda: Path("/nonexistent"), fetcher=boom), Drives())


# --- approval gate ----------------------------------------------------------------------------------

def test_agent_tool_approval_shows_refreshed_full_pin(plugin):
    hook = plugin.hooks["pre_tool_call"]
    directive = hook(tool_name="test_drive_run", args={"plugin": "jalancost"})
    assert directive["action"] == "approve"
    assert "aa3f8f9254de" in directive["message"] and len(directive["rule_key"].split("@")[-1]) == 64
    assert hook(tool_name="test_drive_run", args={"plugin": "unknown"})["action"] == "block"
    assert hook(tool_name="terminal", args={}) is None


def test_approval_rule_changes_with_repo_or_subdir(plugin, store, monkeypatch):
    catalog = store.load()
    monkeypatch.setattr(store, "load", lambda **kwargs: catalog)
    hook = plugin.hooks["pre_tool_call"]
    args = {"plugin": "jalancost"}
    original = hook(tool_name="test_drive_run", args=args)["rule_key"]
    entry = catalog.get("jalancost")
    entry.repo = "https://github.com/example/other"
    changed_repo = hook(tool_name="test_drive_run", args=args)["rule_key"]
    entry.subdir = "other"
    changed_subdir = hook(tool_name="test_drive_run", args=args)["rule_key"]
    assert len({original, changed_repo, changed_subdir}) == 3


# --- find --------------------------------------------------------------------------------------------

def test_find_tool_returns_top3_with_trust_cards(plugin):
    out = json.loads(plugin.tools["test_drive_find"]["handler"]({"need": "culture gap analysis Germany Japan"}))
    assert out["candidates"][0]["name"] == "global-business-culture"
    assert 1 <= len(out["candidates"]) <= 3
    c = out["candidates"][0]
    assert c["in_catalog"] and len(c["sha"]) == 40 and c["requires_env"] == [] and "trust" not in c
    assert "Keys needed: none" in c["card"] and "next" in out


def test_find_tool_errors_cleanly(plugin):
    assert "error" in json.loads(plugin.tools["test_drive_find"]["handler"]({"need": "  "}))


def test_find_command_text(plugin):
    text = cmd(plugin, "Malaysia road trip toll and fuel cost")
    assert "**1. JalanCost** (`jalancost`)" in text and "/testdrive run jalancost" in text
    assert "Nothing is installed until you approve" in text
    assert "No catalog plugin matched" in cmd(plugin, "zzqx frobnicate")


def test_card_and_help(plugin):
    assert "ABSTRACT_API_KEY ✗ missing" in cmd(plugin, "card abstract-provider")
    assert "not in the plugin catalog" in cmd(plugin, "card nope")
    assert "| `/testdrive history` |" in cmd(plugin, "")


# --- gap detection and nudges ---------------------------------------------------------------------------

def turn(ctx, user, reply, session="s1", platform="cli", failures=()):
    ctx.hooks["pre_llm_call"](session_id=session, user_message=user, platform=platform, is_first_turn=False)
    for name, result in failures:
        ctx.hooks["post_tool_call"](tool_name=name, args={}, result=result, session_id=session, status="error")
    return ctx.hooks["transform_llm_output"](response_text=reply, session_id=session, model="m", platform=platform)


def test_gap_nudges_once_with_top_candidate(plugin):
    plugin.td = None
    cmd(plugin, "culture")  # warms the catalog cache
    out = turn(plugin, GAP_ASK, GAP_REPLY)
    assert out.startswith(GAP_REPLY)
    assert "🧪 test-drive" in out and "`/testdrive run global-business-culture`" in out
    assert out.count("🧪") == 1
    # Cooldown: the next gap in the same session gets no nudge.
    assert turn(plugin, GAP_ASK, GAP_REPLY) is None
    # The gap is remembered for /testdrive with no args.
    assert "Last gap noticed" in cmd(plugin, "")


def test_no_nudge_for_normal_or_policy_replies(plugin):
    cmd(plugin, "culture")
    assert turn(plugin, GAP_ASK, "Here are the gaps: Evaluating 68, Communicating 65.") is None
    assert turn(plugin, "how do I pick a lock", "I can't help with that; it could be illegal.", session="s2") is None


def test_no_nudge_in_cron_or_when_disabled(plugin):
    cmd(plugin, "culture")
    assert turn(plugin, GAP_ASK, GAP_REPLY, session="c", platform="cron") is None
    plugin.config["nudges"] = False
    assert turn(plugin, GAP_ASK, GAP_REPLY, session="d") is None


def test_tool_failure_triggers_nudge(plugin):
    cmd(plugin, "weather")
    out = turn(plugin, "what's the weather forecast in Seattle tomorrow?", "Here is a general idea instead.",
               failures=[("web_search", '{"error": "web_search is not available: no search backend configured"}')])
    assert out and "`/testdrive run weather`" in out


def test_cold_catalog_gives_generic_nudge_and_never_blocks(ctx, env, monkeypatch):
    started = []
    monkeypatch.setattr(core.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: started.append(kw)})())
    _register(ctx, core.CatalogStore(lambda: Path("/nonexistent-td"), fetcher=lambda u: b"{}"), Drives())
    out = turn(ctx, GAP_ASK, GAP_REPLY)
    assert "searches the plugin catalog" in out and started  # background warm requested, reply not delayed


def test_llm_check_confirms_ambiguous_gap(plugin):
    cmd(plugin, "culture")
    plugin.config["llm_check"] = True
    plugin.llm = FakeLLM("yes")
    out = turn(plugin, "culture gap analysis Germany Japan", "Hmm, I can't, sorry.", session="amb")
    assert out and plugin.llm.calls and plugin.llm.calls[0]["max_tokens"] == 3


def test_hook_errors_never_break_the_reply(plugin, monkeypatch):
    monkeypatch.setattr(core.gaps, "detect", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert turn(plugin, GAP_ASK, GAP_REPLY) is None


# --- run: approval paths --------------------------------------------------------------------------------

def test_run_command_asks_and_runs_on_yes(plugin):
    plugin.choices = ["yes"]
    text = cmd(plugin, "run global-business-culture -- culture gap DE vs JP")
    question, choices = plugin.asked[0]
    assert "throwaway Hermes home" in question and "OTARI_API_KEY" in question
    assert [c["value"] for c in choices] == ["yes", "no"]
    call = plugin.drives.calls[0]
    assert call["plugin"] == "global-business-culture" and call["request"] == "culture gap DE vs JP"
    assert call["env"]["OTARI_API_KEY"] == "sk-test" and "HERMES_HOME" not in call["env"]
    assert call["config"]["approvals"]["single_query_mode"] == "deny"
    assert call["run_baseline"] is True  # no failure captured for this request -> auto baseline
    assert "**With global-business-culture:** Evaluating 68" in text
    assert "it called culture_gap_analysis" in text and "deleted ✓" in text
    assert "/testdrive keep global-business-culture" in text
    r = plugin.state.data["drives"][-1]
    assert r["status"] == "ok" and r["plugin_tools_used"] and r["env_passed"] == ["OTARI_API_KEY"]


def test_run_uses_captured_failure_as_before(plugin):
    cmd(plugin, "culture")
    turn(plugin, GAP_ASK, GAP_REPLY)
    plugin.choices = ["yes"]
    text = cmd(plugin, "run global-business-culture")
    assert plugin.drives.calls[0]["request"] == GAP_ASK
    assert plugin.drives.calls[0]["run_baseline"] is False
    assert f"**Without plugin:** {GAP_REPLY[:40]}" in text and "(from this conversation)" in text


def test_run_denied_downloads_nothing(plugin):
    plugin.choices = ["no"]
    text = cmd(plugin, "run jalancost -- KL to Penang fuel")
    assert text.startswith("Cancelled") and not plugin.drives.calls
    assert plugin.state.data["drives"][-1]["status"] == "denied"


def test_run_no_answer_times_out_cleanly(plugin):
    plugin.choices = [None]
    text = cmd(plugin, "run jalancost -- KL to Penang fuel")
    assert "No answer, so nothing was downloaded" in text and "--yes" in text
    assert not plugin.drives.calls


def test_run_explicit_yes_flag(plugin):
    text = cmd(plugin, "run jalancost --yes -- KL to Penang fuel")
    assert not plugin.asked and plugin.drives.calls and "Test drive: JalanCost" in text


def test_run_missing_keys_skips_but_shows_card(plugin):
    text = cmd(plugin, "run abstract-provider -- hello")
    assert "needs ABSTRACT_API_KEY" in text and "skipped (nothing was downloaded)" in text
    assert "ABSTRACT_API_KEY ✗ missing" in text
    assert not plugin.asked and not plugin.drives.calls
    assert plugin.state.data["drives"][-1]["status"] == "skipped_keys"


def test_run_refuses_unknown_removed_and_needs_request(plugin):
    assert "not in the plugin catalog" in cmd(plugin, "run nope -- x")
    assert "What should the sandbox try?" in cmd(plugin, "run jalancost", session_id="fresh")
    assert "cannot test-drive itself" not in cmd(plugin, "run weather --yes -- forecast")


def test_run_tool_handler_after_gate(plugin):
    args = {"plugin": "weather", "request": "forecast for Seattle"}
    assert plugin.hooks["pre_tool_call"](tool_name="test_drive_run", args=args)["action"] == "approve"
    out = json.loads(plugin.tools["test_drive_run"]["handler"](args, session_id="s9"))
    assert out["status"] == "ok" and plugin.drives.calls


def test_oauth_model_needs_sandbox_settings(ctx, store, monkeypatch):
    monkeypatch.setattr(TestDrive, "_user_model", lambda self: ({"provider": "nous", "default": "hermes-4"}, None))
    td = TestDrive(ctx, store=store, drive_fn=Drives())
    prep = td.prepare("jalancost", "KL to Penang")
    assert "OAuth" in prep["error"] and "sandbox_key_env" in prep["error"]


def test_missing_model_key_is_explained(ctx, store, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(TestDrive, "_user_model", lambda self: ({"provider": "openrouter", "default": "x"}, None))
    monkeypatch.setattr(TestDrive, "_registry_keys", staticmethod(lambda p: []))
    monkeypatch.setattr(TestDrive, "_secret", staticmethod(lambda n: None))
    prep = TestDrive(ctx, store=store, drive_fn=Drives()).prepare("jalancost", "x")
    assert "OPENROUTER_API_KEY" in prep["error"]


def test_failed_drive_is_reported_and_receipted(plugin):
    plugin.drives.outcome = sandbox.DriveOutcome(plugin="jalancost", error="install failed (exit 1): boom",
                                                 cleanup_ok=True)
    text = cmd(plugin, "run jalancost --yes -- x")
    assert "not run: install failed (exit 1): boom" in text and "/testdrive keep" not in text
    assert plugin.state.data["drives"][-1]["status"] == "failed"


def test_one_drive_at_a_time(plugin):
    td = plugin.commands["testdrive"]["handler"].__closure__  # noqa: F841 - just ensure closure exists
    prep_holder = {}

    class Slow(Drives):
        def __call__(self, **kw):
            prep_holder["inner"] = cmd(plugin, "run weather --yes -- second")
            return super().__call__(**kw)
    # Re-register with a drive function that tries to start a second drive while running.
    from conftest import FakeCtx  # noqa
    plugin2 = plugin
    _register(plugin2, core.CatalogStore(lambda: Path(plugin.state.data_dir) / "c2",
                                         fetcher=lambda u: (Path(__file__).parent / "fixtures" / "catalog-sample.json").read_bytes()),
              Slow())
    cmd(plugin2, "run jalancost --yes -- first")
    assert "already running" in prep_holder["inner"]


# --- keep / discard / history ----------------------------------------------------------------------------

def test_keep_without_desktop_gives_hermes_install_command(plugin):
    cmd(plugin, "run jalancost --yes -- x")
    text = cmd(plugin, "keep jalancost")
    assert plugin.dispatched[0] == ("manage_catalog", {
        "action": "install", "items": [{"kind": "plugin", "id": "jalancost"}],
        "reason": "You test-drove it with test-drive and chose to keep it."})
    assert "hermes plugins install" in text and "--ref aa3f8f9254de" in text
    assert plugin.state.data["drives"][-1]["decision"] == "keep: install pending"


def test_keep_through_desktop_card(plugin):
    cmd(plugin, "run jalancost --yes -- x")
    plugin.dispatch_result = json.dumps({"rows": [{"id": "jalancost", "status": "installed"}]})
    assert cmd(plugin, "keep jalancost").startswith("Installed `jalancost`")
    assert plugin.state.data["drives"][-1]["decision"] == "kept"


def test_keep_survives_dispatch_exception(plugin):
    plugin.dispatch_result = RuntimeError("no agent")
    assert "hermes plugins install" in cmd(plugin, "keep weather")


def test_discard_and_history(plugin):
    cmd(plugin, "run jalancost --yes -- KL to Penang")
    assert "temporary profile was deleted" in cmd(plugin, "discard jalancost")
    assert "No test drive of `weather`" in cmd(plugin, "discard weather")
    hist = cmd(plugin, "history")
    assert "`jalancost` @ aa3f8f92: ran, did not use its tools" in hist and "discarded" in hist
    keep = json.loads(plugin.tools["test_drive_keep"]["handler"]({"plugin": "jalancost", "decision": "discard"}))
    assert "Discarded" in keep["result"]


def test_history_empty_and_capped(plugin):
    assert "No test drives yet" in cmd(plugin, "history")
    for i in range(core.MAX_RECEIPTS + 5):
        cmd(plugin, f"run jalancost --yes -- trip {i}")
    assert len(plugin.state.data["drives"]) == core.MAX_RECEIPTS


def test_settings_coercion(ctx, store):
    td = TestDrive(ctx, store=store, drive_fn=Drives())
    ctx.config.update(run_timeout="30", nudges="false", install_timeout="bad")
    assert td.setting("run_timeout") == 30 and td.setting("nudges") is False
    assert td.setting("install_timeout") == 240


def test_honest_verdict_when_plugin_tools_unused(plugin):
    res = sandbox.RunResult(True, "Total RM95.23, RM31.74 each.", tools_called=["execute_code"], seconds=9.0, evidence_available=True)
    base = sandbox.RunResult(True, "About RM95 in total.", tools_called=["terminal"], seconds=8.0, evidence_available=True)
    plugin.drives.outcome = sandbox.DriveOutcome(plugin="jalancost", installed=True, baseline=base,
                                                 with_plugin=res, cleanup_ok=True)
    text = cmd(plugin, "run jalancost --yes -- KL to Penang 350 km")
    assert "**Without plugin:** About RM95 in total." in text and "tools: terminal" in text
    assert "did not call any of jalancost's tools (estimate_road_trip_cost)" in text
    assert "did not earn its keep" in text
    assert plugin.state.data["drives"][-1]["plugin_tools_used"] is False


def test_cleanup_problem_is_surfaced(plugin):
    res = sandbox.RunResult(True, "ok", tools_called=["estimate_road_trip_cost"], evidence_available=True)
    plugin.drives.outcome = sandbox.DriveOutcome(plugin="jalancost", installed=True, with_plugin=res,
                                                 cleanup_ok=False, cleanup_notes=["could not delete /tmp/x"])
    text = cmd(plugin, "run jalancost --yes -- x")
    assert "cleanup incomplete ⚠ could not delete /tmp/x" in text
    assert plugin.state.data["drives"][-1]["cleanup_ok"] is False


def test_long_gap_need_is_trimmed_in_help(plugin):
    cmd(plugin, "culture")
    turn(plugin, "culture gap analysis " + "Germany Japan " * 30, GAP_REPLY, session="long")
    help_text = cmd(plugin, "", session_id="long")
    assert "Last gap noticed: “culture gap analysis" in help_text and "…”" in help_text


def test_keep_suggests_current_exact_catalog_pin(plugin):
    text = cmd(plugin, "keep global-business-culture")
    assert ("hermes plugins install https://github.com/pswong661-eng/hermes-global-business-culture "
            "--ref 49218a35f670ceae6f79dc07d40536b19da23cc6") in text


def test_discard_does_not_claim_cleanup_succeeded(plugin):
    plugin.drives.outcome = sandbox.DriveOutcome(plugin="jalancost", cleanup_ok=False)
    cmd(plugin, "run jalancost --yes -- x")
    assert "Check the drive report" in cmd(plugin, "discard jalancost")


def test_long_request_is_kept_whole_for_the_sandbox(plugin):
    cmd(plugin, "culture")
    long_ask = "Compute the exact culture gap analysis between Germany and Japan. " + "Context. " * 60
    turn(plugin, long_ask, GAP_REPLY, session="lr")
    plugin.choices = ["yes"]
    cmd(plugin, "run global-business-culture", session_id="lr")
    assert plugin.drives.calls[-1]["request"] == long_ask.strip()
    assert plugin.drives.calls[-1]["run_baseline"] is False


def test_other_session_gap_is_never_reused(plugin):
    cmd(plugin, "culture")
    turn(plugin, GAP_ASK, GAP_REPLY, session="a")
    assert "Last gap noticed" not in cmd(plugin, "", session_id="b")
    assert "What should the sandbox try?" in cmd(plugin, "run global-business-culture", session_id="b")
    assert not plugin.drives.calls


def test_unknown_evidence_is_not_reported_as_no_tool_use(plugin):
    result = sandbox.RunResult(True, "reply", evidence_available=False)
    plugin.drives.outcome = sandbox.DriveOutcome(plugin="jalancost", installed=True,
                                                 with_plugin=result, cleanup_ok=True)
    text = cmd(plugin, "run jalancost --yes -- trip")
    assert "tool-use evidence is unavailable" in text
    assert "did not call" not in text and "did not earn" not in text
    assert plugin.state.data["drives"][-1]["plugin_tools_used"] is None
    assert "tool-use evidence unavailable" in cmd(plugin, "history")


def test_approval_prompt_shows_request_excerpt(plugin):
    hook = plugin.hooks["pre_tool_call"]
    req = "Compare\n\nGermany and Japan " + "x" * 400
    msg = hook(tool_name="test_drive_run", args={"plugin": "jalancost", "request": req})["message"]
    assert 'Request it will run: "Compare Germany and Japan' in msg and "\n" not in msg.split("Request it will run:")[1]
    assert len(msg.split("Request it will run: ")[1]) <= 203
    assert "No request given" in hook(tool_name="test_drive_run", args={"plugin": "jalancost"})["message"]


def test_slash_approval_message_shows_request(ctx, store, env):
    td = TestDrive(ctx, store=store, drive_fn=Drives())
    prep = td.prepare("jalancost", "KL to\nPenang by bus")
    assert 'Request: "KL to Penang by bus"' in td.approval_message(prep)


def test_literal_only_key_is_refused_and_never_copied(ctx, store, monkeypatch):
    monkeypatch.setattr(TestDrive, "_user_model", lambda self: (
        {"provider": "custom", "default": "m", "base_url": "https://x/v1", "api_key": "sk-LITERAL"},
        ("custom", {"api_key": "sk-LITERAL2"})))
    monkeypatch.setattr(TestDrive, "_registry_keys", staticmethod(lambda p: []))
    monkeypatch.setattr(TestDrive, "_secret", staticmethod(lambda n: None))
    drives = Drives()
    prep = TestDrive(ctx, store=store, drive_fn=drives).prepare("jalancost", "x")
    assert "literally" in prep["error"] and "sandbox_key_env" in prep["error"]
    assert "LITERAL" not in str(prep.get("model")) and not drives.calls


def test_literal_key_dropped_when_env_key_available(ctx, store, monkeypatch):
    monkeypatch.setenv("WORK_KEY", "sk-env")
    monkeypatch.setattr(TestDrive, "_user_model", lambda self: (
        {"provider": "custom", "default": "m", "base_url": "https://x/v1", "api_key": "sk-LITERAL",
         "key_env": "WORK_KEY"}, None))
    monkeypatch.setattr(TestDrive, "_secret", staticmethod(lambda n: __import__("os").environ.get(n)))
    prep = TestDrive(ctx, store=store, drive_fn=Drives()).prepare("jalancost", "x")
    assert not prep.get("error") and "api_key" not in prep["model"] and prep["child_env"]["WORK_KEY"] == "sk-env"
