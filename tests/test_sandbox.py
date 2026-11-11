import json
import os
import stat
import subprocess
import sys

import pytest

from conftest import session_row
from test_drive import sandbox
from test_drive.sandbox import Sandbox, build_config, drive, dump_yaml, model_key_env

MODEL = {"default": "deepseek/deepseek-v4-flash", "provider": "custom",
         "base_url": "https://api.otari.ai/api/v1", "key_env": "OTARI_API_KEY"}
ENV = {"PATH": "/usr/bin", "OTARI_API_KEY": "sk-test"}


def export_out(*rows):
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def ok_script(with_tools=("culture_gap_analysis",), baseline=False):
    rows = ([session_row(started=1.0)] if baseline else []) + [session_row(with_tools, started=2.0, cost=0.0004)]
    return {
        "chat -q": {"rc": 0, "out": "Ranked gaps: Evaluating 68…\n\nsession_id: local-abc\n"},
        "plugins install": {"rc": 0, "out": "✓ Plugin global-business-culture enabled."},
        "sessions export": {"rc": 0, "out": export_out(*rows)},
        "gateway stop": {"rc": 0, "out": "✓ Stopped gateway for this profile"},
    }


@pytest.fixture(autouse=True)
def no_real_signals(monkeypatch):
    killed = []
    monkeypatch.setattr(sandbox.os, "killpg", lambda pid, sig: killed.append((pid, sig)), raising=False)
    return killed


def test_full_drive_lifecycle(tmp_path, fake_hermes):
    popen = fake_hermes(ok_script())
    out = drive(plugin="global-business-culture", request="culture gap DE vs JP", hermes=["hermes"], env=ENV,
                config=build_config(MODEL), run_baseline=False, install_timeout=30, run_timeout=20,
                popen=popen, tmp_root=str(tmp_path))
    assert out.installed and out.cleanup_ok and not out.error
    assert out.with_plugin.ok and out.with_plugin.output == "Ranked gaps: Evaluating 68…"
    assert out.with_plugin.tools_called == ["culture_gap_analysis"] and out.with_plugin.cost_usd == 0.0004
    steps = [" ".join(p.argv[1:3]) for p in popen.log]
    assert steps == ["plugins install", "chat -q", "sessions export", "gateway stop"]
    install = popen.log[0]
    assert install.argv == ["hermes", "plugins", "install", "global-business-culture", "--enable", "--yes-deps"]
    chat = popen.log[1]
    assert chat.argv == ["hermes", "chat", "-q", "culture gap DE vs JP", "-Q"]  # -q: single-query, not -z
    assert chat.kw["stdin"] is subprocess.DEVNULL
    assert chat.kw["env"]["HERMES_HOME"] == out.sandbox and chat.kw["cwd"] == out.sandbox
    if os.name == "posix":
        assert chat.kw["start_new_session"] is True
    assert not os.path.exists(out.sandbox)  # destroyed


def test_sandbox_config_is_minimal_and_private(tmp_path):
    box = Sandbox(hermes=["hermes"], env=ENV, config=build_config(MODEL, ("otari", {"base_url": "x"})),
                  tmp_root=str(tmp_path))
    home = box.create()
    try:
        assert home.name.startswith("hermes-testdrive-")
        assert stat.S_IMODE(home.stat().st_mode) == 0o700
        cfg = home / "config.yaml"
        assert stat.S_IMODE(cfg.stat().st_mode) == 0o600
        text = cfg.read_text()
        assert 'single_query_mode: "deny"' in text and 'mode: "manual"' in text
        assert 'key_env: "OTARI_API_KEY"' in text and "sk-test" not in text
        assert sorted(p.name for p in home.iterdir()) == ["config.yaml"]  # no memory, sessions, .env, auth.json
    finally:
        box.close()
    assert not home.exists()


def test_child_env_has_no_parent_hermes_vars(tmp_path):
    box = Sandbox(hermes=["hermes"], env={"PATH": "/bin"}, config={}, tmp_root=str(tmp_path))
    box.create()
    try:
        env = box.env
        assert set(env) == {"PATH", "HERMES_HOME", "PYTHONDONTWRITEBYTECODE", "NO_COLOR"}
    finally:
        box.close()


def test_timeout_kills_process_group_and_still_cleans_up(tmp_path, fake_hermes, no_real_signals):
    script = ok_script()
    script["chat -q"] = {"hang": True}
    popen = fake_hermes(script)
    out = drive(plugin="p", request="r", hermes=["hermes"], env=ENV, config={}, run_baseline=False,
                install_timeout=30, run_timeout=5, popen=popen, tmp_root=str(tmp_path))
    assert out.installed and out.with_plugin.timed_out and not out.with_plugin.ok
    assert "timed out after 5s" in out.with_plugin.error
    if os.name == "posix":
        assert no_real_signals and no_real_signals[0][0] == 999999
    assert out.cleanup_ok and not os.path.exists(out.sandbox)


def test_install_failure_skips_run_and_cleans_up(tmp_path, fake_hermes):
    script = ok_script()
    script["plugins install"] = {"rc": 1, "err": "Plugin requires Hermes >=0.99"}
    popen = fake_hermes(script)
    out = drive(plugin="p", request="r", hermes=["hermes"], env=ENV, config={}, run_baseline=False,
                install_timeout=30, run_timeout=5, popen=popen, tmp_root=str(tmp_path))
    assert not out.installed and out.with_plugin is None
    assert "install failed (exit 1)" in out.error and "0.99" in out.error
    assert [" ".join(p.argv[1:3]) for p in popen.log] == ["plugins install", "gateway stop"]
    assert out.cleanup_ok and not os.path.exists(out.sandbox)


def test_install_timeout(tmp_path, fake_hermes):
    script = ok_script()
    script["plugins install"] = {"hang": True}
    out = drive(plugin="p", request="r", hermes=["hermes"], env=ENV, config={}, run_baseline=False,
                install_timeout=7, run_timeout=5, popen=fake_hermes(script), tmp_root=str(tmp_path))
    assert out.error == "install timed out after 7s" and out.cleanup_ok


def test_missing_hermes_binary_is_reported_and_cleaned(tmp_path):
    def popen(argv, **kw):
        raise FileNotFoundError("hermes")
    out = drive(plugin="p", request="r", hermes=["hermes"], env=ENV, config={}, run_baseline=False,
                install_timeout=7, run_timeout=5, popen=popen, tmp_root=str(tmp_path))
    assert "FileNotFoundError" in out.error
    assert out.sandbox and not os.path.exists(out.sandbox)


def test_baseline_then_plugin_sessions_map_in_order(tmp_path, fake_hermes):
    popen = fake_hermes(ok_script(baseline=True))
    out = drive(plugin="p", request="r", hermes=["hermes"], env=ENV, config={}, run_baseline=True,
                install_timeout=30, run_timeout=20, popen=popen, tmp_root=str(tmp_path))
    assert [" ".join(p.argv[1:3]) for p in popen.log][:3] == ["chat -q", "plugins install", "chat -q"]
    assert out.baseline.tools_called == [] and out.with_plugin.tools_called == ["culture_gap_analysis"]


def test_reaps_leftover_gateway_only_for_its_own_home(tmp_path, monkeypatch):
    box = Sandbox(hermes=["hermes"], env=ENV, config={}, tmp_root=str(tmp_path),
                  popen=lambda argv, **kw: type("P", (), {"pid": 1, "returncode": 0,
                                                          "communicate": lambda self, timeout=None: ("", "")})())
    home = box.create()
    sent = []
    alive = {4242: True}
    monkeypatch.setattr(sandbox, "_alive", lambda pid: alive.get(pid, False))

    def kill(pid, sig):
        sent.append((pid, sig))
        alive[pid] = False
    monkeypatch.setattr(sandbox.os, "kill", kill)
    (home / "gateway.pid").write_text(json.dumps({"pid": 4242, "hermes_home": "/somewhere/else"}))
    assert box._reap_gateway() and sent == []          # not ours: never signalled
    (home / "gateway.pid").write_text(json.dumps({"pid": 4242, "hermes_home": str(home)}))
    assert box.close() and sent and sent[0][0] == 4242
    assert not home.exists() and any("still alive" in n for n in box.notes)


def test_model_key_env_resolution():
    assert model_key_env(MODEL) == ["OTARI_API_KEY"]
    assert model_key_env({"provider": "openrouter"}) == ["OPENROUTER_API_KEY"]
    assert model_key_env({"provider": "acme"}, registry=lambda p: ["ACME_KEY"]) == ["ACME_KEY"]
    assert model_key_env({"provider": "work"}, {"api_key": "${WORK_LLM_KEY}"}) == ["WORK_LLM_KEY"]
    assert model_key_env({"provider": "x"}, registry=lambda p: (_ for _ in ()).throw(RuntimeError())) == []


def test_session_tools_reads_transcript_not_fingerprint():
    tools, cost = sandbox.session_tools(session_row(["a", "b", "a"], cost=0.0))
    assert tools == ["a", "b"] and cost is None


def test_dump_yaml_round_trips():
    yaml = pytest.importorskip("ruamel.yaml")
    cfg = build_config(MODEL, ("otari", {"base_url": "https://x", "models": ["a", "b"], "extra": {}}))
    assert yaml.YAML(typ="safe").load(dump_yaml(cfg)) == cfg


def test_hermes_argv(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda n: None)
    assert sandbox.hermes_argv() == [sys.executable, "-m", "hermes_cli.main"]
    monkeypatch.setattr(sandbox.shutil, "which", lambda n: "/usr/local/bin/hermes")
    assert sandbox.hermes_argv() == ["/usr/local/bin/hermes"]
    assert sandbox.hermes_argv("uvx hermes --flag") == ["uvx", "hermes", "--flag"]


def test_clean_output_drops_session_line():
    assert sandbox.clean_output("hello\n\nsession_id: local-1\n") == "hello"


def test_older_hermes_falls_back_to_pinned_url(tmp_path, fake_hermes):
    script = ok_script()
    calls = []

    def install(proc):
        calls.append(proc.argv)
        if len(calls) == 1:
            return {"rc": 1, "err": "Error: 'gbc' is not in the Hermes plugin catalog and is not \na Git URL"}
        return {"rc": 0}
    script["plugins install"] = install
    sha = "49218a35f670ceae6f79dc07d40536b19da23cc6"
    out = drive(plugin="gbc", request="r", hermes=["hermes"], env=ENV, config={}, run_baseline=False,
                install_timeout=30, run_timeout=20, repo="https://github.com/o/gbc", sha=sha,
                popen=fake_hermes(script), tmp_root=str(tmp_path))
    assert out.installed and out.with_plugin.ok
    assert calls[1] == ["hermes", "plugins", "install", "https://github.com/o/gbc", "--ref", sha, "--enable", "--yes-deps"]
    assert any("pinned URL" in n for n in out.cleanup_notes) and out.cleanup_ok


def test_no_url_fallback_for_subdir_or_other_errors(tmp_path, fake_hermes):
    script = ok_script()
    script["plugins install"] = {"rc": 1, "err": "'x' is not in the Hermes plugin catalog"}
    popen = fake_hermes(script)
    out = drive(plugin="x", request="r", hermes=["hermes"], env=ENV, config={}, run_baseline=False,
                install_timeout=30, run_timeout=20, repo="https://github.com/o/x", sha="a" * 40, subdir="plugins/x",
                popen=popen, tmp_root=str(tmp_path))
    assert not out.installed and sum(1 for p in popen.log if p.argv[1] == "plugins") == 1
