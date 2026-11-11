"""Load the plugin directory as the package ``test_drive`` (its folder name has a hyphen) and give
tests a fake PluginContext, a fake catalog fetcher and a fake ``hermes`` CLI. Nothing here touches
the network or starts a real process."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != PLUGIN_DIR]


def _load_package():
    if "test_drive" in sys.modules:
        return sys.modules["test_drive"]
    spec = importlib.util.spec_from_file_location(
        "test_drive", PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_drive"] = module
    spec.loader.exec_module(module)
    return module


_load_package()

from test_drive.catalog import CatalogStore  # noqa: E402


def sample_bytes() -> bytes:
    return (FIXTURES / "catalog-sample.json").read_bytes()


class FakeState:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data: dict[str, Any] = {}

    def get(self, key, default=None):
        return json.loads(json.dumps(self.data[key])) if key in self.data else default

    def set(self, key, value):
        self.data[key] = json.loads(json.dumps(value))


class FakeLLM:
    def __init__(self, reply: str = "yes"):
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kw):
        self.calls.append(kw)

        class R:
            text = self.reply
        return R()


class FakeCtx:
    def __init__(self, tmp: Path):
        self.config: dict[str, Any] = {}
        self.state = FakeState(tmp / "plugin-data")
        self.tools: dict[str, dict] = {}
        self.hooks: dict[str, Callable] = {}
        self.commands: dict[str, dict] = {}
        self.choices: list[Optional[str]] = []
        self.asked: list[tuple[str, list]] = []
        self.dispatched: list[tuple[str, dict]] = []
        self.dispatch_result: Any = json.dumps({"error": "manage_catalog is available only in the Hermes desktop app"})
        self.llm = None

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kw):
        self.tools[kw["name"]] = kw

    def register_hook(self, name, fn, **kw):
        self.hooks[name] = fn

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = {"handler": handler, "description": description, "args_hint": args_hint}

    def ask_choice(self, question, choices):
        self.asked.append((question, choices))
        return self.choices.pop(0) if self.choices else None

    def dispatch_tool(self, name, args, **kw):
        self.dispatched.append((name, args))
        if isinstance(self.dispatch_result, Exception):
            raise self.dispatch_result
        return self.dispatch_result


@pytest.fixture
def ctx(tmp_path):
    return FakeCtx(tmp_path)


@pytest.fixture
def store(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return sample_bytes()

    s = CatalogStore(lambda: tmp_path / "cache", fetcher=fetch)
    s.calls = calls
    return s


class FakeProc:
    """Stands in for subprocess.Popen; behaviour comes from a script keyed by the hermes subcommand."""

    def __init__(self, argv, script, log, **kw):
        self.argv = argv
        self.kw = kw
        self.pid = 999999
        self.returncode = None
        log.append(self)
        key = " ".join(argv[1:3])
        if len(argv) > 2 and argv[1] == "chat" and argv[2].startswith("-q"):
            key = "chat -q"
        self.spec = script.get(key) or script.get(argv[1]) or {"rc": 0}
        if callable(self.spec):
            self.spec = self.spec(self)

    def communicate(self, timeout=None):
        if self.spec.get("hang") and self.returncode is None:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        self.returncode = self.spec.get("rc", 0) if self.returncode is None else self.returncode
        return self.spec.get("out", ""), self.spec.get("err", "")

    def kill(self):
        self.returncode = -9


@pytest.fixture
def fake_hermes():
    def make(script: dict):
        log: list[FakeProc] = []

        def popen(argv, **kw):
            return FakeProc(argv, script, log, **kw)
        popen.log = log
        return popen
    return make


def session_row(tools=(), started=1.0, cost=None):
    msgs = [{"role": "user", "content": "q"}]
    for t in tools:
        msgs += [{"role": "assistant", "content": ""}, {"role": "tool", "tool_name": t, "content": "{}"}]
    msgs.append({"role": "assistant", "content": "answer"})
    return {"id": f"s{started}", "started_at": started, "message_count": len(msgs), "messages": msgs,
            "tool_names": "c08d11b2839678b07e40b762ee772ded", "estimated_cost_usd": cost}
