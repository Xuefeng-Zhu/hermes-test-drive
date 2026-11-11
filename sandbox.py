"""The throwaway Hermes: a fresh temporary HERMES_HOME, driven only through the public CLI.

Lifecycle (every step bounded by a timeout, cleanup always runs):

1. ``mkdtemp`` a 0700 directory and write a minimal ``config.yaml``: the user's model block (and
   the named provider it points at), plus approvals that DENY dangerous commands in single-query
   mode. Nothing else is copied: no memory, sessions, skills, .env, auth.json or other plugins.
2. Optional baseline: ``hermes chat -q<request> -Q`` before the plugin exists.
3. ``hermes plugins install <repo>[#subdir] --ref <sha> --enable --yes-deps`` installs the approved
   exact source with Hermes's scanner and dependency admission inside the sandbox home.
4. ``hermes chat -q<request> -Q`` with the plugin. ``chat -q`` (not ``-z``) is single-query mode,
   so a dangerous command in the child is refused, never auto-approved.
5. ``hermes sessions export --format jsonl -`` to read which tools each run called.
6. ``hermes gateway stop`` for the per-home runtime the CLI started, a pid-file check that it is
   gone, then ``rmtree``.

The child gets a least-privilege environment: system basics (PATH, locale, proxies, CA bundles)
plus only the env vars the model route needs and the candidate's declared ``requires_env`` keys.
No HERMES_* variable from the parent leaks in (so the child never attaches to the user's gateway).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

PASS_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "USERNAME", "SHELL", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ",
    "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "WINDIR", "APPDATA",
    "LOCALAPPDATA", "PROGRAMDATA", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "XDG_CACHE_HOME", "XDG_RUNTIME_DIR",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "no_proxy", "all_proxy",
    "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "PIP_INDEX_URL",
    "GIT_SSL_CAINFO",
)
# Fallback when the provider registry cannot be asked. Model route -> key env var(s).
PROVIDER_KEYS = {
    "openrouter": ["OPENROUTER_API_KEY"], "anthropic": ["ANTHROPIC_API_KEY"], "openai": ["OPENAI_API_KEY"],
    "openai-api": ["OPENAI_API_KEY", "OPENAI_BASE_URL"], "custom": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "deepseek": ["DEEPSEEK_API_KEY"], "xai": ["XAI_API_KEY"],
    "nvidia": ["NVIDIA_API_KEY"], "mistral": ["MISTRAL_API_KEY"], "groq": ["GROQ_API_KEY"],
    "together": ["TOGETHER_API_KEY"], "huggingface": ["HF_TOKEN"], "ollama-cloud": ["OLLAMA_API_KEY"],
    "zai": ["GLM_API_KEY", "ZAI_API_KEY"], "kimi-coding": ["KIMI_API_KEY"], "minimax": ["MINIMAX_API_KEY"],
}
OAUTH_ONLY = {"nous", "openai-codex", "qwen-oauth", "minimax-oauth", "xai-oauth", "copilot"}
SANDBOX_PREFIX = "hermes-testdrive-"


class SandboxError(RuntimeError):
    pass


@dataclass
class Step:
    name: str
    argv: list[str]
    rc: Optional[int]
    seconds: float
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    def tail(self, n: int = 600) -> str:
        text = (self.stderr or self.stdout or "").strip()
        return text[-n:]


@dataclass
class RunResult:
    ok: bool
    output: str = ""
    tools_called: list[str] = field(default_factory=list)
    seconds: float = 0.0
    timed_out: bool = False
    error: str = ""
    cost_usd: Optional[float] = None
    session_id: str = ""
    evidence_available: bool = False


@dataclass
class DriveOutcome:
    plugin: str
    sandbox: str = ""
    installed: bool = False
    baseline: Optional[RunResult] = None
    with_plugin: Optional[RunResult] = None
    error: str = ""
    cleanup_ok: bool = False
    cleanup_notes: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    env_passed: list[str] = field(default_factory=list)


def hermes_argv(configured: str = "") -> list[str]:
    """How to start Hermes: an explicit setting, else ``hermes`` on PATH, else this interpreter."""
    if configured:
        import shlex
        return shlex.split(configured)
    found = shutil.which("hermes")
    return [found] if found else [sys.executable, "-m", "hermes_cli.main"]


def model_key_env(model: dict[str, Any], provider_entry: Optional[dict[str, Any]] = None,
                  registry: Optional[Callable[[str], list[str]]] = None) -> list[str]:
    names: list[str] = []
    for block in (model, provider_entry or {}):
        for key in ("key_env", "api_key_env"):
            v = block.get(key)
            if isinstance(v, str) and v.strip():
                names.append(v.strip())
        ak = block.get("api_key")
        if isinstance(ak, str) and ak.startswith("${") and ak.endswith("}"):
            names.append(ak[2:-1].split(":-")[0])
    provider = str(model.get("provider") or "").strip().lower()
    if provider and not names:
        found: list[str] = []
        if registry is not None:
            try:
                found = list(registry(provider) or [])
            except Exception:
                found = []
        names.extend(found or PROVIDER_KEYS.get(provider, []))
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


_CRED_WORDS = ("key", "token", "secret", "password", "passwd", "auth", "credential", "cookie", "bearer")
_CRED_SAFE_SUFFIXES = ("_env", "_var", "_type", "_mode", "_url", "_header_name")


def _is_env_ref(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("${") and value.strip().endswith("}")


def _is_cred_field(name: str) -> bool:
    n = str(name).lower()
    if n.endswith(_CRED_SAFE_SUFFIXES) or n in ("key_env", "api_key_env"):
        return False
    return any(w in n for w in _CRED_WORDS)


def strip_literal_credentials(block: Any) -> tuple[Any, bool]:
    """Return a copy of a config block with every literal credential removed, and whether one was.

    A field counts as a credential when its name looks like one (api_key, token, secret, password,
    Authorization header, ...). Env-var references (``${VAR}``) and ``*_env`` pointers are kept: the
    child resolves them from the env test-drive passes, so no secret value is written to disk.
    """
    if isinstance(block, dict):
        out: dict[str, Any] = {}
        found = False
        for k, v in block.items():
            if _is_cred_field(k) and not isinstance(v, (dict, list)):
                if v in (None, "") or _is_env_ref(v):
                    out[k] = v
                else:
                    found = True
                continue
            clean, f = strip_literal_credentials(v)
            found = found or f
            out[k] = clean
        return out, found
    if isinstance(block, list):
        items, found = [], False
        for v in block:
            clean, f = strip_literal_credentials(v)
            items.append(clean)
            found = found or f
        return items, found
    return block, False


def build_config(model: dict[str, Any], provider_entry: Optional[tuple[str, dict[str, Any]]] = None) -> dict[str, Any]:
    # Defense in depth: never write a literal credential into the sandbox, where the plugin under test can read it.
    model, _ = strip_literal_credentials(dict(model))
    if provider_entry:
        provider_entry = (provider_entry[0], strip_literal_credentials(dict(provider_entry[1]))[0])
    cfg: dict[str, Any] = {
        "model": dict(model),
        # Single-query (-q) children refuse dangerous commands instead of auto-approving them.
        "approvals": {"mode": "manual", "single_query_mode": "deny", "cron_mode": "deny", "unattended_mode": "deny"},
    }
    if provider_entry:
        cfg["providers"] = {provider_entry[0]: dict(provider_entry[1])}
    return cfg


def dump_yaml(data: Any, indent: int = 0) -> str:
    """Tiny YAML writer for the sandbox config (plain dicts/lists/scalars), stdlib only."""
    pad = "  " * indent
    out = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:")
                out.append(dump_yaml(v, indent + 1))
            else:
                out.append(f"{pad}{k}: {_scalar(v)}")
    elif isinstance(data, list):
        for v in data:
            if isinstance(v, (dict, list)) and v:
                body = dump_yaml(v, indent + 1).lstrip()
                out.append(f"{pad}- {body}")
            else:
                out.append(f"{pad}- {_scalar(v)}")
    return "\n".join(out)


def _scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (dict, list)):
        return "{}" if isinstance(v, dict) else "[]"
    return json.dumps(str(v))


def clean_output(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if not ln.startswith("session_id:")]
    return "\n".join(lines).strip()


Runner = Callable[..., subprocess.CompletedProcess]


class Sandbox:
    """One throwaway Hermes home. Use as a context manager; ``close`` always cleans up."""

    def __init__(self, *, hermes: list[str], env: dict[str, str], config: dict[str, Any],
                 popen: Callable[..., Any] = subprocess.Popen, tmp_root: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.hermes = hermes
        self.base_env = env
        self.config = config
        self._popen = popen
        self._tmp_root = tmp_root
        self._clock = clock
        self.home: Optional[Path] = None
        self.steps: list[Step] = []
        self.notes: list[str] = []

    # lifecycle -------------------------------------------------------------------------------
    def __enter__(self) -> "Sandbox":
        self.create()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def create(self) -> Path:
        self.home = Path(tempfile.mkdtemp(prefix=SANDBOX_PREFIX, dir=self._tmp_root))
        os.chmod(self.home, 0o700)
        cfg = self.home / "config.yaml"
        fd = os.open(cfg, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("# test-drive throwaway profile; deleted when the drive ends\n")
            fh.write(dump_yaml(self.config) + "\n")
        return self.home

    @property
    def env(self) -> dict[str, str]:
        env = dict(self.base_env)
        env["HERMES_HOME"] = str(self.home)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.setdefault("NO_COLOR", "1")
        return env

    def run(self, name: str, args: list[str], timeout: float) -> Step:
        if self.home is None:
            raise SandboxError("sandbox not created")
        argv = [*self.hermes, *args]
        start = self._clock()
        kwargs: dict[str, Any] = dict(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      env=self.env, cwd=str(self.home), text=True)
        if os.name == "posix":
            kwargs["start_new_session"] = True
        proc = self._popen(argv, **kwargs)
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(proc)
            try:
                out, err = proc.communicate(timeout=10)
            except Exception:
                out, err = "", ""
        step = Step(name, argv, None if timed_out else proc.returncode, round(self._clock() - start, 2),
                    out or "", err or "", timed_out)
        self.steps.append(step)
        return step

    @staticmethod
    def _kill(proc: Any) -> None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    def close(self) -> bool:
        """Stop the sandbox's runtime and delete the directory. Returns True when nothing is left."""
        if self.home is None:
            return True
        ok = True
        if self.home.exists():
            try:
                self.run("gateway stop", ["gateway", "stop"], timeout=30)
            except Exception as exc:  # pragma: no cover - defensive
                self.notes.append(f"gateway stop failed: {exc}")
            ok = self._reap_gateway() and ok
            shutil.rmtree(self.home, ignore_errors=True)
            if self.home.exists():
                ok = False
                self.notes.append(f"could not delete {self.home}")
        self.home_removed = not self.home.exists()
        return ok

    def _reap_gateway(self) -> bool:
        pid_file = self.home / "gateway.pid" if self.home else None
        if not pid_file or not pid_file.exists():
            return True
        try:
            info = json.loads(pid_file.read_text(encoding="utf-8") or "{}")
        except (OSError, ValueError):
            return True
        pid = info.get("pid") if isinstance(info, dict) else None
        home = str(info.get("hermes_home") or "") if isinstance(info, dict) else ""
        if not isinstance(pid, int) or pid <= 1 or (home and Path(home).resolve() != self.home.resolve()):
            return True
        if not _alive(pid):
            return True
        self.notes.append(f"gateway pid {pid} still alive after stop; sending SIGTERM")
        for sig, wait in ((signal.SIGTERM, 3.0), (getattr(signal, "SIGKILL", signal.SIGTERM), 2.0)):
            try:
                os.kill(pid, sig)
            except OSError:
                return True
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if not _alive(pid):
                    return True
                time.sleep(0.1)
        return not _alive(pid)

    # high level --------------------------------------------------------------------------------
    def chat(self, request: str, timeout: float) -> RunResult:
        step = self.run("chat -q", ["chat", f"-q{request}", "-Q"], timeout)
        # -Q prints the session ID to stderr, outside the model's response.
        ids = re.findall(r"(?m)^session_id: ([^\s]+)\s*$", step.stderr)
        session_id = ids[0] if len(ids) == 1 else ""
        if step.timed_out:
            return RunResult(False, clean_output(step.stdout), seconds=step.seconds, timed_out=True,
                             error=f"timed out after {timeout:.0f}s", session_id=session_id)
        if step.rc != 0:
            return RunResult(False, clean_output(step.stdout), seconds=step.seconds,
                             error=f"hermes exited {step.rc}: {step.tail(300)}", session_id=session_id)
        return RunResult(True, clean_output(step.stdout), seconds=step.seconds, session_id=session_id)

    def install(self, name: str, timeout: float) -> Step:
        return self.run("plugins install", ["plugins", "install", name, "--enable", "--yes-deps"], timeout)

    def install_pinned(self, repo: str, sha: str, subdir: str, timeout: float) -> Step:
        identifier = f"{repo}#{subdir}" if subdir else repo
        return self.run("plugins install", ["plugins", "install", identifier, "--ref", sha, "--enable", "--yes-deps"], timeout)

    def sessions(self, timeout: float = 60) -> list[dict[str, Any]]:
        step = self.run("sessions export", ["sessions", "export", "--format", "jsonl", "--yes", "-"], timeout)
        rows = []
        for line in (step.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        rows.sort(key=lambda r: r.get("started_at") or 0)
        return rows


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def session_tools(row: dict[str, Any]) -> tuple[list[str], Optional[float]]:
    """Tool names one exported session called, read from its transcript (``role: tool`` rows carry
    ``tool_name``; ``tool_names`` on the row is only a fingerprint), plus its cost when priced."""
    tools: list[str] = []
    for m in row.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "tool" and isinstance(m.get("tool_name"), str) and m["tool_name"]:
            tools.append(m["tool_name"])
    cost = row.get("actual_cost_usd") if row.get("actual_cost_usd") is not None else row.get("estimated_cost_usd")
    cost = float(cost) if isinstance(cost, (int, float)) and cost > 0 else None
    return list(dict.fromkeys(tools)), cost


def transcript_complete(row: dict[str, Any]) -> bool:
    """Do not infer tool non-use from an incomplete or rotated transcript."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or row.get("parent_session_id"):
        return False
    if any(not isinstance(m, dict) or not isinstance(m.get("role"), str) for m in messages):
        return False
    roles = {m["role"] for m in messages}
    if not {"user", "assistant"}.issubset(roles):
        return False
    count = row.get("message_count")
    if not isinstance(count, int) or isinstance(count, bool) or count != len(messages):
        return False
    if any(m["role"] == "tool" and (not isinstance(m.get("tool_name"), str) or not m["tool_name"])
           for m in messages):
        return False
    tool_count = row.get("tool_call_count")
    if (not isinstance(tool_count, int) or isinstance(tool_count, bool)
            or tool_count != sum(m["role"] == "tool" for m in messages)):
        return False
    return True


def valid_source(repo: str, sha: str, subdir: str) -> bool:
    """Refuse ambiguous source identifiers before starting a baseline or installer."""
    try:
        url = urlsplit(repo)
        if (not repo.startswith("https://") or url.scheme != "https" or not url.hostname
                or url.username is not None or url.password is not None
                or url.query or url.fragment or not url.path or any(c.isspace() for c in repo)):
            return False
        # Hermes treats GitHub browser paths and `.git/<path>` specially as implicit subdirs.
        # Do not let those override the separate subdir that the user reviewed.
        if url.hostname.lower() == "github.com" and len(url.path.strip("/").split("/")) != 2:
            return False
        if ".git/" in url.path and url.path.partition(".git/")[2].strip("/"):
            return False
        if url.port is not None and not 1 <= url.port <= 65535:
            return False
    except ValueError:
        return False
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        return False
    return not subdir or (bool(re.fullmatch(r"[A-Za-z0-9._/-]+", subdir))
                          and all(p not in ("", ".", "..") for p in subdir.split("/")))


def drive(*, plugin: str, request: str, hermes: list[str], env: dict[str, str], config: dict[str, Any],
          run_baseline: bool, install_timeout: float, run_timeout: float, repo: str = "", sha: str = "",
          subdir: str = "", popen: Callable[..., Any] = subprocess.Popen,
          tmp_root: Optional[str] = None) -> DriveOutcome:
    """Full test drive in one sandbox. Never raises for subprocess trouble; reports it instead."""
    outcome = DriveOutcome(plugin=plugin)
    if not valid_source(repo, sha, subdir):
        outcome.error = "invalid catalog source, commit SHA, or subdirectory; refusing install"
        return outcome
    box = Sandbox(hermes=hermes, env=env, config=config, popen=popen, tmp_root=tmp_root)
    try:
        box.create()
        outcome.sandbox = str(box.home)
        if run_baseline:
            outcome.baseline = box.chat(request, run_timeout)
        inst = box.install_pinned(repo, sha, subdir, install_timeout)
        if inst.timed_out:
            outcome.error = f"install timed out after {install_timeout:.0f}s"
        elif inst.rc != 0:
            outcome.error = f"install failed (exit {inst.rc}): {inst.tail(400)}"
        else:
            outcome.installed = True
            outcome.with_plugin = box.chat(request, run_timeout)
        if outcome.baseline or outcome.with_plugin:
            rows = [r for r in box.sessions() if r.get("message_count") or r.get("messages")]
            runs = [r for r in (outcome.baseline, outcome.with_plugin) if r is not None]
            for res in runs:
                matching = [row for row in rows if res.session_id and row.get("id") == res.session_id]
                # Never assign another run's evidence when an ID is missing or duplicated.
                if (len(matching) == 1 and sum(r.session_id == res.session_id for r in runs) == 1
                        and transcript_complete(matching[0])):
                    res.tools_called, res.cost_usd = session_tools(matching[0])
                    res.evidence_available = True
    except Exception as exc:  # create() failure, Popen failure (hermes missing), ...
        outcome.error = outcome.error or f"{type(exc).__name__}: {exc}"
    finally:
        try:
            outcome.cleanup_ok = box.close()
        except Exception as exc:  # pragma: no cover - defensive
            outcome.cleanup_ok = False
            box.notes.append(f"cleanup error: {exc}")
        outcome.cleanup_notes = list(box.notes)
        outcome.steps = list(box.steps)
    return outcome
