"""test-drive service: gap capture and nudges, catalog matching, approved sandbox drives, receipts.

Everything that touches Hermes goes through the PluginContext handed to ``register`` (settings,
state, ask_choice, dispatch_tool, llm) or through the public ``hermes`` CLI in a child process.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import gaps, sandbox, trust
from .catalog import Catalog, CatalogError, CatalogStore, Entry
from .ranking import Match, Ranker

logger = logging.getLogger(__name__)

PLUGIN_ID = "test-drive"
VERSION = "0.1.0"
MAX_RECEIPTS = 50
NUDGE_COOLDOWN_TURNS = 3
SKIP_PLATFORMS = {"cron", "kanban", "subagent", "batch"}

DEFAULTS: dict[str, Any] = {
    "nudges": True,
    "llm_check": False,
    "run_timeout": 120,
    "install_timeout": 240,
    "baseline": "auto",          # auto | always | never
    "catalog_ttl_hours": 6,
    "sandbox_provider": "",
    "sandbox_model": "",
    "sandbox_base_url": "",
    "sandbox_key_env": "",
    "hermes_command": "",
}


def _excerpt(text: str, n: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def request_excerpt(text: str, n: int = 200) -> str:
    """The request as shown in approval prompts: newlines collapsed, first n characters."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= n else flat[: n - 1].rstrip() + "…"


@dataclass
class TurnState:
    user_message: str = ""
    failures: list[gaps.ToolFailure] = field(default_factory=list)
    turn: int = 0
    last_nudge_turn: int = -10
    nudged: set = field(default_factory=set)


class TestDrive:
    __test__ = False  # not a pytest class

    def __init__(self, ctx: Any, *, store: Optional[CatalogStore] = None,
                 drive_fn: Callable[..., sandbox.DriveOutcome] = sandbox.drive,
                 clock: Callable[[], float] = time.time):
        self.ctx = ctx
        self._drive_fn = drive_fn
        self._clock = clock
        self.store = store or CatalogStore(self._data_dir, ttl_s=float(self.setting("catalog_ttl_hours")) * 3600)
        self._ranker: Optional[Ranker] = None
        self._ranker_for: Optional[int] = None
        self._sessions: dict[str, TurnState] = {}
        self._lock = threading.Lock()
        self._drive_lock = threading.Lock()
        self._warming = False

    # settings / state ----------------------------------------------------------------------------
    def setting(self, key: str) -> Any:
        default = DEFAULTS[key]
        try:
            value = self.ctx.get_config(key, default=default)
        except Exception:
            value = default
        if isinstance(default, bool):
            return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                return default
        return "" if value is None else str(value)

    def _data_dir(self) -> Path:
        try:
            return Path(self.ctx.state.data_dir)
        except Exception:
            from plugins.plugin_storage import plugin_data_dir
            return Path(plugin_data_dir(PLUGIN_ID))

    def _state_get(self, key: str, default: Any) -> Any:
        try:
            value = self.ctx.state.get(key, default=default)
        except Exception:
            return default
        return default if value is None else value

    def _state_set(self, key: str, value: Any) -> None:
        try:
            self.ctx.state.set(key, value)
        except Exception as exc:
            logger.warning("test-drive: could not save %s: %s", key, exc)

    # catalog ------------------------------------------------------------------------------------
    def catalog(self, *, force: bool = False) -> Catalog:
        return self.store.load(force=force)

    def ranker(self, cat: Catalog) -> Ranker:
        if self._ranker is None or self._ranker_for != id(cat):
            self._ranker, self._ranker_for = Ranker(cat), id(cat)
        return self._ranker

    def _warm_in_background(self) -> None:
        if self._warming or self.store.cached() is not None:
            return
        self._warming = True

        def work():
            try:
                self.store.load()
            except Exception as exc:
                logger.debug("test-drive: catalog warm failed: %s", exc)
            finally:
                self._warming = False

        threading.Thread(target=work, name="test-drive-catalog", daemon=True).start()

    def find(self, need: str, limit: int = 3) -> tuple[list[Match], Catalog]:
        cat = self.catalog()
        matches = self.ranker(cat).rank(need, limit=max(limit * 3, 9) if self.setting("llm_check") else limit)
        if self.setting("llm_check") and len(matches) > limit:
            matches = self._llm_rerank(need, matches)[:limit]
        if matches:
            floor = matches[0].score * 0.25
            matches = [m for m in matches if m.score >= floor]
        return matches[:limit], cat

    def _llm_rerank(self, need: str, matches: list[Match]) -> list[Match]:
        llm = getattr(self.ctx, "llm", None)
        if llm is None:
            return matches
        listing = "\n".join(f"{i + 1}. {m.entry.name}: {_excerpt(m.entry.description, 200)}" for i, m in enumerate(matches))
        try:
            res = llm.complete(messages=[
                {"role": "system", "content": "You rank Hermes plugins for a user's need. Reply with the numbers "
                                              "of the best matches, best first, comma-separated. Nothing else."},
                {"role": "user", "content": f"Need: {need}\n\nPlugins:\n{listing}"}],
                max_tokens=20, timeout=10, purpose="test-drive: rerank catalog matches")
            order = [int(x) - 1 for x in str(res.text).replace(" ", "").split(",") if x.strip().isdigit()]
        except Exception as exc:
            logger.debug("test-drive: rerank failed: %s", exc)
            return matches
        picked = [matches[i] for i in order if 0 <= i < len(matches)]
        return list(dict.fromkeys(picked)) + [m for m in matches if m not in picked]

    # env / secrets --------------------------------------------------------------------------------
    @staticmethod
    def _secret(name: str) -> Optional[str]:
        try:
            from agent.secret_scope import get_secret
            return get_secret(name)
        except ImportError:
            return os.environ.get(name)
        except Exception:
            return None

    def has_env(self, name: str) -> bool:
        return bool(self._secret(name))

    def _user_model(self) -> tuple[dict[str, Any], Optional[tuple[str, dict[str, Any]]]]:
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly() or {}
        except Exception:
            cfg = {}
        model = dict(cfg.get("model") or {}) if isinstance(cfg.get("model"), dict) else {}
        for key, cfg_key in (("sandbox_provider", "provider"), ("sandbox_model", "default"),
                             ("sandbox_base_url", "base_url"), ("sandbox_key_env", "key_env")):
            value = self.setting(key)
            if value:
                model[cfg_key] = value
                if cfg_key == "key_env":
                    model.pop("api_key", None)
        provider = str(model.get("provider") or "")
        entry = None
        providers = cfg.get("providers")
        if provider and isinstance(providers, dict) and isinstance(providers.get(provider), dict):
            entry = (provider, dict(providers[provider]))
        return model, entry

    @staticmethod
    def _registry_keys(provider: str) -> list[str]:
        from providers import get_provider_profile
        prof = get_provider_profile(provider)
        return list(getattr(prof, "env_vars", ()) or ()) if prof else []

    def sandbox_env(self, entry: Entry, model: dict[str, Any],
                    provider_entry: Optional[tuple[str, dict[str, Any]]]) -> tuple[dict[str, str], list[str]]:
        env = {k: v for k, v in os.environ.items() if k in sandbox.PASS_ENV}
        passed: list[str] = []
        names = sandbox.model_key_env(model, provider_entry[1] if provider_entry else None, self._registry_keys)
        for name in [*names, *entry.requires_env]:
            value = self._secret(name)
            if value and name not in env:
                env[name] = value
                passed.append(name)
        return env, passed

    def model_problem(self, model: dict[str, Any], passed: list[str], key_names: list[str],
                      literal_key: bool = False) -> Optional[str]:
        provider = str(model.get("provider") or "").lower()
        if not model:
            return ("No model is configured in this profile, so the sandbox has nothing to run. Set one with "
                    "`hermes model`, or set test-drive's sandbox_provider / sandbox_model settings.")
        if provider in sandbox.OAUTH_ONLY and not self.setting("sandbox_key_env"):
            return (f"Your model ({provider}) signs in with OAuth, and test-drive never copies sign-ins into a "
                    "throwaway profile. Point the sandbox at an API-key route instead: set test-drive's "
                    "sandbox_provider, sandbox_model and sandbox_key_env (and sandbox_base_url for a custom endpoint).")
        if literal_key and not any(n in passed for n in key_names):
            return ("Your model key is written literally in config.yaml, and test-drive never copies a literal key "
                    "into a throwaway profile (the plugin under test could read it). Put the key in an env var "
                    "and point test-drive's sandbox_key_env setting at it (or use key_env / ${VAR} in config.yaml).")
        if key_names and not any(n in passed for n in key_names):
            local = str(model.get("base_url") or "").startswith(("http://localhost", "http://127.", "http://0.0.0.0"))
            if not local:
                return (f"The sandbox needs your model key in {' or '.join(key_names)}, which is not set in this "
                        "profile. Set it, or point test-drive's sandbox_key_env at the variable that holds it.")
        return None

    # gap detection hooks ----------------------------------------------------------------------------
    def _turn(self, session_id: str) -> TurnState:
        with self._lock:
            st = self._sessions.get(session_id or "")
            if st is None:
                if len(self._sessions) > 256:
                    self._sessions.pop(next(iter(self._sessions)))
                st = self._sessions[session_id or ""] = TurnState()
            return st

    def on_pre_llm_call(self, session_id: str = "", user_message: str = "", platform: str = "", **_: Any):
        st = self._turn(session_id)
        st.user_message = user_message or ""
        st.failures = []
        st.turn += 1
        if self.setting("nudges") and platform not in SKIP_PLATFORMS:
            self._warm_in_background()
        return None

    def on_post_tool_call(self, tool_name: str = "", result: Any = None, session_id: str = "",
                          task_id: str = "", status: Optional[str] = None, error_message: Optional[str] = None,
                          **_: Any) -> None:
        if tool_name.startswith("test_drive_"):
            return
        failure = gaps.tool_failure(tool_name, result, status, error_message)
        if failure:
            self._turn(session_id or task_id).failures.append(failure)

    def on_transform_llm_output(self, response_text: str = "", session_id: str = "", platform: str = "",
                                **_: Any) -> Optional[str]:
        try:
            footer = self._nudge_for(response_text, session_id, platform)
        except Exception as exc:  # a hook must never break the reply
            logger.debug("test-drive: nudge failed: %s", exc)
            return None
        return f"{response_text.rstrip()}\n\n{footer}" if footer else None

    def _nudge_for(self, response: str, session_id: str, platform: str) -> Optional[str]:
        if not self.setting("nudges") or platform in SKIP_PLATFORMS or not response:
            return None
        st = self._turn(session_id)
        if st.user_message.lstrip().startswith("/testdrive"):
            return None
        gap = gaps.detect(st.user_message, response, st.failures)
        if not gap.is_gap and gaps.is_ambiguous(gap) and self.setting("llm_check"):
            if self._llm_says_gap(st.user_message, response):
                gap.score = gaps.THRESHOLD
                gap.reasons.append("model check agreed")
        if not gap.is_gap:
            return None
        self._remember_gap(session_id, gap, request=st.user_message)
        if st.turn - st.last_nudge_turn < NUDGE_COOLDOWN_TURNS:
            return None
        cat = self.store.cached()
        if cat is None:
            st.last_nudge_turn = st.turn
            return (f"🧪 test-drive: that looks like a missing capability. `/testdrive {_excerpt(gap.need, 80)}` "
                    "searches the plugin catalog and can try a match in a throwaway sandbox first.")
        matches = [m for m in self.ranker(cat).rank(gap.need, limit=3, min_score=6.0) if m.entry.name not in st.nudged]
        if not matches:
            return None
        top = matches[0]
        st.last_nudge_turn = st.turn
        st.nudged.add(top.entry.name)
        self._remember_gap(session_id, gap, candidate=top.entry.name, request=st.user_message)
        e = top.entry
        keys = f" · needs {', '.join(e.requires_env)}" if e.requires_env else " · no keys needed"
        return (f"🧪 test-drive: a catalog plugin may cover this: **{e.display}** (`{e.name}`, {e.tier}, "
                f"pinned {e.sha[:8]}{keys}). Try it in a throwaway sandbox before installing anything: "
                f"`/testdrive run {e.name}` · other options: `/testdrive {_excerpt(gap.need, 60)}`")

    def _llm_says_gap(self, user_message: str, response: str) -> bool:
        llm = getattr(self.ctx, "llm", None)
        if llm is None:
            return False
        try:
            res = llm.complete(messages=[
                {"role": "system", "content": "Answer yes or no. Did the assistant fail because it lacks a tool, "
                                              "integration or access (not because of policy or missing facts)?"},
                {"role": "user", "content": f"User: {_excerpt(user_message, 500)}\n\nAssistant: {_excerpt(response, 800)}"}],
                max_tokens=3, timeout=8, purpose="test-drive: capability-gap check")
            return str(res.text).strip().lower().startswith("yes")
        except Exception:
            return False

    def _remember_gap(self, session_id: str, gap: gaps.Gap, candidate: str = "", request: str = "") -> None:
        record = {"session_id": session_id, "need": gap.need, "request": (request or gap.need).strip()[:4000],
                  "failure": gap.failure,
                  "reasons": gap.reasons, "candidate": candidate, "at": self._clock()}
        gaps_by_session = self._state_get("last_gaps", {})
        if not isinstance(gaps_by_session, dict):
            gaps_by_session = {}
        gaps_by_session[session_id or "_"] = record
        if len(gaps_by_session) > 20:
            for key in sorted(gaps_by_session, key=lambda k: gaps_by_session[k].get("at", 0))[:-20]:
                gaps_by_session.pop(key, None)
        self._state_set("last_gaps", gaps_by_session)
        self._state_set("last_gap", record)

    def last_gap(self, session_id: str = "") -> Optional[dict[str, Any]]:
        by_session = self._state_get("last_gaps", {})
        if session_id:
            return by_session.get(session_id) if isinstance(by_session, dict) and isinstance(by_session.get(session_id), dict) else None
        # Never reuse another session's request when the caller has no session identity.
        return None

    # receipts ----------------------------------------------------------------------------------------
    def receipts(self) -> list[dict[str, Any]]:
        data = self._state_get("drives", [])
        return data if isinstance(data, list) else []

    def _save_receipt(self, receipt: dict[str, Any]) -> None:
        items = [r for r in self.receipts() if r.get("id") != receipt.get("id")]
        items.append(receipt)
        self._state_set("drives", items[-MAX_RECEIPTS:])

    def set_decision(self, name: str, decision: str) -> Optional[dict[str, Any]]:
        items = self.receipts()
        for r in reversed(items):
            if r.get("plugin") == name:
                r["decision"] = decision
                r["decided_at"] = self._clock()
                self._state_set("drives", items)
                return r
        return None

    def history_text(self, limit: int = 10) -> str:
        items = self.receipts()[-limit:]
        if not items:
            return "No test drives yet. Try `/testdrive <what you need>`."
        lines = [f"**Test-drive receipts** (last {len(items)} of {len(self.receipts())})"]
        for r in reversed(items):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.get("at", 0)))
            used = r.get("plugin_tools_used")
            verdict = {"ok": ("tool-use evidence unavailable" if used is None else
                              "used its tools ✓" if used else "ran, did not use its tools"),
                       "skipped_keys": f"skipped: missing {', '.join(r.get('missing_env') or [])}",
                       "denied": "not approved", "failed": f"failed: {_excerpt(r.get('error', ''), 80)}"}.get(
                r.get("status"), r.get("status", "?"))
            decision = f" · {r['decision']}" if r.get("decision") else ""
            secs = f" · {r['seconds']:.0f}s" if isinstance(r.get("seconds"), (int, float)) else ""
            lines.append(f"- {when} `{r.get('plugin')}` @ {str(r.get('sha', ''))[:8]}: {verdict}{secs}{decision}"
                         f" · “{_excerpt(r.get('request', ''), 60)}”")
        return "\n".join(lines)

    # find / cards --------------------------------------------------------------------------------------
    def find_payload(self, need: str) -> dict[str, Any]:
        need = (need or "").strip()
        if not need:
            return {"error": "Say what you need, e.g. 'convert a PDF to markdown'."}
        try:
            matches, cat = self.find(need)
        except CatalogError as exc:
            return {"error": f"Could not load the plugin catalog: {exc}"}
        cands = []
        for i, m in enumerate(matches, 1):
            env = trust.env_check(m.entry, self.has_env)
            cands.append({"rank": i, "score": m.score, "matched": m.matched,
                          "card": trust.card(m.entry, cat, env=env, rank=i, why=m.matched),
                          **trust.card_dict(m.entry, cat, env)})
        return {"need": need, "catalog": {"source": cat.source, "generated_at": cat.generated_at,
                                          "entries": len(cat.entries)}, "candidates": cands}

    def find_text(self, need: str) -> str:
        payload = self.find_payload(need)
        if "error" in payload:
            return payload["error"]
        if not payload["candidates"]:
            return (f"No catalog plugin matched “{_excerpt(need, 80)}”. Try other words, or browse with "
                    "`hermes plugins search <words>`.")
        cards = "\n\n".join(c["card"] for c in payload["candidates"])
        first = payload["candidates"][0]["name"]
        src = payload["catalog"]
        return (f"**Top {len(payload['candidates'])} catalog matches for** “{_excerpt(need, 80)}” "
                f"({src['entries']} entries, {src['source']}):\n\n{cards}\n\n"
                f"Test-drive one in a throwaway profile: `/testdrive run {first}` "
                f"(add `-- <request>` to choose what to try). Nothing is installed until you approve.")

    def card_text(self, name: str) -> str:
        try:
            cat = self.catalog()
        except CatalogError as exc:
            return f"Could not load the plugin catalog: {exc}"
        entry = cat.get(name)
        if not entry:
            return f"`{name}` is not in the plugin catalog."
        return trust.card(entry, cat, env=trust.env_check(entry, self.has_env))

    # the drive ------------------------------------------------------------------------------------------
    def prepare(self, name: str, request: str, session_id: str = "") -> dict[str, Any]:
        """Everything decided before approval: entry, request, key check, model route."""
        try:
            cat = self.catalog()
        except CatalogError as exc:
            return {"error": f"Could not load the plugin catalog: {exc}"}
        entry = cat.get(name)
        if entry is None:
            return {"error": f"`{name}` is not in the plugin catalog. Search with `/testdrive <need>`."}
        if cat.is_removed(entry):
            return {"error": f"`{name}` was removed from the catalog; test-drive will not run it."}
        if not sandbox.valid_source(entry.repo, entry.sha, entry.subdir):
            return {"error": f"`{name}` has an invalid catalog source, commit SHA, or subdirectory; refusing install."}
        if entry.name == PLUGIN_ID:
            return {"error": "test-drive cannot test-drive itself."}
        gap = self.last_gap(session_id) or {}
        remembered = str(gap.get("request") or gap.get("need") or "")
        request = (request or "").strip() or remembered
        if not request:
            return {"error": f"What should the sandbox try? `/testdrive run {name} -- <your request>`"}
        env_check = trust.env_check(entry, self.has_env)
        failure = str(gap.get("failure") or "") if request in (remembered, gap.get("need")) else ""
        prep = {"entry": entry, "catalog": cat, "request": request, "env": env_check, "failure": failure}
        if env_check.missing:
            prep["skip"] = (f"`{name}` needs {', '.join(env_check.missing)}, which "
                            f"{'is' if len(env_check.missing) == 1 else 'are'} not set in this profile, so the "
                            "test drive is skipped (nothing was downloaded). Set "
                            f"{'it' if len(env_check.missing) == 1 else 'them'} in this profile's .env (`hermes config env-path` "
                            "shows where it is), then run it again.")
            return prep
        model, provider_entry = self._user_model()
        env, passed = self.sandbox_env(entry, model, provider_entry)
        key_names = sandbox.model_key_env(model, provider_entry[1] if provider_entry else None, self._registry_keys)
        # Literal keys never go into the sandbox config: the plugin under test could read them.
        model, literal_m = sandbox.strip_literal_credentials(model)
        literal_p = False
        if provider_entry:
            clean, literal_p = sandbox.strip_literal_credentials(provider_entry[1])
            provider_entry = (provider_entry[0], clean)
        problem = self.model_problem(model, passed, key_names, literal_key=literal_m or literal_p)
        if problem:
            prep["error"] = problem
            return prep
        prep.update(model=model, provider_entry=provider_entry, child_env=env, passed=passed)
        return prep

    def approval_message(self, prep: dict[str, Any]) -> str:
        e: Entry = prep["entry"]
        keys = f" It gets these keys: {', '.join(prep.get('passed') or []) or 'none'}."
        return (f"Test-drive {e.name} ({e.tier}, {e.repo} @ {e.sha[:12]}): download it and its Python dependencies "
                f"into a throwaway Hermes home, run your request once there with your model "
                f"(Hermes child process, dangerous commands denied, {self.setting('run_timeout')}s limit), then delete "
                f"everything.{keys} Request: \"{request_excerpt(prep.get('request') or '')}\"")

    def run_drive(self, prep: dict[str, Any]) -> dict[str, Any]:
        """Run an APPROVED drive. Callers must have obtained approval first."""
        e: Entry = prep["entry"]
        receipt: dict[str, Any] = {"id": uuid.uuid4().hex[:12], "at": self._clock(), "plugin": e.name,
                                   "sha": e.sha, "version": e.version, "tier": e.tier,
                                   "request": _excerpt(prep["request"], 300)}
        if prep.get("skip"):
            receipt.update(status="skipped_keys", missing_env=prep["env"].missing)
            self._save_receipt(receipt)
            return {"status": "skipped_keys", "text": prep["skip"] + "\n\n" + trust.card(e, prep["catalog"], env=prep["env"]),
                    "receipt": receipt}
        if not self._drive_lock.acquire(blocking=False):
            return {"status": "busy", "text": "A test drive is already running; try again when it finishes."}
        try:
            mode = self.setting("baseline").lower()
            failure = prep.get("failure") or ""
            run_baseline = mode == "always" or (mode == "auto" and not failure)
            config = sandbox.build_config(prep["model"], prep.get("provider_entry"))
            started = self._clock()
            outcome = self._drive_fn(plugin=e.name, request=prep["request"],
                                     hermes=sandbox.hermes_argv(self.setting("hermes_command")),
                                     env=prep["child_env"], config=config, run_baseline=run_baseline,
                                     install_timeout=float(self.setting("install_timeout")),
                                     run_timeout=float(self.setting("run_timeout")),
                                     repo=e.repo, sha=e.sha, subdir=e.subdir)
            seconds = round(self._clock() - started, 1)
        finally:
            self._drive_lock.release()
        evidence_available = bool(outcome.with_plugin and outcome.with_plugin.evidence_available)
        used = [t for t in (outcome.with_plugin.tools_called if evidence_available else []) if t in e.tools]
        status = "ok" if outcome.with_plugin and outcome.with_plugin.ok else "failed"
        error = outcome.error or (outcome.with_plugin.error if outcome.with_plugin else "")
        receipt.update(status=status, seconds=seconds, error=_excerpt(error, 300),
                       plugin_tools_used=bool(used) if evidence_available else None,
                       tools_called=outcome.with_plugin.tools_called if evidence_available else None,
                       before=_excerpt(failure or (outcome.baseline.output if outcome.baseline else ""), 300),
                       after=_excerpt(outcome.with_plugin.output if outcome.with_plugin else "", 300),
                       cleanup_ok=outcome.cleanup_ok, env_passed=prep.get("passed") or [],
                       cost_usd=sum(r.cost_usd or 0 for r in (outcome.baseline, outcome.with_plugin) if r) or None)
        self._save_receipt(receipt)
        return {"status": status, "text": self.render(e, prep, outcome, used, failure, seconds), "receipt": receipt}

    def render(self, e: Entry, prep: dict[str, Any], o: sandbox.DriveOutcome, used: list[str],
               failure: str, seconds: float) -> str:
        pin = f"{e.version + ' @ ' if e.version else ''}{e.sha[:8]}"
        lines = [f"🧪 **Test drive: {e.display}** (`{e.name}`, {e.tier}, {pin})", f"Request: “{_excerpt(prep['request'], 160)}”", ""]
        if failure:
            before = f"{failure} (from this conversation)"
        elif o.baseline is not None:
            before = (_excerpt(o.baseline.output, 400) or "(no reply)") if o.baseline.ok else f"(baseline run failed: {o.baseline.error})"
            if o.baseline.tools_called:
                before += f"\n   tools: {', '.join(o.baseline.tools_called)}"
        else:
            before = "(not captured)"
        lines.append(f"**Without plugin:** {before}")
        if o.with_plugin is not None:
            w = o.with_plugin
            body = _excerpt(w.output, 900) if w.output else "(no reply)"
            lines.append(f"**With {e.name}:** {body}" if w.ok else f"**With {e.name}:** run failed: {w.error}\n{body}")
            if not w.evidence_available:
                lines.append("Evidence: tool-use evidence is unavailable for this run.")
            elif used:
                lines.append(f"Evidence: it called {', '.join(used)} (declared by {e.name}).")
            elif w.ok:
                lines.append(f"Evidence: the reply did not call any of {e.name}'s tools"
                             f"{' (' + ', '.join(e.tools[:4]) + ')' if e.tools else ''}; it may not help with this request.")
        else:
            lines.append(f"**With {e.name}:** not run: {o.error or 'install did not finish'}")
        cost = sum(r.cost_usd or 0 for r in (o.baseline, o.with_plugin) if r)
        cleanup = "deleted ✓" if o.cleanup_ok else "cleanup incomplete ⚠ " + "; ".join(o.cleanup_notes)
        lines.append(f"Sandbox: {cleanup} · {seconds:.0f}s total" + (f" · ~${cost:.4f} model spend" if cost else ""))
        if o.with_plugin is not None and o.with_plugin.ok:
            if not o.with_plugin.evidence_available or used or not e.tools:
                lines.append(f"\nKeep it? `/testdrive keep {e.name}` installs it into this profile through Hermes's "
                             f"normal install and consent prompts · `/testdrive discard {e.name}` forgets it.")
            else:
                lines.append(f"\nIt did not earn its keep on this request: `/testdrive discard {e.name}` · or keep it "
                             f"anyway with `/testdrive keep {e.name}`.")
        return "\n".join(lines)

    # keep / discard -------------------------------------------------------------------------------------
    def keep(self, name: str) -> str:
        try:
            cat = self.catalog()
        except CatalogError as exc:
            return f"Could not load the plugin catalog: {exc}"
        entry = cat.get(name)
        if entry is None:
            return f"`{name}` is not in the plugin catalog."
        result = ""
        try:
            result = self.ctx.dispatch_tool("manage_catalog", {
                "action": "install", "items": [{"kind": "plugin", "id": entry.name}],
                "reason": "You test-drove it with test-drive and chose to keep it."})
        except Exception as exc:
            result = json.dumps({"error": str(exc)})
        ok = _looks_installed(result)
        self.set_decision(entry.name, "kept" if ok else "keep: install pending")
        if ok:
            return (f"Installed `{entry.name}` through Hermes's install card. Hermes selected it by name; "
                    f"its catalog pin may differ from the test drive. {_excerpt(_text_of(result), 300)}")
        import shlex
        source = entry.repo + (f"#{entry.subdir}" if entry.subdir else "")
        command = f"hermes plugins install {shlex.quote(source)} --ref {shlex.quote(entry.sha)}"
        return (f"To keep `{entry.name}`, install it with Hermes's own installer, which shows its consent and "
                f"dependency prompts:\n\n    {command}\n\n"
                f"(current catalog pin {entry.sha[:12]} may differ from the test drive; start a new session afterwards to load it). Marked as pending in "
                "`/testdrive history`.")

    def discard(self, name: str) -> str:
        rec = self.set_decision(name, "discarded")
        if not rec:
            return f"No test drive of `{name}` on record."
        cleanup = "Its temporary profile was deleted." if rec.get("cleanup_ok") else "Check the drive report for cleanup status."
        return f"Discarded `{name}`. {cleanup}"


def _text_of(result: Any) -> str:
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except ValueError:
            return result
        if isinstance(data, dict):
            return str(data.get("error") or data.get("message") or data.get("note") or data)
    return str(result)


def _looks_installed(result: Any) -> bool:
    if not isinstance(result, str):
        return False
    try:
        data = json.loads(result)
    except ValueError:
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    rows = data.get("rows") or data.get("items") or data.get("results") or []
    return any(isinstance(r, dict) and str(r.get("status", "")).lower() in ("installed", "enabled") for r in rows)
