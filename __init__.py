"""test-drive: don't guess which plugin helps. Test-drive it.

Notices when Hermes hits a capability wall, matches the public plugin catalog, and (after you
approve) installs a candidate into a throwaway Hermes home, runs your original request there once,
deletes the sandbox and shows you the before/after. Keep it through Hermes's own installer, or
discard it. Receipts of every drive stay in this plugin's state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
from typing import Any

# Submodules are imported inside register(): the module itself stays importable on its own (pytest
# imports a package's __init__ by path while collecting), and Hermes pays nothing until it registers.

logger = logging.getLogger(__name__)

__all__ = ["register"]

TOOLSET = "test_drive"

FIND_SCHEMA = {
    "name": "test_drive_find",
    "description": (
        "Search the official Hermes plugin catalog for plugins that add a capability you lack. Use it when "
        "you cannot do what the user asked because no loaded tool covers it. Returns the top 3 matches, each "
        "with a trust card (tier, catalog pin, keys needed, declared tools/hooks, disclosed network/shell use). "
        "Read-only: installs and runs nothing. Show the cards to the user; offer test_drive_run only if they want "
        "to try one."
    ),
    "parameters": {
        "type": "object",
        "properties": {"need": {"type": "string", "description": "The capability needed, in plain words "
                                                                  "(usually the user's request)."}},
        "required": ["need"],
        "additionalProperties": False,
    },
}

RUN_SCHEMA = {
    "name": "test_drive_run",
    "description": (
        "Test-drive one catalog plugin: after the user approves (Hermes asks them), install it into a throwaway "
        "Hermes home, run the request there once with the user's model, delete the sandbox, and return a "
        "before/after comparison. Slow (up to several minutes) and spends model tokens. Only call it after the "
        "user agreed to try that plugin. Never use it to install a plugin for real."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "plugin": {"type": "string", "description": "Catalog name exactly as test_drive_find returned it."},
            "request": {"type": "string", "description": "What to try in the sandbox: the user's original "
                                                          "request, unchanged."},
        },
        "required": ["plugin", "request"],
        "additionalProperties": False,
    },
}

KEEP_SCHEMA = {
    "name": "test_drive_keep",
    "description": (
        "The user chose to keep a plugin they test-drove: hand it to Hermes's own installer (an approval card in "
        "the desktop app, otherwise the exact `hermes plugins install` command for the user to run) and record "
        "the decision. Use decision='discard' when they do not want it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "plugin": {"type": "string", "description": "Catalog name of the test-driven plugin."},
            "decision": {"type": "string", "enum": ["keep", "discard"], "default": "keep"},
        },
        "required": ["plugin"],
        "additionalProperties": False,
    },
}

USAGE = """**/testdrive**: don't guess which plugin helps. Test-drive it.

| Command | What it does |
|---|---|
| `/testdrive <need>` | Top 3 catalog plugins for a need, with trust cards |
| `/testdrive run <plugin> [-- <request>]` | Approve, then try it in a throwaway Hermes home (default request: the one Hermes just failed) |
| `/testdrive card <plugin>` | Trust card for one plugin |
| `/testdrive keep <plugin>` | Install it for real through Hermes's own installer |
| `/testdrive discard <plugin>` | Forget it |
| `/testdrive history` | Receipts of past drives |"""


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def register(ctx) -> None:
    from .core import TestDrive

    td = TestDrive(ctx)

    # --- tools -------------------------------------------------------------------------------------
    def find_handler(args: dict, **_: Any) -> str:
        try:
            payload = td.find_payload(str((args or {}).get("need") or ""))
        except Exception as exc:
            return _json({"error": f"test_drive_find failed: {exc}"})
        if "candidates" in payload:
            payload["next"] = ("Show these trust cards to the user. If they want to try one, call test_drive_run "
                               "with its name and their original request; Hermes will ask them to approve first.")
        return _json(payload)

    def run_handler(args: dict, **kw: Any) -> str:
        args = args or {}
        try:
            prep = td.prepare(str(args.get("plugin") or ""), str(args.get("request") or ""),
                              str(kw.get("session_id") or kw.get("task_id") or ""))
            if prep.get("error") and not prep.get("skip"):
                return _json({"error": prep["error"]})
            # Approval already happened: the pre_tool_call hook below escalated this call to Hermes's
            # approval gate, which blocks the call unless the user approved it.
            result = td.run_drive(prep)
        except Exception as exc:
            logger.exception("test-drive: run failed")
            return _json({"error": f"test_drive_run failed: {exc}"})
        return _json({"status": result["status"], "report": result["text"], "receipt": result.get("receipt")})

    def keep_handler(args: dict, **_: Any) -> str:
        args = args or {}
        name = str(args.get("plugin") or "").strip()
        if not name:
            return _json({"error": "plugin is required"})
        text = td.discard(name) if args.get("decision") == "discard" else td.keep(name)
        return _json({"result": text})

    ctx.register_tool(name="test_drive_find", toolset=TOOLSET, schema=FIND_SCHEMA, handler=find_handler,
                      description=FIND_SCHEMA["description"], emoji="🧪")
    ctx.register_tool(name="test_drive_run", toolset=TOOLSET, schema=RUN_SCHEMA, handler=run_handler,
                      description=RUN_SCHEMA["description"], emoji="🧪")
    ctx.register_tool(name="test_drive_keep", toolset=TOOLSET, schema=KEEP_SCHEMA, handler=keep_handler,
                      description=KEEP_SCHEMA["description"], emoji="🧪")

    # --- hooks -------------------------------------------------------------------------------------
    def pre_tool_call(tool_name: str = "", args: Any = None, **_: Any):
        """Every test_drive_run goes through Hermes's human-approval gate (fails closed on deny,
        timeout, or when no human is present, e.g. cron)."""
        if tool_name != "test_drive_run":
            return None
        name = str((args or {}).get("plugin") or "").strip() if isinstance(args, dict) else ""
        try:
            cat = td.catalog(force=True)
            entry = cat.get(name)
            from .sandbox import valid_source
            if not entry or cat.is_removed(entry) or not valid_source(entry.repo, entry.sha, entry.subdir):
                raise ValueError("invalid catalog entry")
        except Exception:
            return {"action": "block", "message": "Could not verify the catalog source and pin; try again or use /testdrive run."}
        req = str((args or {}).get("request") or "") if isinstance(args, dict) else ""
        from .core import request_excerpt
        shown = (f'Request it will run: "{request_excerpt(req)}"' if req.strip()
                 else "No request given: it will use the request Hermes just failed in this session.")
        pin = f"{entry.repo} @ {entry.sha}" + (f" (subdir {entry.subdir})" if entry.subdir else "")
        identity = json.dumps([entry.repo, entry.sha, entry.subdir], ensure_ascii=False, separators=(",", ":"))
        rule_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return {"action": "approve",
                "message": (f"test-drive wants to download {name or 'a plugin'} ({pin}) into a throwaway Hermes "
                            "home and run your request there (child Hermes process; dangerous commands "
                            "denied; cleanup attempted afterwards). " + shown),
                "rule_key": f"test-drive:{name}@{rule_digest}"}

    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("pre_llm_call", td.on_pre_llm_call)
    ctx.register_hook("post_tool_call", td.on_post_tool_call)
    ctx.register_hook("transform_llm_output", td.on_transform_llm_output)

    # --- slash command -----------------------------------------------------------------------------
    def command(raw_args: str = "", **kw: Any) -> str:
        return handle_command(td, ctx, raw_args, session_id=str(kw.get("session_id") or ""))

    ctx.register_command("testdrive", handler=command,
                         description="Find a catalog plugin for a need and test-drive it in a throwaway profile",
                         args_hint="<need> | run <plugin> | keep <plugin> | history")


def handle_command(td: Any, ctx: Any, raw_args: str, *, session_id: str = "") -> str:
    raw = (raw_args or "").strip()
    if not raw or raw in ("help", "-h", "--help"):
        gap = td.last_gap(session_id)
        tail = ""
        if gap and gap.get("need"):
            cand = f" Suggested: `/testdrive run {gap['candidate']}`." if gap.get("candidate") else ""
            need = gap["need"] if len(gap["need"]) <= 120 else gap["need"][:119].rstrip() + "…"
            tail = f"\n\nLast gap noticed: “{need}”.{cand}"
        return USAGE + tail
    head, _, rest = raw.partition(" ")
    verb = head.lower()
    rest = rest.strip()
    if verb == "history":
        return td.history_text()
    if verb in ("card", "info", "show"):
        return td.card_text(rest) if rest else "Usage: `/testdrive card <plugin>`"
    if verb == "keep":
        return td.keep(rest) if rest else "Usage: `/testdrive keep <plugin>`"
    if verb == "discard":
        return td.discard(rest) if rest else "Usage: `/testdrive discard <plugin>`"
    if verb == "run":
        return _command_run(td, ctx, rest, session_id)
    return td.find_text(raw)


def _command_run(td: Any, ctx: Any, rest: str, session_id: str) -> str:
    request = ""
    if " -- " in f" {rest} ":
        rest, _, request = f" {rest} ".partition(" -- ")
    try:
        words = shlex.split(rest)
    except ValueError:
        words = rest.split()
    yes = any(w in ("--yes", "-y") for w in words)
    words = [w for w in words if w not in ("--yes", "-y")]
    if not words:
        return "Usage: `/testdrive run <plugin> [-- <request>]`"
    name = words[0]
    prep = td.prepare(name, request.strip(), session_id)
    if prep.get("error") and not prep.get("skip"):
        return prep["error"]
    if prep.get("skip"):
        return td.run_drive(prep)["text"]
    approved = yes
    if not approved:
        choice = None
        try:
            choice = ctx.ask_choice(td.approval_message(prep), [
                {"value": "yes", "label": "Run the test drive", "description": "download, run once, delete"},
                {"value": "no", "label": "Cancel"}])
        except Exception as exc:
            logger.debug("test-drive: ask_choice failed: %s", exc)
        approved = choice == "yes"
        if choice is None:
            return ("No answer, so nothing was downloaded (the question was cancelled, timed out, or this surface "
                    f"cannot ask). To approve in one step, type `/testdrive run {name} --yes"
                    f"{' -- ' + request.strip() if request.strip() else ''}`.")
    if not approved:
        td._save_receipt({"id": f"denied-{int(td._clock())}", "at": td._clock(), "plugin": prep["entry"].name,
                          "sha": prep["entry"].sha, "request": prep["request"][:300], "status": "denied"})
        return f"Cancelled. Nothing was downloaded for `{name}`."
    return td.run_drive(prep)["text"]

