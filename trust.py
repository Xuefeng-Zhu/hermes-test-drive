"""Trust cards: what the catalog entry declares, laid out so a person can decide before anything
is downloaded. Everything comes from the entry itself; the network/shell/storage flags are read
from the entry's own disclosure text and labelled as such (the catalog has no structured field)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .catalog import Catalog, Entry

_DISCLOSURE = re.compile(r"disclosure\s*[—:\-–]+\s*(.+)$", re.IGNORECASE | re.DOTALL)
_FLAGS = {
    "network": re.compile(r"\b(?:network|https?://|api\.|calls?\s+\S+\.(?:com|ai|io|net|org)|sends?\s|uploads?|"
                          r"hosted|cloud|server|webhook|fetch(?:es)?\s)", re.IGNORECASE),
    "no-network": re.compile(r"\bno\s+network|\boffline\b|never\s+(?:calls|contacts)\s+the\s+network", re.IGNORECASE),
    "shell": re.compile(r"\b(?:shell|subprocess|runs?\s+`|terminal|docker|executes?\s+commands?|spawns?|cli\b)", re.IGNORECASE),
    "storage": re.compile(r"\b(?:stores?|writes?|saves?|sqlite|database|plugin-data|cache[sd]?|log(?:s|book)?)\b", re.IGNORECASE),
    "background": re.compile(r"\b(?:background|daemon|poll(?:s|ing|er)|scheduled|every\s+\d+\s*(?:s|sec|min))", re.IGNORECASE),
}


@dataclass
class EnvCheck:
    present: list[str]
    missing: list[str]


def disclosure_text(entry: Entry) -> str:
    m = _DISCLOSURE.search(entry.description or "")
    return " ".join(m.group(1).split()) if m else ""


def declared_flags(entry: Entry) -> dict[str, bool]:
    text = disclosure_text(entry) or entry.description or ""
    flags = {k: bool(rx.search(text)) for k, rx in _FLAGS.items()}
    if flags.pop("no-network"):
        flags["network"] = False
    return flags


def env_check(entry: Entry, has_env: Callable[[str], bool]) -> EnvCheck:
    present, missing = [], []
    for name in entry.requires_env:
        (present if has_env(name) else missing).append(name)
    return EnvCheck(present, missing)


def _short(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def card(entry: Entry, catalog: Optional[Catalog], *, env: Optional[EnvCheck] = None,
         rank: Optional[int] = None, why: Optional[list[str]] = None) -> str:
    """Markdown trust card for one entry."""
    removed = bool(catalog and catalog.is_removed(entry))
    head = f"{rank}. " if rank else ""
    lines = [f"**{head}{entry.display}** (`{entry.name}`) · {entry.tier} · by {entry.maintainer or 'unknown'}"]
    lines.append(f"   {_short(entry.description.split(' Disclosure')[0], 220)}")
    pin = f"{entry.version + ' @ ' if entry.version else ''}{entry.sha[:8] or 'no sha'}"
    status = "REMOVED from the catalog (installs are refused)" if removed else "merged in the official catalog"
    lines.append(f"   • Catalog: {status}; pinned {pin}; {entry.repo or 'no repo'}"
                 + (f" (subdir `{entry.subdir}`)" if entry.subdir else ""))
    caps = []
    if entry.tools:
        caps.append(f"tools {', '.join(entry.tools[:6])}{' …' if len(entry.tools) > 6 else ''}")
    if entry.hooks:
        caps.append(f"hooks {', '.join(entry.hooks[:6])}")
    if entry.middleware:
        caps.append(f"middleware {', '.join(entry.middleware[:4])}")
    lines.append(f"   • Declares: {'; '.join(caps) if caps else 'no tools/hooks (skills, MCP or UI only)'}")
    if entry.requires_env:
        if env is None:
            lines.append(f"   • Keys needed: {', '.join(entry.requires_env)}")
        else:
            parts = [f"{n} ✓" for n in env.present] + [f"{n} ✗ missing" for n in env.missing]
            lines.append(f"   • Keys needed: {', '.join(parts)}")
    else:
        lines.append("   • Keys needed: none")
    flags = declared_flags(entry)
    shown = [k for k, v in flags.items() if v]
    disc = disclosure_text(entry)
    lines.append("   • Disclosure mentions: " + (", ".join(shown) if shown else "no network, shell or storage")
                 + ("" if disc else " (no disclosure line; read from its description)"))
    if disc:
        lines.append(f"   • Disclosure: “{_short(disc, 240)}”")
    meta = []
    if entry.requires_hermes:
        meta.append(f"Hermes {entry.requires_hermes}")
    if entry.platforms:
        meta.append("/".join(entry.platforms))
    if entry.category:
        meta.append(f"category {entry.category}")
    if meta:
        lines.append(f"   • {' · '.join(meta)}")
    for issue in entry.known_issues[:2]:
        lines.append(f"   • Known issue: {_short(issue, 160)}")
    if why:
        lines.append(f"   • Matched on: {', '.join(why[:6])}")
    return "\n".join(lines)


def card_dict(entry: Entry, catalog: Optional[Catalog], env: Optional[EnvCheck] = None) -> dict:
    return {
        "name": entry.name, "title": entry.display, "tier": entry.tier, "maintainer": entry.maintainer,
        "in_catalog": not (catalog and catalog.is_removed(entry)), "repo": entry.repo, "sha": entry.sha,
        "version": entry.version, "requires_hermes": entry.requires_hermes, "category": entry.category,
        "tools": entry.tools, "hooks": entry.hooks, "middleware": entry.middleware,
        "requires_env": entry.requires_env,
        "env_missing": env.missing if env else None,
        "declared": declared_flags(entry), "disclosure": disclosure_text(entry) or None,
        "known_issues": entry.known_issues,
    }
