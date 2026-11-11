"""The public Hermes plugin catalog: fetched over HTTPS from Nous hosts only, cached with a TTL.

Source: the same document ``hermes plugins search`` / ``install`` read
(``https://hermes-agent.nousresearch.com/docs/api/plugin-catalog.json``). Redirects are followed
by hand, one hop at a time, and only to ``https`` URLs on Nous hosts; anything else is refused.
The cache is a compact projection (no READMEs, no images) written atomically to this plugin's
data directory. A failed fetch falls back to the last good cache whatever its age.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urljoin, urlsplit

CATALOG_URL = "https://hermes-agent.nousresearch.com/docs/api/plugin-catalog.json"
# Hosts the catalog may be served from (today the docs host 301s to the GitHub Pages mirror).
ALLOWED_HOSTS = ("hermes-agent.nousresearch.com", "nousresearch.github.io")
ALLOWED_HOST_SUFFIXES = (".nousresearch.com",)
MAX_BYTES = 16 * 1024 * 1024
MAX_REDIRECTS = 3
TIMEOUT_S = 10.0
DEFAULT_TTL_S = 6 * 3600
CACHE_NAME = "catalog-cache.json"
USER_AGENT = "hermes-test-drive/0.1.0 (+https://github.com/Xuefeng-Zhu/hermes-test-drive)"


class CatalogError(RuntimeError):
    pass


def host_allowed(url: str) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and bool(host) and (
        host in ALLOWED_HOSTS or any(host.endswith(s) for s in ALLOWED_HOST_SUFFIXES))


@dataclass
class Entry:
    name: str
    repo: str = ""
    sha: str = ""
    description: str = ""
    maintainer: str = ""
    tier: str = "community"
    category: str = ""
    title: str = ""
    version: str = ""
    requires_hermes: str = ""
    platforms: list[str] = field(default_factory=list)
    known_issues: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    middleware: list[str] = field(default_factory=list)
    requires_env: list[str] = field(default_factory=list)
    subdir: str = ""

    @property
    def display(self) -> str:
        return self.title or self.name

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> Optional["Entry"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not raw["name"]:
            return None
        caps = raw.get("capabilities") if isinstance(raw.get("capabilities"), dict) else {}

        def strs(value: Any) -> list[str]:
            return [str(v) for v in value if isinstance(v, (str, int, float))] if isinstance(value, list) else []

        def s(key: str) -> str:
            v = raw.get(key)
            return str(v) if isinstance(v, (str, int, float)) else ""

        return cls(
            name=raw["name"], repo=s("repo"), sha=s("sha"), description=" ".join(s("description").split()),
            maintainer=s("maintainer"), tier=s("tier") or "community", category=s("category"),
            title=s("title"), version=s("version"), requires_hermes=s("requires_hermes"),
            platforms=strs(raw.get("platforms")), known_issues=strs(raw.get("known_issues")),
            tools=strs(caps.get("provides_tools")), hooks=strs(caps.get("provides_hooks")),
            middleware=strs(caps.get("provides_middleware")), requires_env=strs(caps.get("requires_env")),
            subdir=s("subdir"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in ("", [], None)}


@dataclass
class Catalog:
    entries: list[Entry]
    removed: set[str] = field(default_factory=set)
    removed_repos: set[str] = field(default_factory=set)
    generated_at: str = ""
    fetched_at: float = 0.0
    source: str = "live"  # live | cache | stale-cache

    def get(self, name: str) -> Optional[Entry]:
        key = (name or "").strip().lower()
        for e in self.entries:
            if e.name.lower() == key:
                return e
        return None

    def is_removed(self, entry: Entry) -> bool:
        return entry.name in self.removed or (bool(entry.repo) and entry.repo.rstrip("/").lower() in self.removed_repos)


def parse_document(data: Any) -> Catalog:
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise CatalogError("unexpected catalog payload (no 'entries' list)")
    entries = [e for e in (Entry.from_mapping(r) for r in data["entries"]) if e is not None]
    if not entries:
        raise CatalogError("catalog payload has no usable entries")
    removed, removed_repos = set(), set()
    for r in data.get("removed") or []:
        if isinstance(r, dict):
            if isinstance(r.get("name"), str):
                removed.add(r["name"])
            if isinstance(r.get("repo"), str) and r["repo"]:
                removed_repos.add(r["repo"].rstrip("/").lower())
    return Catalog(entries=entries, removed=removed, removed_repos=removed_repos,
                   generated_at=str(data.get("generated_at") or ""))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - we follow redirects by hand
        return None


def http_get(url: str, *, timeout: float = TIMEOUT_S, max_bytes: int = MAX_BYTES) -> bytes:
    """GET ``url`` following at most MAX_REDIRECTS hops, each one re-checked against Nous hosts."""
    opener = urllib.request.build_opener(_NoRedirect)
    current = url
    for _hop in range(MAX_REDIRECTS + 1):
        if not host_allowed(current):
            raise CatalogError(f"refusing to fetch the catalog from a non-Nous URL: {current}")
        req = urllib.request.Request(current, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            resp = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                location = exc.headers.get("Location") if exc.headers else None
                if not location:
                    raise CatalogError(f"redirect without Location from {current}") from None
                current = urljoin(current, location)
                continue
            raise CatalogError(f"catalog fetch failed: HTTP {exc.code}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise CatalogError(f"catalog fetch failed: {exc}") from None
        with resp:
            body = resp.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise CatalogError("catalog payload too large")
        return body
    raise CatalogError("too many redirects fetching the catalog")


class CatalogStore:
    """TTL cache in front of the live catalog. ``fetcher`` is injectable for tests."""

    def __init__(self, cache_dir: Callable[[], Path], *, ttl_s: float = DEFAULT_TTL_S,
                 fetcher: Callable[[str], bytes] = http_get, clock: Callable[[], float] = time.time):
        self._cache_dir = cache_dir
        self.ttl_s = ttl_s
        self._fetch = fetcher
        self._clock = clock
        self._memo: Optional[Catalog] = None

    def _path(self) -> Path:
        return Path(self._cache_dir()) / CACHE_NAME

    def _read_cache(self) -> Optional[Catalog]:
        path = self._path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cat = Catalog(entries=[e for e in (Entry.from_mapping(_unflatten(r)) for r in raw["entries"]) if e],
                          removed=set(raw.get("removed") or []), removed_repos=set(raw.get("removed_repos") or []),
                          generated_at=raw.get("generated_at", ""), fetched_at=float(raw.get("fetched_at", 0)),
                          source="cache")
            return cat if cat.entries else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_cache(self, cat: Catalog) -> None:
        path = self._path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            doc = {"generated_at": cat.generated_at, "fetched_at": cat.fetched_at,
                   "removed": sorted(cat.removed), "removed_repos": sorted(cat.removed_repos),
                   "entries": [_flatten(e) for e in cat.entries]}
            fd, tmp = tempfile.mkstemp(prefix=CACHE_NAME + ".", dir=str(path.parent))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, separators=(",", ":"))
            os.replace(tmp, path)
        except OSError:
            pass

    def cached(self) -> Optional[Catalog]:
        """Whatever is in memory or on disk, never touching the network (for hooks)."""
        if self._memo is None:
            self._memo = self._read_cache()
        return self._memo

    def load(self, *, force: bool = False) -> Catalog:
        now = self._clock()
        cat = self.cached()
        if cat is not None and not force and now - cat.fetched_at < self.ttl_s:
            return cat
        try:
            fresh = parse_document(json.loads(self._fetch(CATALOG_URL)))
        except (CatalogError, ValueError) as exc:
            if cat is not None:
                cat.source = "stale-cache"
                return cat
            raise CatalogError(str(exc)) from None
        fresh.fetched_at = now
        fresh.source = "live"
        self._write_cache(fresh)
        self._memo = fresh
        return fresh


def _flatten(e: Entry) -> dict[str, Any]:
    d = e.to_mapping()
    caps = {"provides_tools": d.pop("tools", []), "provides_hooks": d.pop("hooks", []),
            "provides_middleware": d.pop("middleware", []), "requires_env": d.pop("requires_env", [])}
    d["capabilities"] = caps
    return d


def _unflatten(raw: dict[str, Any]) -> dict[str, Any]:
    return raw
