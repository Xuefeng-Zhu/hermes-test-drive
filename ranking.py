"""Rank catalog entries against a need: BM25 over the description plus weighted exact hits on the
name, title, tool names and category, with a small synonym table so "pdf" finds "document" tools.
Pure, deterministic, offline. An optional model rerank lives in core (off by default)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .catalog import Catalog, Entry

SELF_NAMES = {"test-drive"}

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from into onto about as is are was were be been
being it its this that these those i me my we our you your he she they them their what which who whom how
why when where can could would should will shall may might must do does did done have has had having not no
yes please just also any some all each every more most other such only own same so than too very s t don
want need help let lets get got make made use using via per one two up down out over under again further
here there hermes plugin plugins agent tool tools able something thing things like know tell give find show
way ways between work works working run runs running hi hello thanks thank kind sorry ok okay sure really
""".split())

# Small, hand-made synonym table: each key expands to the listed terms (one hop, lower weight).
SYNONYMS = {
    "email": ["mail", "gmail", "inbox", "imap", "smtp"], "mail": ["email", "inbox"],
    "calendar": ["schedule", "events", "meeting", "gcal"], "meeting": ["calendar", "meetings"],
    "pdf": ["document", "documents", "docs"], "doc": ["document", "docs"], "spreadsheet": ["sheets", "csv", "excel"],
    "csv": ["spreadsheet", "table", "data"], "excel": ["spreadsheet", "xlsx"],
    "image": ["images", "picture", "photo", "vision"], "photo": ["image", "picture"],
    "voice": ["speech", "audio", "tts", "stt"], "speech": ["voice", "audio"], "audio": ["voice", "sound"],
    "transcribe": ["transcription", "stt", "speech"], "video": ["youtube", "clip", "movie"],
    "browser": ["web", "browse", "chrome"], "web": ["browser", "website", "internet"],
    "search": ["lookup", "find", "query"], "weather": ["forecast", "climate"],
    "stock": ["stocks", "market", "finance", "ticker"], "crypto": ["bitcoin", "blockchain", "token"],
    "github": ["git", "repo", "pull"], "git": ["github", "repository"], "notion": ["notes", "wiki"],
    "slack": ["chat", "workspace"], "telegram": ["chat", "messaging"], "discord": ["chat", "server"],
    "memory": ["remember", "recall"], "remember": ["memory", "recall"], "cost": ["spend", "budget", "price"],
    "budget": ["cost", "spend"], "translate": ["translation", "language"], "culture": ["cultural", "country"],
    "cultural": ["culture"], "vaccine": ["vaccination", "immunization"], "map": ["maps", "location", "geo"],
    "music": ["song", "audio", "spotify"], "database": ["sql", "postgres", "sqlite"], "sql": ["database", "query"],
    "chart": ["charts", "plot", "graph"], "plot": ["chart", "graph"], "diagram": ["chart", "drawing"],
    "security": ["guard", "scan", "vulnerability"], "deploy": ["hosting", "publish"], "phone": ["sms", "call"],
    "todo": ["task", "tasks", "kanban"], "task": ["todo", "kanban"], "news": ["feed", "rss", "headlines"],
    "rss": ["feed", "news"], "recipe": ["cooking", "food"], "health": ["medical", "fitness"],
    "fitness": ["workout", "health"], "travel": ["trip", "flight", "hotel"], "flight": ["travel", "airline"],
    "trip": ["travel", "road"], "fuel": ["gas", "petrol"], "toll": ["road"], "windows": ["win"],
}

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.]*")


def stem(word: str) -> str:
    if len(word) >= 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) >= 5 and word.endswith("es") and (word[-3] in "sxz" or word[-4:-2] in ("ch", "sh")):
        return word[:-2]
    if len(word) >= 6 and word.endswith("ing"):
        return word[:-3]
    if len(word) >= 5 and word.endswith("ed"):
        return word[:-2]
    if len(word) >= 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokens(text: str) -> list[str]:
    words = _WORD.findall((text or "").lower().replace("_", " ").replace("-", " "))
    return [stem(w.strip(".")) for w in words if w.strip(".") and w.strip(".") not in STOPWORDS and len(w) > 1]


def query_terms(need: str) -> dict[str, float]:
    """Query term -> weight (1.0 for the user's words, 0.5 for synonyms)."""
    terms: dict[str, float] = {}
    for raw in _WORD.findall((need or "").lower().replace("_", " ").replace("-", " ")):
        raw = raw.strip(".")
        if not raw or raw in STOPWORDS or len(raw) < 2:
            continue
        terms[stem(raw)] = 1.0
        for syn in SYNONYMS.get(raw, []) + SYNONYMS.get(stem(raw), []):
            terms.setdefault(stem(syn), 0.5)
    return terms


@dataclass
class Match:
    entry: Entry
    score: float
    matched: list[str] = field(default_factory=list)


@dataclass
class _Doc:
    entry: Entry
    desc: Counter
    length: int
    name_terms: set
    tool_terms: set
    category: str


class Ranker:
    """Indexes a catalog once; ``rank`` is then cheap enough to run inside a hook."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.docs: list[_Doc] = []
        df: Counter = Counter()
        for e in catalog.entries:
            if e.name in SELF_NAMES or catalog.is_removed(e):
                continue
            desc = Counter(tokens(e.description))
            doc = _Doc(e, desc, sum(desc.values()) or 1, set(tokens(f"{e.name} {e.title}")),
                       set(tokens(" ".join(e.tools))), stem((e.category or "").lower()))
            self.docs.append(doc)
            df.update(set(desc) | doc.name_terms | doc.tool_terms)
        n = max(len(self.docs), 1)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.avg_len = sum(d.length for d in self.docs) / n
        self.max_idf = math.log(1 + (n + 0.5) / 0.5)

    def rank(self, need: str, *, limit: int = 3, min_score: float = 0.0,
             exclude: Iterable[str] = ()) -> list[Match]:
        q = query_terms(need)
        if not q:
            return []
        excluded = {x.lower() for x in exclude}
        k1, b = 1.4, 0.75
        out: list[Match] = []
        for d in self.docs:
            if d.entry.name.lower() in excluded:
                continue
            score, hit = 0.0, []
            for term, w in q.items():
                idf = self.idf.get(term, self.max_idf)
                tf = d.desc.get(term, 0)
                part = 0.0
                if tf:
                    part += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * d.length / self.avg_len))
                if term in d.name_terms:
                    part += 2.2 * idf
                if term in d.tool_terms:
                    part += 1.6 * idf
                if term == d.category:
                    part += 0.8 * idf
                if part:
                    score += w * part
                    if w == 1.0:
                        hit.append(term)
            if not hit:  # synonyms alone never qualify an entry
                continue
            coverage = len(hit) / sum(1 for w in q.values() if w == 1.0)
            score *= 0.6 + 0.8 * coverage
            if d.entry.tier == "official":
                score *= 1.15
            elif d.entry.tier == "verified":
                score *= 1.1
            if score >= min_score:
                out.append(Match(d.entry, round(score, 3), hit))
        out.sort(key=lambda m: (-m.score, m.entry.name))
        return out[:limit]


def best(catalog: Optional[Catalog], need: str, **kw) -> list[Match]:
    return Ranker(catalog).rank(need, **kw) if catalog else []
