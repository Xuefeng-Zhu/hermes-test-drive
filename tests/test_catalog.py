import email.message
import io
import json
import urllib.error

import pytest

from conftest import sample_bytes
from test_drive import catalog
from test_drive.catalog import CATALOG_URL, CatalogError, CatalogStore, host_allowed, parse_document


def test_parse_real_catalog_sample():
    cat = parse_document(json.loads(sample_bytes()))
    e = cat.get("global-business-culture")
    assert e and e.tier == "community" and len(e.sha) == 40
    assert e.tools == ["culture_brief", "culture_gap_analysis", "culture_countries"]
    assert cat.get("ABSTRACT-PROVIDER").requires_env == ["ABSTRACT_API_KEY"]
    assert "or-picker" in cat.removed


def test_parse_rejects_garbage():
    with pytest.raises(CatalogError):
        parse_document({"entries": "nope"})
    with pytest.raises(CatalogError):
        parse_document({"entries": [{"no": "name"}]})


@pytest.mark.parametrize("url,ok", [
    ("https://hermes-agent.nousresearch.com/docs/api/plugin-catalog.json", True),
    ("https://nousresearch.github.io/hermes-agent/docs/api/plugin-catalog.json", True),
    ("https://cdn.nousresearch.com/x.json", True),
    ("http://hermes-agent.nousresearch.com/docs/api/plugin-catalog.json", False),
    ("https://evil.github.io/plugin-catalog.json", False),
    ("https://nousresearch.com.evil.io/x", False),
    ("https://raw.githubusercontent.com/NousResearch/hermes-agent/main/x.json", False),
])
def test_host_allowlist(url, ok):
    assert host_allowed(url) is ok


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _redirect(url, to):
    headers = email.message.Message()
    headers["Location"] = to
    return urllib.error.HTTPError(url, 301, "Moved", headers, None)


def _opener(monkeypatch, plan):
    seen = []

    class Opener:
        def open(self, req, timeout=None):
            seen.append(req.full_url)
            step = plan[len(seen) - 1]
            if isinstance(step, Exception):
                raise step
            return _Resp(step)

    monkeypatch.setattr(catalog.urllib.request, "build_opener", lambda *a: Opener())
    return seen


def test_follows_redirect_to_nous_mirror(monkeypatch):
    mirror = "https://nousresearch.github.io/hermes-agent/docs/api/plugin-catalog.json"
    seen = _opener(monkeypatch, [_redirect(CATALOG_URL, mirror), b'{"entries": []}'])
    assert catalog.http_get(CATALOG_URL) == b'{"entries": []}'
    assert seen == [CATALOG_URL, mirror]


def test_refuses_redirect_off_nous_hosts(monkeypatch):
    seen = _opener(monkeypatch, [_redirect(CATALOG_URL, "https://evil.example.com/catalog.json"), b"{}"])
    with pytest.raises(CatalogError, match="non-Nous"):
        catalog.http_get(CATALOG_URL)
    assert seen == [CATALOG_URL]


def test_refuses_https_downgrade_and_redirect_loops(monkeypatch):
    _opener(monkeypatch, [_redirect(CATALOG_URL, "http://hermes-agent.nousresearch.com/x")])
    with pytest.raises(CatalogError):
        catalog.http_get(CATALOG_URL)
    _opener(monkeypatch, [_redirect(CATALOG_URL, CATALOG_URL)] * 5)
    with pytest.raises(CatalogError, match="too many redirects"):
        catalog.http_get(CATALOG_URL)


def test_refuses_oversized_payload(monkeypatch):
    _opener(monkeypatch, [b"x" * 101])
    with pytest.raises(CatalogError, match="too large"):
        catalog.http_get(CATALOG_URL, max_bytes=100)


def test_store_caches_with_ttl(tmp_path):
    now = [1000.0]
    calls = []

    def fetch(url):
        calls.append(url)
        return sample_bytes()

    s = CatalogStore(lambda: tmp_path, ttl_s=60, fetcher=fetch, clock=lambda: now[0])
    assert s.load().source == "live" and len(calls) == 1
    assert s.load().get("jalancost") and len(calls) == 1          # memory hit
    fresh = CatalogStore(lambda: tmp_path, ttl_s=60, fetcher=fetch, clock=lambda: now[0])
    assert fresh.load().source == "cache" and len(calls) == 1      # disk hit, no network
    assert fresh.load().get("jalancost").tools == ["estimate_road_trip_cost"]
    now[0] += 61
    assert fresh.load().source == "live" and len(calls) == 2       # expired


def test_store_falls_back_to_stale_cache_then_errors(tmp_path):
    ok = CatalogStore(lambda: tmp_path, ttl_s=1, fetcher=lambda u: sample_bytes(), clock=lambda: 0.0)
    ok.load()

    def boom(url):
        raise CatalogError("offline")

    stale = CatalogStore(lambda: tmp_path, ttl_s=1, fetcher=boom, clock=lambda: 10_000.0)
    cat = stale.load()
    assert cat.source == "stale-cache" and cat.get("weather")
    empty = CatalogStore(lambda: tmp_path / "nothing", fetcher=boom)
    with pytest.raises(CatalogError, match="offline"):
        empty.load()
    assert empty.cached() is None


def test_cached_never_fetches(tmp_path):
    def boom(url):
        raise AssertionError("network in a hook")

    assert CatalogStore(lambda: tmp_path, fetcher=boom).cached() is None
