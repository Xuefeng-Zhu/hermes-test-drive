import json
from pathlib import Path

import pytest

from conftest import FIXTURES, sample_bytes
from test_drive import trust
from test_drive.catalog import Entry, parse_document
from test_drive.ranking import Ranker, query_terms, tokens


@pytest.fixture(scope="module")
def cat():
    return parse_document(json.loads(sample_bytes()))


@pytest.mark.parametrize("need,top", [
    ("culture gap analysis between Germany and Japan for my team", "global-business-culture"),
    ("Malaysia road trip fuel and toll cost split", "jalancost"),
    ("check my child's vaccination card for missing doses", "hathor-catchup"),
    ("what's the weather forecast in Seattle tomorrow", "weather"),
    ("convert a PDF to markdown", None),  # either doc tool is fine; asserted below
    ("track how much I spend on model API calls", None),
])
def test_rank_top_hit(cat, need, top):
    matches = Ranker(cat).rank(need)
    assert matches, need
    if top:
        assert matches[0].entry.name == top
    assert len(matches) <= 3
    assert all(m.score > 0 and m.matched for m in matches)


def test_rank_pdf_and_spend(cat):
    assert Ranker(cat).rank("convert a PDF to markdown")[0].entry.name in {"doc-markdown", "tabella"}
    assert Ranker(cat).rank("track how much I spend on model API calls")[0].entry.name in {
        "cost-sentinel", "budget-leash"}


def test_synonyms_alone_do_not_qualify(cat):
    # "forecast" is only a synonym hop from weather; a need of pure stopwords returns nothing.
    assert Ranker(cat).rank("please help me with something") == []
    assert query_terms("email")["mail"] == 0.5


def test_removed_and_self_are_excluded(cat):
    cat.entries.append(Entry(name="test-drive", description="test drive catalog plugins sandbox"))
    cat.entries.append(Entry(name="or-picker", description="openrouter model picker"))
    names = {m.entry.name for m in Ranker(cat).rank("test drive openrouter picker", limit=10)}
    assert "test-drive" not in names and "or-picker" not in names


def test_exclude_param(cat):
    assert all(m.entry.name != "weather" for m in Ranker(cat).rank("weather forecast", exclude=["weather"]))


def test_tokens_stem_and_drop_stopwords():
    assert tokens("Checking the Vaccines schedules") == ["check", "vaccine", "schedule"]


def _yaml_entry(name):
    yaml = pytest.importorskip("ruamel.yaml")
    data = yaml.YAML(typ="safe").load((FIXTURES / f"{name}.yaml").read_text())
    return Entry.from_mapping(data)


def test_trust_card_from_catalog_yaml_no_keys(cat):
    e = _yaml_entry("global-business-culture")
    text = trust.card(e, cat, env=trust.env_check(e, lambda n: False), rank=1, why=["culture", "gap"])
    assert "**1. Global Business Culture** (`global-business-culture`) · community · by pswong661-eng" in text
    assert "merged in the official catalog; pinned 1.0.0 @ 49218a35" in text
    assert "tools culture_brief, culture_gap_analysis, culture_countries" in text
    assert "Keys needed: none" in text
    assert "no network, shell or storage" in text
    assert "Known issue: Scale positions" in text and "Matched on: culture, gap" in text


def test_trust_card_keys_and_disclosure(cat):
    e = _yaml_entry("abstract-provider")
    env = trust.env_check(e, lambda n: False)
    assert env.missing == ["ABSTRACT_API_KEY"]
    text = trust.card(e, cat, env=env)
    assert "ABSTRACT_API_KEY ✗ missing" in text
    assert "Disclosure: “registers one model provider" in text
    assert "no tools/hooks" in text
    d = trust.card_dict(e, cat, env)
    assert d["env_missing"] == ["ABSTRACT_API_KEY"] and d["declared"]["network"] is True


def test_trust_card_hooks_and_flags(cat):
    e = _yaml_entry("hathor-catchup")
    text = trust.card(e, cat)
    assert "hooks pre_llm_call, post_tool_call, transform_llm_output" in text
    assert "Hermes >=0.23.0" in text
    assert trust.declared_flags(e)["network"] is False  # "no network calls"
    nomads = _yaml_entry("nomads-planner")
    assert trust.declared_flags(nomads)["network"] is True


def test_trust_card_marks_removed(cat):
    e = Entry(name="or-picker", sha="a" * 40, repo="https://github.com/x/y")
    assert "REMOVED from the catalog" in trust.card(e, cat)


def test_yaml_and_published_json_agree(cat):
    for name in ("global-business-culture", "jalancost", "hathor-catchup", "weather"):
        y, j = _yaml_entry(name), cat.get(name)
        assert (y.sha, y.tools, y.hooks, y.requires_env, y.tier) == (j.sha, j.tools, j.hooks, j.requires_env, j.tier)
