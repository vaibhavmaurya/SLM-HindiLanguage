"""Unit tests for WikiCrawler."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slm_hindi.config.settings import WikiCrawlConfig, WikiSeedConfig
from slm_hindi.ingestion.wiki_crawler import WikiCrawler, _clean_extract, _title_from_url

FIXTURES = Path(__file__).parent.parent / "fixtures" / "sample_wiki"


@pytest.fixture()
def wiki_config() -> WikiCrawlConfig:
    return WikiCrawlConfig(
        delay_seconds=0.0,
        max_retries=1,
        save_raw_responses=False,
        exclude_sections=["सन्दर्भ", "बाहरी कड़ियाँ"],
        skip_namespaces=["Wikipedia", "Talk", "Help"],
    )


@pytest.fixture()
def seed() -> WikiSeedConfig:
    return WikiSeedConfig(
        url="https://hi.wikipedia.org/wiki/%E0%A4%AE%E0%A4%B9%E0%A4%BE%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4",
        name="mahabharata",
        category="epic_literature",
        follow_links=False,
        max_depth=0,
        max_pages=5,
    )


@pytest.fixture()
def extract_response() -> dict:
    return json.loads((FIXTURES / "sample_extract_response.json").read_text(encoding="utf-8"))


@pytest.fixture()
def links_response() -> dict:
    return json.loads((FIXTURES / "sample_links_response.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# _title_from_url
# ---------------------------------------------------------------------------

class TestTitleFromUrl:
    def test_encoded_hindi_url(self) -> None:
        url = "https://hi.wikipedia.org/wiki/%E0%A4%AE%E0%A4%B9%E0%A4%BE%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4"
        assert _title_from_url(url) == "महाभारत"

    def test_underscore_to_space(self) -> None:
        url = "https://hi.wikipedia.org/wiki/हिंदी_भाषा"
        assert _title_from_url(url) == "हिंदी भाषा"

    def test_no_wiki_prefix(self) -> None:
        url = "https://hi.wikipedia.org/महाभारत"
        assert _title_from_url(url) == "महाभारत"


# ---------------------------------------------------------------------------
# _clean_extract
# ---------------------------------------------------------------------------

class TestCleanExtract:
    def test_removes_excluded_section(self) -> None:
        text = "मुख्य सामग्री यहाँ है।\n\n== सन्दर्भ ==\nकुछ सन्दर्भ।"
        result = _clean_extract(text, ["सन्दर्भ"])
        assert "सन्दर्भ" not in result
        assert "मुख्य सामग्री" in result

    def test_strips_heading_markers(self) -> None:
        text = "== परिचय ==\nयह एक परिचय है।"
        result = _clean_extract(text, [])
        assert "==" not in result
        assert "परिचय" in result

    def test_collapses_blank_lines(self) -> None:
        text = "पहला\n\n\n\nदूसरा"
        result = _clean_extract(text, [])
        assert "\n\n\n" not in result

    def test_empty_input(self) -> None:
        assert _clean_extract("", []) == ""

    def test_no_excluded_sections(self) -> None:
        text = "कुछ पाठ है।\n\n== अध्याय ==\nअध्याय सामग्री।"
        result = _clean_extract(text, [])
        assert "अध्याय सामग्री" in result


# ---------------------------------------------------------------------------
# WikiCrawler
# ---------------------------------------------------------------------------

class TestWikiCrawlerFetchExtract:
    def test_returns_text_and_url(self, wiki_config, extract_response) -> None:
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        with patch.object(crawler, "_api_get", return_value=extract_response):
            text, url = crawler._fetch_extract("महाभारत")
        assert "महाभारत" in text
        assert "hi.wikipedia.org/wiki/महाभारत" in url

    def test_raises_on_missing_page(self, wiki_config) -> None:
        from slm_hindi.ingestion.wiki_crawler import WikiCrawlError
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        missing = {"query": {"pages": {"-1": {"ns": 0, "title": "Missing"}}}}
        with patch.object(crawler, "_api_get", return_value=missing):
            with pytest.raises(WikiCrawlError, match="Page not found"):
                crawler._fetch_extract("Missing")


class TestWikiCrawlerFetchLinks:
    def test_returns_namespace_zero_titles(self, wiki_config, links_response) -> None:
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        with patch.object(crawler, "_api_get", return_value=links_response):
            titles = crawler._fetch_links("महाभारत")
        assert "कुरुक्षेत्र" in titles
        assert "पाण्डव" in titles

    def test_pagination_followed(self, wiki_config) -> None:
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        page1 = {
            "query": {"pages": {"1": {"links": [{"title": "A"}]}}},
            "continue": {"plcontinue": "12345|0|B"},
        }
        page2 = {"query": {"pages": {"1": {"links": [{"title": "B"}]}}}}
        with patch.object(crawler, "_api_get", side_effect=[page1, page2]):
            titles = crawler._fetch_links("Test")
        assert "A" in titles and "B" in titles


class TestWikiCrawlerFilterLinks:
    def test_skips_namespace_prefixes(self, wiki_config, seed) -> None:
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        titles = ["Wikipedia:नीतियाँ", "Talk:कुरुक्षेत्र", "पाण्डव"]
        result = crawler._filter_links(titles, seed)
        assert result == ["पाण्डव"]

    def test_global_exclude_pattern(self, wiki_config, seed) -> None:
        wiki_config.link_exclude_pattern = r"^\d"
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        titles = ["123_article", "महाभारत"]
        result = crawler._filter_links(titles, seed)
        assert "123_article" not in result
        assert "महाभारत" in result

    def test_seed_include_pattern(self, wiki_config, seed) -> None:
        seed.link_include_pattern = "पाण्डव"
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        titles = ["पाण्डव", "कौरव"]
        result = crawler._filter_links(titles, seed)
        assert result == ["पाण्डव"]

    def test_seed_exclude_pattern(self, wiki_config, seed) -> None:
        seed.link_exclude_pattern = "कौरव"
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        titles = ["पाण्डव", "कौरव"]
        result = crawler._filter_links(titles, seed)
        assert "कौरव" not in result


class TestWikiCrawlerProcessPage:
    def test_produces_corpus_records(self, wiki_config, seed, extract_response) -> None:
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        with patch.object(crawler, "_api_get", return_value=extract_response):
            records = crawler._process_page("महाभारत", seed, depth=0)
        assert len(records) >= 1
        assert all(r.source_type == "wiki" for r in records)
        assert all(r.language == "hi" for r in records)
        assert all(r.ingestion_run_id == "test-run" for r in records)

    def test_skips_short_paragraphs(self, wiki_config, seed) -> None:
        wiki_config.min_section_chars = 500
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        with patch.object(crawler, "_api_get", return_value={
            "query": {"pages": {"1": {"title": "Test", "extract": "छोटा पाठ।"}}}
        }):
            records = crawler._process_page("Test", seed, depth=0)
        assert records == []

    def test_skips_missing_page(self, wiki_config, seed) -> None:
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        missing = {"query": {"pages": {"-1": {"ns": 0, "title": "Missing"}}}}
        with patch.object(crawler, "_api_get", return_value=missing):
            records = crawler._process_page("Missing", seed, depth=0)
        assert records == []


class TestWikiCrawlerRetry:
    def test_retries_on_timeout(self, wiki_config, extract_response) -> None:
        import requests as req
        wiki_config.max_retries = 2
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = extract_response
        with patch.object(crawler._session, "get", side_effect=[req.Timeout(), mock_resp]):
            data = crawler._api_get({"titles": "महाभारत"})
        assert "query" in data

    def test_raises_after_max_retries(self, wiki_config) -> None:
        from slm_hindi.ingestion.wiki_crawler import WikiCrawlError
        import requests as req
        wiki_config.max_retries = 2
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        with patch.object(crawler._session, "get", side_effect=req.Timeout()):
            with pytest.raises(WikiCrawlError, match="unreachable"):
                crawler._api_get({"titles": "Test"})


class TestWikiCrawlerCrawlSeed:
    def test_bfs_respects_max_pages(self, wiki_config, seed, extract_response, links_response) -> None:
        seed.max_pages = 2
        seed.follow_links = True
        seed.max_depth = 1
        crawler = WikiCrawler(wiki_config, run_id="test-run")

        def api_get(params):
            if params.get("prop") == "links":
                return links_response
            return extract_response

        with patch.object(crawler, "_api_get", side_effect=api_get):
            records = crawler.crawl_seed(seed)
        assert isinstance(records, list)

    def test_progress_callback_called(self, wiki_config, seed, extract_response) -> None:
        seed.follow_links = False
        seed.max_pages = 1
        crawler = WikiCrawler(wiki_config, run_id="test-run")
        calls: list[int] = []
        with patch.object(crawler, "_api_get", return_value=extract_response):
            crawler.crawl_seed(seed, progress_callback=lambda n: calls.append(n))
        assert len(calls) >= 1
