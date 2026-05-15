"""Hindi Wikipedia crawler using the MediaWiki Action API.

Crawls from seed URLs using BFS, extracts plain-text content per article,
and produces CorpusRecord objects ready for the standard pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from urllib.parse import unquote, urlparse

import requests

from slm_hindi.config.settings import WikiCrawlConfig, WikiSeedConfig
from slm_hindi.schema.corpus_record import CorpusRecord

if TYPE_CHECKING:
    from slm_hindi.observability.file_registry import FileRegistry
    from slm_hindi.observability.run_logger import IngestionRunLogger

logger = logging.getLogger(__name__)

_PHASE = "wiki_crawl"
_COMPONENT = "wiki_crawler"
_MW_API_PARAMS_EXTRACT = {
    "action": "query",
    "format": "json",
    "prop": "extracts",
    "explaintext": "1",
    "exsectionformat": "plain",
    "redirects": "1",
}
_MW_API_PARAMS_LINKS = {
    "action": "query",
    "format": "json",
    "prop": "links",
    "pllimit": "500",
    "plnamespace": "0",
    "redirects": "1",
}


def _title_from_url(url: str) -> str:
    """Extract and URL-decode the article title from a Hindi Wikipedia URL."""
    parsed = urlparse(url)
    path = parsed.path
    if "/wiki/" in path:
        raw = path.split("/wiki/", 1)[1]
        return unquote(raw).replace("_", " ")
    return unquote(path.lstrip("/")).replace("_", " ")


def _clean_extract(text: str, exclude_sections: list[str]) -> str:
    """Remove excluded section headers and their content from plain-text extract."""
    if not text:
        return ""

    # Section headers from explaintext look like "\n\n== Section ==\n"
    # Build a pattern that matches any excluded section heading
    if exclude_sections:
        escaped = [re.escape(s) for s in exclude_sections]
        section_pat = re.compile(
            r"\n==+\s*(?:" + "|".join(escaped) + r")\s*==+\n.*?(?=\n==|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        text = section_pat.sub("", text)

    # Remove remaining == heading markers, keep the heading text
    text = re.sub(r"==+\s*(.*?)\s*==+", r"\1", text)
    # Collapse runs of blank lines to two newlines max
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class WikiCrawlError(RuntimeError):
    pass


class WikiCrawler:
    """BFS crawler for Hindi Wikipedia using the MediaWiki Action API."""

    def __init__(self, config: WikiCrawlConfig, run_id: str = "") -> None:
        self._cfg = config
        self._run_id = run_id or str(uuid.uuid4())
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": config.user_agent})

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def crawl_seed(
        self,
        seed: WikiSeedConfig,
        run_logger: IngestionRunLogger | None = None,
        file_registry: FileRegistry | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[CorpusRecord]:
        """BFS-crawl from a single seed and return all extracted CorpusRecords."""
        start_title = _title_from_url(seed.url)
        logger.info("Starting wiki crawl: seed=%r depth=%d max_pages=%d", start_title, seed.max_depth, seed.max_pages)

        if run_logger:
            run_logger.log_event(
                phase=_PHASE, component=_COMPONENT, status="started", source_id=seed.name
            )

        visited: set[str] = set()
        # Queue entries: (title, current_depth)
        queue: deque[tuple[str, int]] = deque([(start_title, 0)])
        records: list[CorpusRecord] = []

        while queue and len(visited) < seed.max_pages:
            title, depth = queue.popleft()
            if title in visited:
                continue
            visited.add(title)

            page_records = self._process_page(title, seed, depth)
            records.extend(page_records)

            if progress_callback:
                progress_callback(1)

            if depth < seed.max_depth and seed.follow_links:
                linked_titles = self._fetch_links(title)
                filtered = self._filter_links(linked_titles, seed)
                for lt in filtered:
                    if lt not in visited:
                        queue.append((lt, depth + 1))

            # Polite delay between requests
            if queue:
                time.sleep(self._cfg.delay_seconds)

        logger.info(
            "Wiki crawl complete: seed=%r pages=%d records=%d",
            seed.name, len(visited), len(records),
        )

        if self._cfg.save_raw_responses:
            self._save_crawl_index(seed, list(visited), file_registry)

        if run_logger:
            run_logger.log_event(
                phase=_PHASE,
                component=_COMPONENT,
                status="completed",
                source_id=seed.name,
                records_out=len(records),
                notes=f"pages_crawled={len(visited)},max_depth={seed.max_depth}",
            )

        return records

    def crawl_all_seeds(
        self,
        seeds: list[WikiSeedConfig],
        run_logger: IngestionRunLogger | None = None,
        file_registry: FileRegistry | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[CorpusRecord]:
        """Crawl every seed in the source config and combine results."""
        all_records: list[CorpusRecord] = []
        for seed in seeds:
            seed_records = self.crawl_seed(
                seed,
                run_logger=run_logger,
                file_registry=file_registry,
                progress_callback=progress_callback,
            )
            all_records.extend(seed_records)
        return all_records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_page(self, title: str, seed: WikiSeedConfig, depth: int) -> list[CorpusRecord]:
        """Fetch one page, extract text, split into paragraph records."""
        try:
            raw_extract, page_url = self._fetch_extract(title)
        except WikiCrawlError as exc:
            logger.warning("Skipping page %r: %s", title, exc)
            return []

        clean_text = _clean_extract(raw_extract, self._cfg.exclude_sections)
        if not clean_text:
            return []

        paragraphs = [p.strip() for p in clean_text.split("\n\n") if len(p.strip()) >= self._cfg.min_section_chars]
        records: list[CorpusRecord] = []
        doc_id = f"wiki_{seed.name}_{title.replace(' ', '_')}"

        for i, para in enumerate(paragraphs):
            rec = CorpusRecord(
                record_id=f"{doc_id}_p{i:04d}",
                document_id=doc_id,
                paragraph_id=f"p{i:04d}",
                source_type="wiki",
                source_name=f"hindi_wikipedia_{seed.name}",
                source_dataset=None,
                source_file_name=None,
                source_url_or_path=page_url,
                raw_text=para,
                final_text=para,
                language="hi",
                script="Devanagari",
                char_count=len(para),
                word_count=len(para.split()),
                estimated_token_count=max(1, int(len(para) / 4.5)),
                cleaning_method="deterministic_normalization",
                cleaning_model=None,
                cleaning_status="pending",
                ingestion_run_id=self._run_id,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            records.append(rec)

        logger.debug("Extracted %d paragraphs from %r (depth=%d)", len(records), title, depth)
        return records

    def _fetch_extract(self, title: str) -> tuple[str, str]:
        """Fetch plain-text extract for a Wikipedia article title.

        Returns (plain_text, canonical_url).
        """
        params = {**_MW_API_PARAMS_EXTRACT, "titles": title}
        data = self._api_get(params)
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                raise WikiCrawlError(f"Page not found: {title!r}")
            extract = page.get("extract", "")
            canonical_title = page.get("title", title)
            page_url = f"{self._cfg.wiki_base_url}/wiki/{canonical_title.replace(' ', '_')}"
            return extract, page_url
        raise WikiCrawlError(f"No pages returned for: {title!r}")

    def _fetch_links(self, title: str) -> list[str]:
        """Return titles of articles linked from the given page (namespace 0 only)."""
        params = {**_MW_API_PARAMS_LINKS, "titles": title}
        titles: list[str] = []
        while True:
            data = self._api_get(params)
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                for link in page.get("links", []):
                    titles.append(link["title"])
            cont = data.get("continue", {})
            if not cont:
                break
            params.update(cont)
        return titles

    def _filter_links(self, titles: list[str], seed: WikiSeedConfig) -> list[str]:
        """Apply namespace and pattern filters to a list of linked titles."""
        result: list[str] = []
        skip_ns = self._cfg.skip_namespaces
        inc_pat = re.compile(seed.link_include_pattern) if seed.link_include_pattern else None
        exc_pat = re.compile(seed.link_exclude_pattern) if seed.link_exclude_pattern else None
        global_exc = re.compile(self._cfg.link_exclude_pattern) if self._cfg.link_exclude_pattern else None

        for title in titles:
            # Skip if matches any namespace prefix
            if any(title.startswith(f"{ns}:") for ns in skip_ns):
                continue
            # Skip if matches global exclude pattern
            if global_exc and global_exc.search(title):
                continue
            # Skip if must match include pattern but doesn't
            if inc_pat and not inc_pat.search(title):
                continue
            # Skip if matches per-seed exclude pattern
            if exc_pat and exc_pat.search(title):
                continue
            result.append(title)
        return result

    def _api_get(self, params: dict) -> dict:
        """GET the MediaWiki API with retry logic."""
        backoff = self._cfg.retry_backoff_base_seconds
        last_exc: Exception | None = None
        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                resp = self._session.get(
                    self._cfg.api_base_url,
                    params=params,
                    timeout=self._cfg.request_timeout_seconds,
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                logger.warning("API request failed (attempt %d/%d): %s", attempt, self._cfg.max_retries, exc)
                if attempt < self._cfg.max_retries:
                    time.sleep(backoff * (2 ** (attempt - 1)))
            except requests.HTTPError as exc:
                raise WikiCrawlError(f"HTTP error from MediaWiki API: {exc}") from exc
        raise WikiCrawlError(f"API unreachable after {self._cfg.max_retries} attempts") from last_exc

    def _save_crawl_index(
        self,
        seed: WikiSeedConfig,
        visited_titles: list[str],
        file_registry: FileRegistry | None,
    ) -> None:
        """Write a JSON index of crawled pages for reproducibility."""
        out_dir = Path(self._cfg.raw_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = out_dir / f"{seed.name}_crawl_index.json"
        index = {
            "seed_name": seed.name,
            "seed_url": seed.url,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "pages_crawled": len(visited_titles),
            "titles": sorted(visited_titles),
        }
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved crawl index: %s", index_path)
        if file_registry:
            file_registry.register_file(
                index_path,
                role="report",
                stage=_PHASE,
                source_id=seed.name,
                file_format="json",
                row_count=len(visited_titles),
            )
