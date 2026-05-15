"""Configuration loader for the Hindi SLM ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SangrahaSourceConfig(BaseModel):
    enabled: bool = True
    dataset_name: str = "ai4bharat/sangraha"
    data_dir: str = "verified/hin"
    split: str = "train"
    streaming: bool = False
    max_records: int | None = None
    cache_dir: str | None = None


class PdfSourceConfig(BaseModel):
    enabled: bool = True
    input_dir: str = "data/raw/pdf"
    require_metadata_json: bool = True


class WikiSeedConfig(BaseModel):
    """A single Wikipedia article seed for the crawler."""

    url: str
    name: str
    category: str = "general"
    follow_links: bool = True
    max_depth: int = 1
    max_pages: int = 50
    link_include_pattern: str | None = None
    link_exclude_pattern: str | None = None


class WikiSourceConfig(BaseModel):
    """Collection of Wikipedia seed pages to crawl."""

    enabled: bool = False
    seeds: list[WikiSeedConfig] = Field(default_factory=list)


class SourcesConfig(BaseModel):
    sangraha: SangrahaSourceConfig = Field(default_factory=SangrahaSourceConfig)
    pdf: PdfSourceConfig = Field(default_factory=PdfSourceConfig)
    wiki: WikiSourceConfig = Field(default_factory=WikiSourceConfig)


class ProjectConfig(BaseModel):
    name: str = "hindi-slm-corpus"
    corpus_version: str = "hindi_corpus_v001"
    data_root: str = "data"
    log_level: str = "INFO"


class RuntimeConfig(BaseModel):
    batch_size: int = 10000
    random_seed: int = 42
    max_shard_size_mb: int = 512


class PdfExtractionConfig(BaseModel):
    primary_engine: str = "pymupdf"
    fallback_engine: str = "pdfplumber"
    ocr_enabled: bool = False
    extract_by: str = "page"
    preserve_page_number: bool = True
    min_page_text_chars: int = 30
    max_page_text_chars: int = 50000


class GenerationOptions(BaseModel):
    temperature: float = 0.0
    top_p: float = 0.9
    repeat_penalty: float = 1.05


class ChunkingConfig(BaseModel):
    max_input_chars: int = 6000
    overlap_chars: int = 200
    split_on_paragraph: bool = True


class CleaningValidationConfig(BaseModel):
    min_output_char_count: int = 30
    min_output_to_input_length_ratio: float = 0.50
    max_output_to_input_length_ratio: float = 1.20
    min_devanagari_ratio: float = 0.60
    reject_if_output_empty: bool = True
    reject_if_prompt_echo: bool = True
    reject_if_repeated_lines: bool = True


class ModelCleaningConfig(BaseModel):
    enabled: bool = True
    engine: str = "ollama"
    model: str = "qwen3"
    endpoint: str = "http://localhost:11434/api/generate"
    request_timeout_seconds: int = 180
    max_retries: int = 3
    retry_backoff_base_seconds: int = 2
    generation_options: GenerationOptions = Field(default_factory=GenerationOptions)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    validation: CleaningValidationConfig = Field(default_factory=CleaningValidationConfig)


class QualityScoreWeights(BaseModel):
    devanagari_ratio: float = 0.70
    length_score: float = 0.30


class QualityFilterConfig(BaseModel):
    min_devanagari_ratio: float = 0.60
    min_char_count: int = 30
    max_char_count: int = 1_000_000
    reject_table_fragments: bool = True
    max_digit_ratio: float = 0.30
    max_symbol_ratio: float = 0.25
    remove_urls: bool = True
    quality_score_weights: QualityScoreWeights = Field(default_factory=QualityScoreWeights)


class ParquetExportConfig(BaseModel):
    enabled: bool = True
    compression: str = "zstd"
    shard_size_mb: int = 512


class TextExportConfig(BaseModel):
    enabled: bool = True
    compression: str = "gzip"
    one_document_per_line: bool = False
    separator: str = "\n\n"


class JsonlExportConfig(BaseModel):
    enabled: bool = True
    compression: str = "gzip"
    include_metadata: bool = True


class SplitsConfig(BaseModel):
    train: float = 0.98
    validation: float = 0.01
    test: float = 0.01
    split_level: str = "document"
    random_seed: int = 42


class NamingConfig(BaseModel):
    corpus_version: str = "hindi_corpus_v001"


class ExportsConfig(BaseModel):
    parquet: ParquetExportConfig = Field(default_factory=ParquetExportConfig)
    text: TextExportConfig = Field(default_factory=TextExportConfig)
    jsonl: JsonlExportConfig = Field(default_factory=JsonlExportConfig)


class ExportConfig(BaseModel):
    exports: ExportsConfig = Field(default_factory=ExportsConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    naming: NamingConfig = Field(default_factory=NamingConfig)


class WikiCrawlConfig(BaseModel):
    """Settings for the Wikipedia crawler (loaded from wiki_crawl_config.yaml)."""

    api_base_url: str = "https://hi.wikipedia.org/w/api.php"
    wiki_base_url: str = "https://hi.wikipedia.org"
    language: str = "hi"
    user_agent: str = "Hindi-SLM-Corpus-Builder/1.0 (educational research)"
    delay_seconds: float = 1.0
    request_timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_base_seconds: int = 2
    respect_robots: bool = True
    extract_plain_text: bool = True
    min_section_chars: int = 50
    exclude_sections: list[str] = Field(
        default_factory=lambda: ["सन्दर्भ", "बाहरी कड़ियाँ", "इन्हें भी देखें", "टिप्पणियाँ", "ग्रन्थसूची"]
    )
    only_follow_wiki_links: bool = True
    skip_namespaces: list[str] = Field(
        default_factory=lambda: ["Wikipedia", "Talk", "User", "Help", "File", "Template", "Category", "Special", "Portal", "Project"]
    )
    link_include_pattern: str | None = None
    link_exclude_pattern: str | None = None
    raw_output_dir: str = "data/raw/wiki"
    save_raw_responses: bool = True


class IngestionSettings(BaseSettings):
    """Top-level settings loaded from the master ingestion config YAML."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    # Sub-configs loaded separately and merged here
    pdf_extraction: PdfExtractionConfig = Field(default_factory=PdfExtractionConfig)
    model_cleaning: ModelCleaningConfig = Field(default_factory=ModelCleaningConfig)
    quality_filter: QualityFilterConfig = Field(default_factory=QualityFilterConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    wiki_crawl: WikiCrawlConfig = Field(default_factory=WikiCrawlConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(config_path: str | Path) -> IngestionSettings:
    """Load and merge all YAML config files relative to the master config location."""
    config_path = Path(config_path)
    config_dir = config_path.parent

    master = _load_yaml(config_path)

    sub_files = {
        "pdf_extraction": "pdf_extraction_config.yaml",
        "model_cleaning": "model_cleaning_config.yaml",
        "quality_filter": "quality_filter_config.yaml",
        "export": "export_config.yaml",
        "wiki_crawl": "wiki_crawl_config.yaml",
    }

    merged: dict[str, Any] = {**master}
    for key, filename in sub_files.items():
        sub_path = config_dir / filename
        if sub_path.exists():
            sub_data = _load_yaml(sub_path)
            merged[key] = sub_data.get(key, sub_data)

    return IngestionSettings(**merged)
