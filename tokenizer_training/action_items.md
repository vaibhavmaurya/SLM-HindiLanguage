# Action Items: Hindi SLM Tokenizer Training Workstream

Each phase ends with a full test run before proceeding. Test cases are written before production code (TDD).

---

## Phase 1 — Scaffold & Tooling

**Goal:** Empty but correctly structured project that collects with zero errors.

**Acceptance criteria:**
- `pytest --collect-only` reports 0 errors
- `pip install -e ".[dev]"` completes without error
- `ruff check src/ tests/` is clean

**Tasks:**

1. Create directory tree:
   ```
   tokenizer_training/
   ├── configs/
   ├── src/hindi_tokenizer/
   │   ├── config/
   │   ├── schema/
   │   ├── corpus/
   │   ├── training/
   │   ├── validation/
   │   ├── packaging/
   │   ├── publishing/
   │   ├── sdk/
   │   ├── observability/
   │   ├── orchestration/
   │   └── ui/
   ├── tests/
   │   ├── fixtures/
   │   │   ├── sample_parquet/
   │   │   ├── sample_text/
   │   │   ├── sample_configs/
   │   │   └── validation_sentences.txt
   │   ├── unit/
   │   └── integration/
   └── data/   ← gitignored
   ```

2. Create `pyproject.toml`:
   - Build system: `hatchling`
   - Package: `hindi_tokenizer` from `src/`
   - pytest config: `testpaths = ["tests"]`, `pythonpath = ["src"]`
   - Coverage: `source = ["hindi_tokenizer"]`
   - Markers: `requires_hf_hub`
   - Ruff config: line-length 120, target-version py311

3. Create `requirements.txt`:
   ```
   tokenizers>=0.19
   transformers>=4.40
   huggingface_hub>=0.22
   pyarrow>=15.0
   pandas>=2.0
   pydantic>=2.0
   pydantic-settings>=2.0
   pyyaml>=6.0
   typer>=0.12
   rich>=13.0
   ```

4. Create `requirements-dev.txt`:
   ```
   pytest>=8.0
   pytest-cov>=5.0
   pytest-mock>=3.14
   ruff>=0.4
   ```

5. Create `__init__.py` files for every package directory under `src/hindi_tokenizer/`.

6. Create `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`.

7. Create `tests/conftest.py` with:
   - `pytest.ini_options` marker registration for `requires_hf_hub`
   - Shared `fixtures_dir` path fixture

8. Create `tests/fixtures/validation_sentences.txt` with ~20 Hindi sentences covering:
   - Pure Devanagari prose
   - Hindi with embedded English acronyms (UPI, GST, ISRO)
   - Numbers and digits
   - Punctuation including danda `।` and double danda `॥`
   - Quoted speech
   - Long compound words

9. Create small fixture Parquet files in `tests/fixtures/sample_parquet/`:
   - `sample_01.parquet` — 50 rows, columns: `record_id`, `final_text`, `source_type`
   - `sample_02.parquet` — 50 rows, same schema
   - Include a mix of valid Hindi text, short records, records with low Devanagari ratio

10. Create `tests/fixtures/sample_text/small_corpus.txt` — 200 lines of Hindi text for trainer unit tests (enough for a very small vocab).

11. Create `tests/fixtures/sample_configs/test_tokenizer_training_config.yaml` — mirrors production config with paths pointing to fixture directories.

12. Create `Makefile` with targets: `make test`, `make lint`, `make lint-fix`, `make run`, `make run-sample`, `make run-train`, `make run-validate`.

13. Create `run_tests.sh` and `run_tests.bat`.

14. Create `.env.example`:
    ```
    HF_TOKEN=your_huggingface_token
    TOKENIZER_VERSION=hindi_slm_tokenizer_v001
    DATA_ROOT=data
    ```

15. Create `.gitignore` covering `data/`, `.venv/`, `__pycache__/`, `*.pyc`, `dist/`, `.env`.

16. Create `CLAUDE.md` documenting tech stack, directory layout, run commands, code style.

**Verify:** `pytest --collect-only` — 0 errors, 0 tests collected (stubs only).

---

## Phase 2 — Config & Schema Layer

**Goal:** Type-safe config loading and Pydantic schemas for all structured data.

**Acceptance criteria:**
- Config loads from YAML, validates at startup, raises on bad values
- All schema models instantiate correctly from sample dicts
- `pytest tests/unit/test_settings.py tests/unit/test_records.py -v` — all green

### TDD Test Cases

#### `tests/unit/test_settings.py`

| Test | Description |
|---|---|
| `test_load_config_from_yaml` | Load `test_tokenizer_training_config.yaml` via `load_settings()`; assert `settings.project.name == "hindi-slm-tokenizer"` |
| `test_default_vocab_size_is_32k` | `settings.tokenizer.default_vocab_size == 32000` |
| `test_vocab_sizes_contains_three_variants` | `settings.tokenizer.vocab_sizes == [24000, 32000, 48000]` |
| `test_min_char_count_default` | `settings.text_filters.min_char_count == 30` |
| `test_max_char_count_default` | `settings.text_filters.max_char_count == 5000` |
| `test_min_devanagari_ratio_default` | `settings.text_filters.min_devanagari_ratio == 0.60` |
| `test_special_tokens_count` | `settings.special_tokens` contains exactly 8 tokens (pad, unk, bos, eos + 4 reserved) |
| `test_pad_token_value` | `settings.special_tokens.pad_token == "<pad>"` |
| `test_random_seed_default` | `settings.sampling.random_seed == 42` |
| `test_algorithm_is_unigram` | `settings.tokenizer.algorithm == "unigram"` |
| `test_normalizer_is_nfkc` | `settings.tokenizer.normalizer == "nfkc"` |
| `test_pre_tokenizer_is_metaspace` | `settings.tokenizer.pre_tokenizer == "metaspace"` |
| `test_validation_max_unk_rate` | `settings.validation.thresholds.max_unk_rate == 0.001` |
| `test_validation_min_roundtrip` | `settings.validation.thresholds.min_roundtrip_success_rate == 0.99` |
| `test_missing_required_field_raises` | Remove `project.name` from test YAML; assert `ValidationError` raised on load |
| `test_env_var_overrides_yaml` | Set `TOKENIZER_VERSION=test_v999` env var; assert `settings.project.tokenizer_version == "test_v999"` |

#### `tests/unit/test_records.py`

| Test | Description |
|---|---|
| `test_sample_manifest_instantiates` | Construct `SampleManifest` from valid dict; assert all fields populated |
| `test_sample_manifest_created_at_populated` | `manifest.created_at` is a non-empty ISO string |
| `test_validation_report_instantiates` | Construct `ValidationReport` with all required fields; no exception |
| `test_validation_report_passes_thresholds_true` | Set metrics within thresholds; `passes_thresholds == True` |
| `test_validation_report_passes_thresholds_false` | Set `unk_rate = 0.05` (above 0.001 threshold); `passes_thresholds == False` |
| `test_validation_report_roundtrip_field` | `roundtrip_success_rate` field stores float between 0.0 and 1.0 |
| `test_comparison_result_instantiates` | Construct `ComparisonResult` with three `ValidationReport` items |
| `test_comparison_result_recommended_variant` | `recommended_variant` must be one of the variant names in `variants` list |
| `test_records_reject_bare_dict` | Passing a bare `dict` where `ValidationReport` is expected raises `TypeError` |

**Tasks:**
1. Create `src/hindi_tokenizer/config/settings.py` — `TokenizerSettings` with all sub-models.
2. Create `configs/tokenizer_training_config.yaml`, `tokenizer_validation_config.yaml`, `publish_config.yaml`.
3. Create `src/hindi_tokenizer/schema/records.py` — `SampleManifest`, `ValidationReport`, `ComparisonResult`.
4. Create `src/hindi_tokenizer/ui/progress.py` — singleton `Console`, `make_progress()`, `setup_logging()`.

**Verify:** `pytest tests/unit/test_settings.py tests/unit/test_records.py -v --cov=src/hindi_tokenizer`

---

## Phase 3 — Corpus Reader & Sampler

**Goal:** Read text from Parquet files, filter to high-quality Hindi, write plain-text sample for tokenizer training.

**Acceptance criteria:**
- `ParquetReader` correctly iterates text from fixture Parquet files
- `CorpusSampler` filters, normalizes, shuffles, and writes a deterministic plain-text output
- `pytest tests/unit/test_parquet_reader.py tests/unit/test_corpus_sampler.py -v` — all green

### TDD Test Cases

#### `tests/unit/test_parquet_reader.py`

| Test | Description |
|---|---|
| `test_read_single_file_returns_strings` | `ParquetReader.read_texts(fixture_parquet)` returns a list where every element is `str` |
| `test_read_single_file_count` | Returns 50 texts from `sample_01.parquet` |
| `test_read_multiple_files` | Reading both fixture files returns 100 texts total |
| `test_read_uses_configured_text_column` | With `text_column="final_text"`, reads `final_text`; with wrong column name raises `ValueError` |
| `test_read_skips_null_values` | Parquet rows with `null` in text column are silently skipped |
| `test_read_skips_non_string_values` | Parquet rows where text column is an integer are skipped |
| `test_read_empty_parquet_returns_empty` | Empty Parquet file returns empty list |
| `test_read_glob_pattern_finds_files` | `ParquetReader` with `file_pattern="*.parquet"` discovers both fixture files |
| `test_read_missing_directory_raises` | Passing a non-existent directory raises `FileNotFoundError` |
| `test_read_no_parquet_files_raises` | Directory with no `.parquet` files raises `FileNotFoundError` |
| `test_read_logs_event_to_run_logger` | When `run_logger` is provided, `log_event` is called with `phase="parquet_read"` |

#### `tests/unit/test_corpus_sampler.py`

| Test | Description |
|---|---|
| `test_devanagari_ratio_pure_hindi` | `devanagari_ratio("भारत")` returns value close to 1.0 |
| `test_devanagari_ratio_pure_ascii` | `devanagari_ratio("hello world")` returns 0.0 |
| `test_devanagari_ratio_mixed` | `devanagari_ratio("भारत India")` returns value between 0.3 and 0.7 |
| `test_devanagari_ratio_only_spaces` | `devanagari_ratio("   ")` returns 0.0 (spaces excluded from denominator) |
| `test_devanagari_ratio_empty_string` | `devanagari_ratio("")` returns 0.0 |
| `test_is_valid_passes_clean_hindi` | 100-char Devanagari text with ratio 0.9 passes |
| `test_is_valid_fails_too_short` | 20-char text fails `min_char_count=30` |
| `test_is_valid_fails_too_long` | 6000-char text fails `max_char_count=5000` |
| `test_is_valid_fails_low_devanagari` | Text with devanagari_ratio=0.3 fails `min_devanagari_ratio=0.60` |
| `test_is_valid_fails_empty_string` | Empty string fails |
| `test_normalize_unicode_nfkc` | Composed and decomposed Devanagari text normalizes to same output |
| `test_normalize_collapses_whitespace` | `"भारत   है"` → `"भारत है"` |
| `test_normalize_strips_leading_trailing` | Leading/trailing whitespace stripped |
| `test_sample_creates_output_file` | After sampling, output `.txt` file exists on disk |
| `test_sample_one_record_per_line` | Every non-empty line in output is a single record (no embedded newlines) |
| `test_sample_excludes_filtered_records` | Records below `min_char_count` are absent from output |
| `test_sample_excludes_low_devanagari_records` | Records with low Devanagari ratio are absent from output |
| `test_sample_is_deterministic` | Two runs with the same seed produce byte-identical output files |
| `test_sample_different_seeds_differ` | Two runs with different seeds produce different output files |
| `test_sample_respects_target_size` | Output file size is within 10% of `target_size_gb` (tested with small fixture) |
| `test_sample_returns_manifest` | Return value is a `SampleManifest` with `actual_size_gb`, `written_records`, `random_seed` |
| `test_sample_manifest_written_records_positive` | `manifest.written_records > 0` |
| `test_sample_handles_no_parquet_files` | Raises `FileNotFoundError` when input directory has no parquet files |
| `test_sample_creates_parent_dirs` | Output `.txt` parent directory is created if it doesn't exist |
| `test_sample_logs_events` | `run_logger.log_event` called with `phase="corpus_sample"` started and completed events |
| `test_sample_registers_output_file` | `file_registry.register_file` called with the output `.txt` path |

**Tasks:**
1. Implement `src/hindi_tokenizer/corpus/parquet_reader.py` — `ParquetReader`.
2. Implement `src/hindi_tokenizer/corpus/corpus_sampler.py` — `CorpusSampler` with helpers `devanagari_ratio()`, `normalize_unicode()`, `is_valid_hindi_text()`.
3. Implement `src/hindi_tokenizer/observability/run_logger.py` — `TokenizerRunLogger`.
4. Implement `src/hindi_tokenizer/observability/file_registry.py` — `FileRegistry`.

**Verify:** `pytest tests/unit/test_parquet_reader.py tests/unit/test_corpus_sampler.py -v --cov=src/hindi_tokenizer`

---

## Phase 4 — Tokenizer Trainer

**Goal:** Train a Unigram+NFKC+Metaspace tokenizer from a plain-text corpus file; produce a complete HF-compatible artifact directory.

**Acceptance criteria:**
- `TokenizerTrainer.train()` on the fixture small corpus produces a valid `tokenizer.json`
- Artifact loads cleanly with `AutoTokenizer.from_pretrained`
- `ExperimentRunner` trains all three vocab-size variants
- `pytest tests/unit/test_tokenizer_trainer.py tests/unit/test_experiment_runner.py -v` — all green

### TDD Test Cases

#### `tests/unit/test_tokenizer_trainer.py`

| Test | Description |
|---|---|
| `test_train_creates_tokenizer_json` | After `train()`, `output_dir/tokenizer.json` exists |
| `test_train_creates_tokenizer_config_json` | `output_dir/tokenizer_config.json` exists |
| `test_train_creates_special_tokens_map_json` | `output_dir/special_tokens_map.json` exists |
| `test_train_creates_metadata_json` | `output_dir/tokenizer_metadata.json` exists |
| `test_train_metadata_contains_algorithm` | `tokenizer_metadata.json["algorithm"] == "unigram"` |
| `test_train_metadata_contains_vocab_size` | `tokenizer_metadata.json["vocab_size"]` equals the requested vocab_size |
| `test_train_metadata_contains_normalizer` | `tokenizer_metadata.json["normalizer"] == "NFKC"` |
| `test_train_metadata_contains_pre_tokenizer` | `tokenizer_metadata.json["pre_tokenizer"] == "Metaspace"` |
| `test_train_vocab_size_32k` | Trained with `vocab_size=32000`; `len(AutoTokenizer.from_pretrained(output_dir))` is close to 32000 |
| `test_train_special_tokens_present` | All 8 special tokens present in loaded tokenizer |
| `test_train_pad_token_id_is_zero` | `tokenizer.pad_token_id == 0` |
| `test_train_unk_token_id_is_one` | `tokenizer.unk_token_id == 1` |
| `test_train_bos_token_id_is_two` | `tokenizer.bos_token_id == 2` |
| `test_train_eos_token_id_is_three` | `tokenizer.eos_token_id == 3` |
| `test_train_loads_with_auto_tokenizer` | `AutoTokenizer.from_pretrained(output_dir)` succeeds without error |
| `test_train_small_fixture_corpus` | Training on `sample_text/small_corpus.txt` completes without exception |
| `test_train_creates_output_dir_if_missing` | Output directory is created automatically if it doesn't exist |
| `test_train_logs_started_event` | `run_logger.log_event` called with `status="started"` |
| `test_train_logs_completed_event` | `run_logger.log_event` called with `status="completed"` |
| `test_train_registers_tokenizer_json` | `file_registry.register_file` called with `tokenizer.json` path |

#### `tests/unit/test_experiment_runner.py`

| Test | Description |
|---|---|
| `test_experiment_runner_trains_all_variants` | Running `ExperimentRunner.run()` creates three artifact directories: 24k, 32k, 48k |
| `test_experiment_runner_uses_configured_vocab_sizes` | Vocab sizes used match `settings.tokenizer.vocab_sizes` |
| `test_experiment_runner_uses_same_input_file` | All three variants trained from the same input `.txt` file |
| `test_experiment_runner_logs_each_variant` | `run_logger.log_event` called separately for each vocab size variant |
| `test_experiment_runner_skips_existing_artifact` | If artifact dir already exists and `force_retrain=False`, skips that variant |
| `test_experiment_runner_force_retrain_overwrites` | With `force_retrain=True`, existing artifact is replaced |
| `test_experiment_runner_returns_artifact_dirs` | Returns list of three paths, each pointing to a valid artifact directory |

**Tasks:**
1. Implement `src/hindi_tokenizer/training/tokenizer_trainer.py` — `TokenizerTrainer`.
2. Implement `src/hindi_tokenizer/training/experiment_runner.py` — `ExperimentRunner`.

**Verify:** `pytest tests/unit/test_tokenizer_trainer.py tests/unit/test_experiment_runner.py -v --cov=src/hindi_tokenizer`

---

## Phase 5 — Tokenizer Validator

**Goal:** Load each tokenizer artifact, compute all quality metrics, write a JSON report, and enforce threshold checks.

**Acceptance criteria:**
- `TokenizerValidator` computes all metrics and writes valid JSON
- Threshold pass/fail correctly reflected in `ValidationReport.passes_thresholds`
- `pytest tests/unit/test_tokenizer_validator.py -v` — all green

### TDD Test Cases

#### `tests/unit/test_tokenizer_validator.py`

| Test | Description |
|---|---|
| `test_validator_loads_tokenizer_via_auto_tokenizer` | `AutoTokenizer.from_pretrained` called with the artifact dir (mock or fixture artifact) |
| `test_unk_rate_computed_correctly` | Inject mock tokenizer returning 1 UNK per 1000 tokens; assert `unk_rate == 0.001` |
| `test_unk_rate_zero_for_clean_hindi` | Fixture-trained tokenizer on validation sentences; `unk_rate` is 0.0 or very close |
| `test_chars_per_token_is_positive_float` | `chars_per_token > 0.0` |
| `test_tokens_per_word_is_positive_float` | `tokens_per_word > 0.0` |
| `test_roundtrip_success_rate_above_threshold` | Fixture-trained tokenizer; `roundtrip_success_rate >= 0.99` for fixture validation sentences |
| `test_roundtrip_compares_stripped_text` | Roundtrip comparison ignores leading/trailing spaces |
| `test_special_token_integrity_no_failures` | All 8 special tokens tokenize to themselves; `special_token_failures == []` |
| `test_special_token_integrity_detects_split` | Mock tokenizer where `"<pad>"` tokenizes to `["<", "pad", ">"]`; failure detected and recorded |
| `test_validation_report_has_all_required_keys` | Report dict contains: `unk_rate`, `chars_per_token`, `tokens_per_word`, `roundtrip_success_rate`, `special_token_failures`, `vocab_size`, `pad_token_id`, `unk_token_id`, `bos_token_id`, `eos_token_id`, `passes_thresholds` |
| `test_validation_report_saved_to_json_file` | After `validate()`, report JSON file exists at `report_path` |
| `test_validation_report_json_is_valid` | Report file parses as valid JSON |
| `test_passes_thresholds_true_when_all_met` | All metrics within configured thresholds → `passes_thresholds == True` |
| `test_passes_thresholds_false_on_high_unk_rate` | `unk_rate = 0.05` → `passes_thresholds == False` |
| `test_passes_thresholds_false_on_low_roundtrip` | `roundtrip_success_rate = 0.95` → `passes_thresholds == False` |
| `test_passes_thresholds_false_on_special_token_failure` | Any entry in `special_token_failures` → `passes_thresholds == False` |
| `test_validator_creates_parent_dirs` | Report parent directory created automatically if missing |
| `test_validator_logs_started_event` | `run_logger.log_event` called with `status="started"` |
| `test_validator_logs_completed_event` | `run_logger.log_event` called with `status="completed"` and `notes` containing vocab_size |
| `test_validator_logs_failed_event_on_thresholds` | When thresholds not met, `log_event` called with `status="failed"` |

**Tasks:**
1. Implement `src/hindi_tokenizer/validation/tokenizer_validator.py` — `TokenizerValidator`.

**Verify:** `pytest tests/unit/test_tokenizer_validator.py -v --cov=src/hindi_tokenizer`

---

## Phase 6 — Tokenizer Comparator

**Goal:** Load validation reports for all three variants, rank by quality metrics, produce a comparison markdown, and identify the recommended variant.

**Acceptance criteria:**
- Comparator loads three JSON reports and produces a valid markdown comparison table
- Recommended variant is one of the three trained names
- `pytest tests/unit/test_tokenizer_comparator.py -v` — all green

### TDD Test Cases

#### `tests/unit/test_tokenizer_comparator.py`

| Test | Description |
|---|---|
| `test_comparator_loads_three_reports` | Given paths to three fixture JSON reports, comparator loads all without error |
| `test_comparator_produces_markdown_file` | After `compare()`, a `.md` file exists at the configured output path |
| `test_comparison_markdown_has_header_row` | Output markdown contains a table with `unk_rate`, `chars_per_token`, `tokens_per_word`, `roundtrip_success_rate` columns |
| `test_comparison_markdown_has_row_per_variant` | Output markdown table has one row for each variant (24k, 32k, 48k) |
| `test_comparison_has_recommendation_section` | Output markdown contains a section with the text "Recommended" |
| `test_recommended_variant_is_one_of_three` | `comparison_result.recommended_variant` is in `["hindi_unigram_24k_v001", "hindi_unigram_32k_v001", "hindi_unigram_48k_v001"]` |
| `test_recommended_variant_has_lowest_unk_rate` | When all else equal, variant with lowest `unk_rate` is recommended |
| `test_recommended_variant_prefers_32k_on_tie` | When 32k and 48k have equal metrics, 32k is preferred (fewer embedding params) |
| `test_comparator_raises_on_missing_report` | `FileNotFoundError` raised if one of the report paths doesn't exist |
| `test_comparator_raises_on_all_variants_failing_thresholds` | If all variants have `passes_thresholds == False`, raises `ValueError` with explanation |
| `test_comparator_returns_comparison_result` | Return value is a `ComparisonResult` pydantic model |
| `test_comparator_logs_event` | `run_logger.log_event` called with `phase="comparison"` |
| `test_comparator_registers_report_file` | `file_registry.register_file` called with the output `.md` path |

**Tasks:**
1. Implement `src/hindi_tokenizer/validation/tokenizer_comparator.py` — `TokenizerComparator`.

**Verify:** `pytest tests/unit/test_tokenizer_comparator.py -v --cov=src/hindi_tokenizer`

---

## Phase 7 — Artifact Packager & Checksum Generator

**Goal:** Assemble the final frozen artifact directory from the selected variant; add VERSION, README, training config, and SHA-256 checksums.

**Acceptance criteria:**
- Final artifact directory contains all required files
- `checksums.json` covers every file in the artifact
- `pytest tests/unit/test_artifact_packager.py tests/unit/test_checksum_generator.py -v` — all green

### TDD Test Cases

#### `tests/unit/test_artifact_packager.py`

| Test | Description |
|---|---|
| `test_packager_copies_tokenizer_json` | `tokenizer.json` present in output dir after `package()` |
| `test_packager_copies_tokenizer_config_json` | `tokenizer_config.json` present in output dir |
| `test_packager_copies_special_tokens_map_json` | `special_tokens_map.json` present in output dir |
| `test_packager_copies_tokenizer_metadata_json` | `tokenizer_metadata.json` present in output dir |
| `test_packager_creates_version_file` | `VERSION` file created containing the version string (e.g. `hindi_slm_tokenizer_v001`) |
| `test_packager_version_file_content` | `VERSION` file content matches `settings.project.tokenizer_version` |
| `test_packager_creates_readme` | `README.md` created in artifact dir |
| `test_packager_readme_contains_algorithm` | README contains the word "Unigram" |
| `test_packager_readme_contains_vocab_size` | README contains the actual vocab size of the selected tokenizer |
| `test_packager_copies_training_config` | `tokenizer_training_config.yaml` present in artifact dir |
| `test_packager_copies_validation_report` | `tokenizer_validation_report.json` present in artifact dir |
| `test_packager_copies_comparison_report` | `tokenizer_comparison_report.md` present in artifact dir |
| `test_packager_creates_output_dir` | Output directory created automatically if it doesn't exist |
| `test_packager_logs_event` | `run_logger.log_event` called with `phase="packaging"` |
| `test_packager_registers_all_files` | `file_registry.register_file` called for each file in the artifact dir |
| `test_packager_raises_on_missing_source_artifact` | Raises `FileNotFoundError` if source variant artifact dir doesn't exist |

#### `tests/unit/test_checksum_generator.py`

| Test | Description |
|---|---|
| `test_checksums_json_created` | `checksums.json` created in target directory after `generate()` |
| `test_checksums_covers_all_files` | Every file in artifact dir (except `checksums.json` itself) has an entry in `checksums.json` |
| `test_checksum_values_are_sha256` | Each checksum value is a 64-character lowercase hex string |
| `test_checksum_is_reproducible` | Running `generate()` twice on the same directory produces the same `checksums.json` content |
| `test_checksum_changes_on_file_modification` | After modifying a file, the checksum for that file changes |
| `test_checksum_is_sha256_not_md5` | Checksum length is 64 (SHA-256), not 32 (MD5) |
| `test_checksum_handles_subdirectories` | Files in subdirectories of the artifact dir are also checksummed |
| `test_checksum_excludes_itself` | `checksums.json` is not listed as an entry within `checksums.json` |

**Tasks:**
1. Implement `src/hindi_tokenizer/packaging/artifact_packager.py` — `ArtifactPackager`.
2. Implement `src/hindi_tokenizer/packaging/checksum_generator.py` — `ChecksumGenerator`.

**Verify:** `pytest tests/unit/test_artifact_packager.py tests/unit/test_checksum_generator.py -v --cov=src/hindi_tokenizer`

---

## Phase 8 — Publisher & Python SDK

**Goal:** Publish the frozen artifact to Hugging Face Hub; provide a thin Python SDK for loading, encoding, and decoding.

**Acceptance criteria:**
- HF Hub calls are correctly formed (mocked in unit tests)
- SDK functions correctly wrap `AutoTokenizer`
- `pytest tests/unit/test_tokenizer_publisher.py tests/unit/test_sdk_*.py -v` — all green

### TDD Test Cases

#### `tests/unit/test_tokenizer_publisher.py`

| Test | Description |
|---|---|
| `test_publisher_calls_create_repo` | Mock `HfApi.create_repo`; assert called with configured `repo_id` and `private=True` |
| `test_publisher_calls_upload_folder` | Mock `HfApi.upload_folder`; assert called with `folder_path` = artifact dir and correct `repo_id` |
| `test_publisher_create_repo_called_before_upload` | Mock both; assert `create_repo` call index < `upload_folder` call index |
| `test_publisher_uses_hf_token_from_env` | Reads HF token from `HF_TOKEN` env var; assert token passed to `HfApi` (mock) |
| `test_publisher_raises_on_missing_artifact_dir` | Raises `FileNotFoundError` if `source_dir` doesn't exist |
| `test_publisher_uses_repo_id_from_config` | `repo_id` taken from `publish_config.yaml`, not hardcoded |
| `test_publisher_exist_ok_true_on_create` | `create_repo` called with `exist_ok=True` so re-publish doesn't fail |
| `test_publisher_logs_started_event` | `run_logger.log_event` called with `status="started"` |
| `test_publisher_logs_completed_event` | `run_logger.log_event` called with `status="completed"` |
| `test_publisher_dry_run_skips_upload` | With `dry_run=True`, no HF API calls made |

#### `tests/unit/test_sdk_loader.py`

| Test | Description |
|---|---|
| `test_load_from_local_path` | `load_hindi_slm_tokenizer(local_artifact_dir)` returns an `AutoTokenizer` instance |
| `test_load_calls_auto_tokenizer_from_pretrained` | Mock `AutoTokenizer.from_pretrained`; assert called with correct path |
| `test_load_raises_on_missing_path` | Raises `OSError` for a path that doesn't exist |

#### `tests/unit/test_sdk_encode.py`

| Test | Description |
|---|---|
| `test_encode_returns_input_ids` | `encode_text(mock_tokenizer, "भारत है।")` returns dict with `"input_ids"` key |
| `test_encode_returns_attention_mask` | Return dict contains `"attention_mask"` key |
| `test_encode_with_special_tokens_true` | `add_special_tokens=True` passed to tokenizer call |
| `test_encode_with_special_tokens_false` | `add_special_tokens=False` passed when specified |
| `test_encode_empty_string` | Empty string encodes to empty `input_ids` list (or `[bos]` if special tokens added) |

#### `tests/unit/test_sdk_decode.py`

| Test | Description |
|---|---|
| `test_decode_returns_string` | `decode_ids(mock_tokenizer, [2, 100, 200, 3])` returns a `str` |
| `test_decode_skips_special_tokens` | `skip_special_tokens=True` passed to tokenizer call |
| `test_decode_empty_ids` | Empty `input_ids` list returns empty string |
| `test_decode_roundtrip` | `decode_ids(tokenizer, encode_text(tokenizer, text)["input_ids"])` reproduces original text |

**Tasks:**
1. Implement `src/hindi_tokenizer/publishing/tokenizer_publisher.py` — `TokenizerPublisher`.
2. Implement `src/hindi_tokenizer/sdk/loader.py`, `sdk/encode.py`, `sdk/decode.py`.

**Verify:** `pytest tests/unit/test_tokenizer_publisher.py tests/unit/test_sdk_loader.py tests/unit/test_sdk_encode.py tests/unit/test_sdk_decode.py -v --cov=src/hindi_tokenizer`

---

## Phase 9 — Observability Unit Tests

**Goal:** Confirm run logger and file registry append correctly and produce valid CSVs.

### TDD Test Cases

#### `tests/unit/test_run_logger.py`

| Test | Description |
|---|---|
| `test_log_event_creates_csv_if_missing` | First `log_event` call creates `pipeline_run_log.csv` |
| `test_log_event_appends_row` | Two `log_event` calls produce a CSV with 2 data rows |
| `test_log_event_row_contains_run_id` | CSV row contains the `run_id` passed at construction |
| `test_log_event_row_contains_timestamp` | CSV row has a non-empty `timestamp` field |
| `test_log_event_row_contains_phase` | CSV row `phase` column matches argument |
| `test_log_event_row_contains_component` | CSV row `component` column matches argument |
| `test_log_event_row_contains_status` | CSV row `status` column matches argument |
| `test_log_event_multiple_runs_accumulate` | Two logger instances with different run_ids append to the same CSV |
| `test_log_event_creates_parent_dirs` | Parent directory of CSV created automatically if missing |

#### `tests/unit/test_file_registry.py`

| Test | Description |
|---|---|
| `test_register_file_creates_csv_if_missing` | First `register_file` call creates `data_file_registry.csv` |
| `test_register_file_appends_row` | Two `register_file` calls produce a CSV with 2 data rows |
| `test_register_file_computes_sha256` | CSV row `sha256` column is a 64-char hex string matching the actual file content |
| `test_register_file_records_size_bytes` | `size_bytes` column matches actual file size on disk |
| `test_register_file_row_contains_run_id` | `run_id` column matches constructor argument |
| `test_register_file_row_contains_role` | `role` column matches argument |
| `test_register_file_row_contains_stage` | `stage` column matches argument |
| `test_register_file_row_contains_file_format` | `file_format` column matches argument |
| `test_register_file_row_contains_row_count` | `row_count` column stores provided value |
| `test_register_file_row_contains_compression` | `compression` column matches argument |
| `test_register_file_creates_parent_dirs` | Parent directory of CSV created automatically if missing |

**Verify:** `pytest tests/unit/test_run_logger.py tests/unit/test_file_registry.py -v --cov=src/hindi_tokenizer`

---

## Phase 10 — Orchestration & Integration Tests

**Goal:** CLI wires all stages; integration tests verify multi-stage pipelines on fixture data with HF Hub calls mocked.

**Acceptance criteria:**
- `--dry-run` exits cleanly
- `--step all` on fixture data produces a valid tokenizer artifact
- `pytest tests/unit/test_run_tokenizer.py tests/integration/ -v -m "not requires_hf_hub"` — all green
- Full unit suite passes: `pytest tests/unit/ -v --cov=src/hindi_tokenizer --cov-report=term-missing`
- Coverage ≥ 80%
- `ruff check src/ tests/` clean

### TDD Test Cases

#### `tests/unit/test_run_tokenizer.py` (CLI unit tests)

| Test | Description |
|---|---|
| `test_dry_run_exits_without_writing_data` | `--dry-run` flag: no files written to `data/` directory |
| `test_dry_run_prints_config_summary` | `--dry-run` prints project name and config path to console |
| `test_step_sample_calls_corpus_sampler` | `--step sample`: mock `CorpusSampler.sample()`; assert called |
| `test_step_train_calls_experiment_runner` | `--step train`: mock `ExperimentRunner.run()`; assert called |
| `test_step_validate_calls_tokenizer_validator` | `--step validate`: mock `TokenizerValidator.validate()`; assert called for each variant |
| `test_step_compare_calls_tokenizer_comparator` | `--step compare`: mock `TokenizerComparator.compare()`; assert called |
| `test_step_package_calls_artifact_packager` | `--step package`: mock `ArtifactPackager.package()`; assert called |
| `test_step_publish_calls_tokenizer_publisher` | `--step publish`: mock `TokenizerPublisher.publish()`; assert called |
| `test_creates_output_directories_at_startup` | All expected `data/` subdirectories created before any stage runs |
| `test_run_id_is_uuid` | UUID `run_id` is generated and printed at startup |
| `test_smoke_test_flag_uses_smoke_sample` | `--smoke-test` flag selects `smoke_test` sample path from config |

#### `tests/integration/test_corpus_pipeline.py`

| Test | Description |
|---|---|
| `test_parquet_reader_to_sampler_end_to_end` | `ParquetReader` reads fixture parquets → `CorpusSampler` writes `.txt` → output file contains valid Hindi lines |
| `test_sampler_output_contains_no_filtered_records` | None of the short or low-Devanagari records from fixtures appear in output |
| `test_sampler_output_is_utf8` | Output file decodes as valid UTF-8 without errors |
| `test_sampler_manifest_matches_actual_file` | `manifest.written_records` matches line count of output file |

#### `tests/integration/test_training_pipeline.py`

| Test | Description |
|---|---|
| `test_trainer_on_fixture_small_corpus` | `TokenizerTrainer.train()` on `sample_text/small_corpus.txt` with `vocab_size=500`; artifact created |
| `test_trained_artifact_loads_with_auto_tokenizer` | `AutoTokenizer.from_pretrained(artifact_dir)` succeeds |
| `test_trained_tokenizer_encodes_hindi` | `tokenizer("भारत है।")` returns non-empty `input_ids` |
| `test_trained_tokenizer_roundtrip` | Encode then decode a fixture validation sentence; decoded text normalizes to original |
| `test_experiment_runner_creates_three_artifacts` | Running `ExperimentRunner` on fixture corpus creates three artifact dirs |

#### `tests/integration/test_full_pipeline.py`

| Test | Description |
|---|---|
| `test_full_pipeline_sample_to_artifact` | Parquet read → sample → train (small vocab) → validate → compare → package; final artifact dir contains all required files |
| `test_full_pipeline_validation_report_passes_thresholds` | On fixture data (small vocab), `passes_thresholds` is evaluated without error |
| `test_full_pipeline_checksums_generated` | `checksums.json` present and non-empty in final artifact dir |
| `test_full_pipeline_publish_mocked` | HF Hub calls mocked; `TokenizerPublisher.publish()` called with correct args |
| `test_full_pipeline_run_log_has_entries` | After pipeline, `pipeline_run_log.csv` has at least one row per stage |
| `test_full_pipeline_file_registry_has_entries` | After pipeline, `data_file_registry.csv` has at least one row per output file |

**Tasks:**
1. Implement `src/hindi_tokenizer/orchestration/run_tokenizer.py` — Typer CLI wiring all stages.
2. Write all integration tests.

**Verify:**
```bash
# Unit tests with coverage
pytest tests/unit/ -v --cov=src/hindi_tokenizer --cov-report=term-missing

# Integration tests (HF Hub mocked)
pytest tests/integration/ -v -m "not requires_hf_hub"

# Linting
ruff check src/ tests/

# Full suite
pytest tests/ -v -m "not requires_hf_hub"
```

---

## Phase 11 — Production Run

**Goal:** Run the actual tokenizer training pipeline on real Sangraha data.

**This phase is not code — it is operations. No new tests are written.**

**Steps:**

1. Confirm `data_ingestion/data/final/parquet/train/*.parquet` exists with at least 5 GB of data.
2. Update `configs/tokenizer_training_config.yaml` with correct `parquet_train_folder` path.
3. Run smoke test (500 MB sample, 32k only):
   ```bash
   python -m hindi_tokenizer.orchestration.run_tokenizer --step sample --smoke-test
   python -m hindi_tokenizer.orchestration.run_tokenizer --step train --smoke-test
   python -m hindi_tokenizer.orchestration.run_tokenizer --step validate --smoke-test
   ```
4. Inspect smoke test validation report. Confirm `unk_rate < 0.001` and `roundtrip_success_rate > 0.99`.
5. If smoke test passes, run experiment (5 GB sample, 24k + 32k + 48k):
   ```bash
   python -m hindi_tokenizer.orchestration.run_tokenizer --step all
   ```
6. Review `data/reports/tokenizer_comparison_report.md`.
7. Manually inspect vocabulary for over-fragmentation of common Hindi words.
8. Select final variant (expected: `hindi_unigram_32k_v001`).
9. Optionally retrain on 10 GB if compute allows.
10. Run `--step package` and `--step publish`.

---

## Phase 12 — TOKENIZER_HANDOFF.md

**Goal:** Document the frozen tokenizer contract for the SLM pretraining workstream.

**Write `TOKENIZER_HANDOFF.md` at the monorepo root:**

Contents:
- Tokenizer name, version, artifact path, HF repo ID
- `vocab_size` (exact value from `len(tokenizer)`)
- Special token IDs: `pad_token_id`, `unk_token_id`, `bos_token_id`, `eos_token_id`
- Algorithm, normalizer, pre-tokenizer, decoder
- Validation metric summary (unk_rate, chars_per_token, roundtrip)
- Python loading snippet:
  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("data/final/hindi_slm_tokenizer_v001")
  config = GPT2Config(vocab_size=len(tokenizer), ...)
  ```
- Freezing declaration: confirm tokenizer is frozen and must not be modified

---

## Summary Checklist

Before moving to SLM pretraining:

- [ ] All unit tests pass: `pytest tests/unit/ -v`
- [ ] All integration tests pass (HF Hub mocked): `pytest tests/integration/ -v -m "not requires_hf_hub"`
- [ ] Coverage ≥ 80%: `pytest --cov=src/hindi_tokenizer --cov-report=term-missing`
- [ ] Linting clean: `ruff check src/ tests/`
- [ ] Smoke test passed on real Sangraha data
- [ ] Three tokenizer variants trained and compared
- [ ] Final variant selected and validation metrics logged
- [ ] Final artifact assembled in `data/final/hindi_slm_tokenizer_v001/`
- [ ] All required artifact files present: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer_metadata.json`, `VERSION`, `README.md`, `checksums.json`
- [ ] `AutoTokenizer.from_pretrained` loads without error
- [ ] Tokenizer published to Hugging Face Hub
- [ ] TOKENIZER_HANDOFF.md written at monorepo root
- [ ] Tokenizer frozen — no further modifications
- [ ] All changes committed and pushed
