"""Tests for ingestion/corpus_splitter.py — CorpusSplitter."""

from slm_hindi.config.settings import SplitsConfig
from slm_hindi.ingestion.corpus_splitter import CorpusSplitter
from slm_hindi.schema.corpus_record import CorpusRecord

_HINDI_TEXT = "भारत एक विविधताओं से भरा देश है।"


def _records(n_docs: int, paras_per_doc: int = 2) -> list[CorpusRecord]:
    records = []
    for d in range(n_docs):
        for p in range(paras_per_doc):
            records.append(CorpusRecord(
                record_id=f"d{d:04d}_p{p:04d}",
                document_id=f"doc_{d:04d}",
                paragraph_id=f"p{p:04d}",
                source_type="huggingface_dataset",
                source_name="test",
                raw_text=_HINDI_TEXT,
                final_text=_HINDI_TEXT,
            ))
    return records


def _make_config(**kwargs) -> SplitsConfig:
    defaults = {"train": 0.98, "validation": 0.01, "test": 0.01, "random_seed": 42}
    defaults.update(kwargs)
    return SplitsConfig(**defaults)


def test_split_returns_three_keys():
    splitter = CorpusSplitter(_make_config())
    splits = splitter.split(_records(100))
    assert set(splits.keys()) == {"train", "validation", "test"}


def test_all_records_appear_in_exactly_one_split():
    splitter = CorpusSplitter(_make_config())
    all_records = _records(100)
    splits = splitter.split(all_records)
    total = sum(len(v) for v in splits.values())
    assert total == len(all_records)


def test_split_name_set_on_records():
    splitter = CorpusSplitter(_make_config())
    splits = splitter.split(_records(50))
    for split_name, records in splits.items():
        assert all(r.split_name == split_name for r in records)


def test_document_level_split_keeps_paragraphs_together():
    splitter = CorpusSplitter(_make_config())
    splits = splitter.split(_records(50, paras_per_doc=3))
    for split_name, records in splits.items():
        doc_ids = {r.document_id for r in records}
        # All paragraphs of same document must be in same split
        all_split_doc_ids = {
            r.document_id for s in splits.values() for r in s
        }
        for doc_id in doc_ids:
            appearances = sum(
                1 for s_name, s_records in splits.items()
                if any(r.document_id == doc_id for r in s_records)
            )
            assert appearances == 1, f"doc {doc_id} appears in multiple splits"


def test_split_ratios_approximate():
    splitter = CorpusSplitter(_make_config())
    # Use enough records to get meaningful ratios
    all_records = _records(200)
    splits = splitter.split(all_records)
    total = len(all_records)
    train_ratio = len(splits["train"]) / total
    val_ratio = len(splits["validation"]) / total
    test_ratio = len(splits["test"]) / total
    assert abs(train_ratio - 0.98) < 0.05
    assert abs(val_ratio - 0.01) < 0.03
    assert abs(test_ratio - 0.01) < 0.03


def test_deterministic_with_same_seed():
    splitter1 = CorpusSplitter(_make_config(random_seed=42))
    splitter2 = CorpusSplitter(_make_config(random_seed=42))
    all_records = _records(100)
    s1 = splitter1.split(all_records)
    s2 = splitter2.split(all_records)
    assert [r.record_id for r in s1["train"]] == [r.record_id for r in s2["train"]]
