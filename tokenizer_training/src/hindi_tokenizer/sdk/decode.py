"""Decodes token IDs to text using the Hindi SLM tokenizer."""

from __future__ import annotations

from typing import Any


def decode_ids(
    tokenizer: Any,
    token_ids: list[int],
    *,
    skip_special_tokens: bool = True,
) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
