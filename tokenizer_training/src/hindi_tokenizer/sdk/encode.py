"""Encodes text to token IDs using the Hindi SLM tokenizer."""

from __future__ import annotations

from typing import Any


def encode_text(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool = True,
) -> dict[str, list[int]]:
    return tokenizer(text, add_special_tokens=add_special_tokens)
