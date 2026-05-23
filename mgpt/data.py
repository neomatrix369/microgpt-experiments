"""Dataset loading and character-level tokeniser construction."""

from __future__ import annotations

import os
import random

from .model import Tokeniser


def load_dataset(*, input_path: str, names_url: str) -> list[str]:
    """Let there be a Dataset: list[str] of documents (e.g. a list of names).

    In production each document would be an internet web page. Here we use
    32,000 names, one per line. The goal is to learn the patterns and then
    generate similar new documents. From ChatGPT's perspective, your
    conversation is just a funny looking "document" and its response is
    just a statistical completion.
    """
    if not os.path.exists(input_path):
        import urllib.request

        urllib.request.urlretrieve(names_url, input_path)

    with open(input_path) as f:
        docs = [line.strip() for line in f if line.strip()]

    if not docs:
        raise ValueError(
            f"Dataset is empty after stripping blank lines: {input_path!r}"
        )

    random.shuffle(docs)
    print(f"Num Docs: {len(docs)}")
    return docs


def build_tokeniser(docs: list[str]) -> Tokeniser:
    """Let there be a Tokeniser: strings to sequences of integer tokens and back.

    Neural networks work with numbers, not characters, so we assign one
    integer to each unique character. The integer values have no meaning;
    each token is just a separate discrete symbol. Production tokenisers
    like tiktoken (GPT-4) operate on chunks of characters for efficiency,
    but character-level is the simplest possible scheme.

    Returns:
        uchars: Sorted unique characters (token ids 0..n-1).
        bos: Beginning of Sequence token id.
        vocab_size: Total number of unique tokens.
    """
    # unique chars become token ids 0..n-1
    uchars = sorted(set("".join(docs)))

    # BOS acts as a delimiter: "a new document starts/ends here". During
    # training each name is wrapped: [BOS, e, m, m, a, BOS]. The model
    # learns that BOS initiates a new name and another BOS ends it.
    bos = len(uchars)

    # 26 lowercase a-z + 1 BOS = 27
    vocab_size = len(uchars) + 1
    if not uchars:
        raise ValueError("No characters in vocabulary — dataset produced an empty charset")
    print(f"Vocab Size: {vocab_size}")
    return Tokeniser(uchars, bos, vocab_size)


def validate_doc_chars(docs: list[str], tok: Tokeniser) -> None:
    """Raise ValueError if any document contains a character outside the vocabulary."""
    allowed = set(tok.uchars)
    for doc_idx, doc in enumerate(docs):
        for ch in doc:
            if ch not in allowed:
                raise ValueError(
                    f"Document {doc_idx} contains out-of-vocabulary character {ch!r}: {doc!r}"
                )
