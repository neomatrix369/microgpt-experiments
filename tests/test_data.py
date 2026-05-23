"""Tests for dataset loading and tokeniser validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mgpt.data import build_tokeniser, load_dataset, validate_doc_chars
from mgpt.model import Tokeniser


class TestLoadDataset(unittest.TestCase):
    def test_given_empty_file_when_loaded_then_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.txt"
            path.write_text("\n\n  \n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty after stripping"):
                load_dataset(input_path=str(path), names_url="http://example.com/names.txt")

    def test_given_valid_lines_when_loaded_then_returns_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "names.txt"
            path.write_text("alice\nbob\n", encoding="utf-8")
            docs = load_dataset(input_path=str(path), names_url="http://example.com/names.txt")
            self.assertEqual(sorted(docs), ["alice", "bob"])


class TestBuildTokeniser(unittest.TestCase):
    def test_given_empty_docs_when_built_then_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty charset"):
            build_tokeniser([])

    def test_given_docs_when_built_then_vocab_includes_chars(self) -> None:
        tok = build_tokeniser(["ab", "bc"])
        self.assertIn("a", tok.uchars)
        self.assertEqual(tok.vocab_size, len(tok.uchars) + 1)


class TestValidateDocChars(unittest.TestCase):
    def test_given_oov_char_when_validated_then_raises(self) -> None:
        tok = build_tokeniser(["abc"])
        with self.assertRaisesRegex(ValueError, "out-of-vocabulary"):
            validate_doc_chars(["abc", "xyz!"], tok)

    def test_given_valid_docs_when_validated_then_passes(self) -> None:
        tok = build_tokeniser(["alice", "bob"])
        validate_doc_chars(["alice"], tok)


if __name__ == "__main__":
    unittest.main()
