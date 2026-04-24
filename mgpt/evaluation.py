"""Sample quality metrics: character-level similarity and three-tier semantic heuristics."""

from __future__ import annotations

from collections import Counter

# Longest allowed consecutive consonant run (English allows 4 in e.g. "twelfths").
_MAX_CONSONANT_RUN = 4

_VOWELS = frozenset("aeiouy")

_IMPOSSIBLE_STARTS = frozenset(
    {
        "zx",
        "bx",
        "dx",
        "fx",
        "gx",
        "jx",
        "kx",
        "px",
        "qx",
        "sx",
        "tx",
        "vx",
        "wx",
        "xx",
    }
)


def _normalized_corpus_set(training_corpus: list[str]) -> set[str]:
    return {name.lower().strip() for name in training_corpus}


def _avg_corpus_length(training_corpus: list[str]) -> float:
    return sum(len(w) for w in training_corpus) / len(training_corpus)


def _corpus_bigrams(training_corpus: list[str]) -> set[str]:
    out: set[str] = set()
    for w in training_corpus:
        out.update(w[j : j + 2].lower() for j in range(len(w) - 1))
    return out


def char_distribution_similarity(samples: list[str], corpus_docs: list[str]) -> float:
    """Similarity in [0, 1] between per-character frequency distributions."""
    sample_text = "".join(samples).lower()
    corpus_text = "".join(corpus_docs).lower()
    if not sample_text or not corpus_text:
        return 0.0
    cs = Counter(sample_text)
    cc = Counter(corpus_text)
    chars = set(cs) | set(cc)
    tot_s = sum(cs.values())
    tot_c = sum(cc.values())
    l1 = sum(abs(cs[ch] / tot_s - cc[ch] / tot_c) for ch in chars)
    return max(0.0, 1.0 - 0.5 * l1)


def evaluate_sample_quality(samples: list[str], corpus_docs: list[str]) -> dict[str, float]:
    """Length-focused summary vs the training corpus."""
    if not samples or not corpus_docs:
        return {"avg_sample_length": 0.0, "length_similarity": 0.0}
    avg_s = sum(len(s) for s in samples) / len(samples)
    lens = [len(d) for d in corpus_docs]
    avg_c = sum(lens) / len(lens)
    len_diff = abs(avg_s - avg_c) / avg_c if avg_c else 0.0
    return {
        "avg_sample_length": avg_s,
        "length_similarity": max(0.0, 1.0 - len_diff),
    }


def count_real_words(samples: list[str], training_corpus: list[str]) -> dict[str, float | list[str]]:
    corpus_set = _normalized_corpus_set(training_corpus)
    real_words: list[str] = []
    for sample in samples:
        if sample.lower().strip() in corpus_set:
            real_words.append(sample)
    n = len(samples)
    return {
        "real_word_count": len(real_words),
        "real_word_ratio": len(real_words) / n if n else 0.0,
        "real_words": real_words,
    }


def is_pronounceable(word: str) -> bool:
    if not word:
        return False
    word_lower = word.lower()
    if not any(c in _VOWELS for c in word_lower):
        return False
    consonant_run = 0
    for char in word_lower:
        if char not in _VOWELS:
            consonant_run += 1
            if consonant_run > _MAX_CONSONANT_RUN:
                return False
        else:
            consonant_run = 0
    for i in range(len(word_lower) - 2):
        if word_lower[i] == word_lower[i + 1] == word_lower[i + 2]:
            return False
    if len(word_lower) >= 2 and word_lower[:2] in _IMPOSSIBLE_STARTS:
        return False
    return True


def _score_plausibility_impl(word: str, avg_corpus_len: float, corpus_bigrams: set[str]) -> float:
    pronounceable = 1.0 if is_pronounceable(word) else 0.0
    len_diff = abs(len(word) - avg_corpus_len) / avg_corpus_len
    len_score = max(0.0, 1.0 - len_diff)
    word_bigrams = {word[i : i + 2].lower() for i in range(len(word) - 1)}
    if word_bigrams:
        bigram_overlap = len(word_bigrams & corpus_bigrams) / len(word_bigrams)
    else:
        bigram_overlap = 0.0
    return (pronounceable + len_score + bigram_overlap) / 3.0


def score_plausibility(word: str, training_corpus: list[str]) -> float:
    if not word or not training_corpus:
        return 0.0
    return _score_plausibility_impl(
        word, _avg_corpus_length(training_corpus), _corpus_bigrams(training_corpus)
    )


def classify_plausible_words(
    samples: list[str],
    training_corpus: list[str],
    plausibility_threshold: float = 0.6,
) -> dict[str, float | list[tuple[str, float]]]:
    if not training_corpus:
        n = len(samples)
        return {
            "plausible_count": 0,
            "plausible_ratio": 0.0,
            "plausible_words": [],
            "avg_plausibility": 0.0,
        }
    corpus_set = _normalized_corpus_set(training_corpus)
    avg = _avg_corpus_length(training_corpus)
    big = _corpus_bigrams(training_corpus)
    plausible: list[tuple[str, float]] = []
    scores: list[float] = []
    for sample in samples:
        score = _score_plausibility_impl(sample, avg, big)
        scores.append(score)
        if sample.lower().strip() not in corpus_set and score >= plausibility_threshold:
            plausible.append((sample, score))
    n = len(samples)
    return {
        "plausible_count": len(plausible),
        "plausible_ratio": len(plausible) / n if n else 0.0,
        "plausible_words": plausible,
        "avg_plausibility": sum(scores) / len(scores) if scores else 0.0,
    }


def is_nonsense(word: str) -> bool:
    if len(word) < 2:
        return True
    word_lower = word.lower()
    if len(set(word_lower)) == 1:
        return True
    if all(c in _VOWELS for c in word_lower):
        return True
    if not any(c in _VOWELS for c in word_lower):
        return True
    if not is_pronounceable(word):
        return True
    return False


def count_nonsense_words(samples: list[str]) -> dict[str, float | list[str]]:
    nonsense = [word for word in samples if is_nonsense(word)]
    n = len(samples)
    return {
        "nonsense_count": len(nonsense),
        "nonsense_ratio": len(nonsense) / n if n else 0.0,
        "nonsense_words": nonsense,
    }


def evaluate_semantic_quality(
    samples: list[str],
    training_corpus: list[str],
    plausibility_threshold: float = 0.6,
) -> dict[str, float | list[str]]:
    nonsense_results = count_nonsense_words(samples)
    total = len(samples)
    if not training_corpus:
        return {
            "tier1_real_count": 0,
            "tier1_real_ratio": 0.0,
            "tier1_examples": [],
            "tier2_plausible_count": 0,
            "tier2_plausible_ratio": 0.0,
            "tier2_avg_score": 0.0,
            "tier2_examples": [],
            "tier3_nonsense_count": int(nonsense_results["nonsense_count"]),
            "tier3_nonsense_ratio": float(nonsense_results["nonsense_ratio"]),
            "tier3_examples": nonsense_results["nonsense_words"][:5],
            "overall_quality_score": max(
                0.0,
                min(
                    1.0,
                    (
                        0.0
                        + 0.0
                        + (1.0 - nonsense_results["nonsense_ratio"]) * 0.3
                    )
                    / 2.0,
                ),
            ),
            "distribution_sum": float(nonsense_results["nonsense_ratio"]),
        }

    corpus_set = _normalized_corpus_set(training_corpus)
    avg = _avg_corpus_length(training_corpus)
    big = _corpus_bigrams(training_corpus)
    real_words: list[str] = []
    plausible_pairs: list[tuple[str, float]] = []
    all_scores: list[float] = []
    for s in samples:
        sc = _score_plausibility_impl(s, avg, big)
        all_scores.append(sc)
        norm = s.lower().strip()
        if norm in corpus_set:
            real_words.append(s)
        elif sc >= plausibility_threshold:
            plausible_pairs.append((s, sc))

    tier2_count = len(plausible_pairs)
    tier2_ratio = tier2_count / total if total else 0.0
    tier2_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    real_ratio = len(real_words) / total if total else 0.0
    overall = (
        real_ratio * 1.0 + tier2_ratio * 0.7 + (1.0 - nonsense_results["nonsense_ratio"]) * 0.3
    ) / 2.0
    overall = max(0.0, min(1.0, overall))
    return {
        "tier1_real_count": len(real_words),
        "tier1_real_ratio": real_ratio,
        "tier1_examples": real_words[:5],
        "tier2_plausible_count": tier2_count,
        "tier2_plausible_ratio": tier2_ratio,
        "tier2_avg_score": tier2_avg,
        "tier2_examples": [w for w, _ in plausible_pairs[:5]],
        "tier3_nonsense_count": int(nonsense_results["nonsense_count"]),
        "tier3_nonsense_ratio": float(nonsense_results["nonsense_ratio"]),
        "tier3_examples": nonsense_results["nonsense_words"][:5],
        "overall_quality_score": overall,
        "distribution_sum": float(real_ratio + tier2_ratio + nonsense_results["nonsense_ratio"]),
    }


def compute_sample_quality_metrics(
    samples: list[str],
    corpus_docs: list[str],
) -> tuple[float, dict[str, float], dict[str, float | list[str]]]:
    """Run all quality evaluators once (for console + run report)."""
    char_dist = char_distribution_similarity(samples, corpus_docs)
    quality_metrics = evaluate_sample_quality(samples, corpus_docs)
    semantic_quality = evaluate_semantic_quality(samples, corpus_docs)
    return char_dist, quality_metrics, semantic_quality


def format_sample_quality_console_lines(
    char_dist_score: float,
    quality_metrics: dict[str, float],
    semantic_quality: dict[str, object],
    *,
    n_samples: int,
) -> list[str]:
    """Human-readable lines for stdout (same numbers as written to the run report)."""
    lines: list[str] = [
        "",
        "=" * 60,
        "SAMPLE QUALITY EVALUATION",
        "=" * 60,
        "",
        "--- Character-level metrics ---",
        f"Char distribution similarity: {char_dist_score:.4f}",
        f"Avg sample length: {quality_metrics['avg_sample_length']:.1f}",
        f"Length similarity: {quality_metrics['length_similarity']:.4f}",
        "",
        "--- Semantic quality (three-tier) ---",
        (
            f"Tier 1 (Real words):       {int(semantic_quality['tier1_real_count']):2d}/{n_samples} "
            f"({float(semantic_quality['tier1_real_ratio']):.1%})"
        ),
    ]
    ex1 = semantic_quality["tier1_examples"]
    if ex1:
        lines.append(f"  Examples: {', '.join(str(x) for x in ex1[:3])}")
    lines.append(
        f"Tier 2 (Plausible words):  {int(semantic_quality['tier2_plausible_count']):2d}/{n_samples} "
        f"({float(semantic_quality['tier2_plausible_ratio']):.1%})"
    )
    lines.append(f"  Avg plausibility: {float(semantic_quality['tier2_avg_score']):.2f}")
    ex2 = semantic_quality["tier2_examples"]
    if ex2:
        lines.append(f"  Examples: {', '.join(str(x) for x in ex2[:3])}")
    lines.append(
        f"Tier 3 (Nonsense words):   {int(semantic_quality['tier3_nonsense_count']):2d}/{n_samples} "
        f"({float(semantic_quality['tier3_nonsense_ratio']):.1%})"
    )
    ex3 = semantic_quality["tier3_examples"]
    if ex3:
        lines.append(f"  Examples: {', '.join(str(x) for x in ex3[:3])}")
    lines.append(
        f"\nOVERALL QUALITY SCORE: {float(semantic_quality['overall_quality_score']):.3f}"
    )
    lines.append("=" * 60)
    lines.append("")
    return lines
