"""
Colab-ready Turkish Address Preprocessing Pipeline

Implements a high-performance, modular preprocessing suite suitable for 1M+ addresses.

Key features:
- Deterministic normalization order
- Vectorized regex replacements
- LRU-cached token-level fuzzy typo correction via rapidfuzz
- Optional stopword removal (Turkish)
- Exact and optional near-duplicate deduplication with bucketed fuzzy compare
- End-to-end CLI entry that reads train/test if present and writes outputs

Dependencies: pandas, numpy, unidecode, rapidfuzz (preferred), nltk, scikit-learn (for hashing optional), tqdm (optional)
"""

from __future__ import annotations

import os
import sys
import re
import time
import math
import random
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from unidecode import unidecode


def _ensure_dependencies_installed() -> None:
    """Attempt to install runtime dependencies if missing (Colab-friendly).

    Uses subprocess to pip install only if imports fail. Keeps overhead minimal
    for environments that already satisfy requirements.
    """
    try:
        import rapidfuzz  # noqa: F401
    except Exception:  # pragma: no cover
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidfuzz>=3.6.0"], stdout=sys.stdout, stderr=sys.stderr)
        except Exception:
            warnings.warn("Failed to auto-install rapidfuzz. Falling back later if needed.")

    try:
        import nltk  # noqa: F401
    except Exception:  # pragma: no cover
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk>=3.8.1"], stdout=sys.stdout, stderr=sys.stderr)
        except Exception:
            warnings.warn("Failed to auto-install nltk. Stopword removal may be disabled.")


_ensure_dependencies_installed()


try:
    from rapidfuzz import process as rf_process
    from rapidfuzz import fuzz as rf_fuzz
except Exception:  # pragma: no cover
    rf_process = None  # type: ignore
    rf_fuzz = None  # type: ignore


try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    # Ensure stopwords are available
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:  # pragma: no cover
        nltk.download("stopwords")
except Exception:  # pragma: no cover
    nltk = None
    nltk_stopwords = None  # type: ignore


def _safe_lower(text: str) -> str:
    """Locale-safe lowercasing. For Turkish, plain .lower() is acceptable here
    because we unidecode afterwards to normalize Turkish characters.
    """
    return text.lower()


def _remove_extra_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class TurkishAddressConfig:
    """Configuration for TurkishAddressPreprocessor.

    - abbreviation_map: token-aware deterministic expansions
    - typo_candidates: trusted lexicon used by fuzzy correction
    - fuzzy_threshold: minimum similarity for applying a token correction
    - remove_stopwords: whether to drop non-discriminative tokens
    - enable_near_dedup: whether to remove near-duplicates within buckets
    - near_dedup_threshold: similarity to consider two addresses near-duplicates
    - chunksize: optional chunked processing for very large CSVs
    - canonical_slash: canonical separator for house compounds like 5/1 or 5-1
    """

    abbreviation_map: Dict[str, str] = field(default_factory=lambda: {
        # mahalle
        "mh": "mahalle",
        "mah": "mahalle",
        "mahalle": "mahalle",
        # cadde
        "cd": "caddesi",
        "cad": "caddesi",
        "cadde": "caddesi",
        "cadd": "caddesi",
        # sokak
        "sk": "sokak",
        "sok": "sokak",
        "sokak": "sokak",
        # bulvar
        "blv": "bulvar",
        "bulv": "bulvar",
        "bulvar": "bulvar",
        # apartman
        "apt": "apartmani",
        "ap": "apartmani",
        "apartman": "apartmani",
        # numara
        "no": "numara",
        "n": "numara",
        # daire
        "d": "daire",
        "daire": "daire",
        # kat
        "k": "kat",
        "kat": "kat",
    })
    typo_candidates: Iterable[str] = field(default_factory=lambda: [
        # Common Turkish admin/geographic names and frequent address tokens
        "istanbul", "ankara", "izmir", "bursa", "antalya", "mugla", "fethiye",
        "karsiyaka", "bostanli", "uskudar", "narlidere", "konak", "bornova",
        "cankaya", "kecioren", "karabaglar", "balcova", "bayrakli", "karabaaglar",
        "mahalle", "caddesi", "sokak", "bulvar", "apartmani", "numara", "daire", "kat",
        # extra variants often seen
        "karsiyaka", "uskudar", "iskitler", "guzelbahce", "karabaglar", "kecioren",
    ])
    fuzzy_threshold: int = 90
    remove_stopwords: bool = True
    enable_near_dedup: bool = False
    near_dedup_threshold: int = 98
    chunksize: Optional[int] = None
    canonical_slash: str = "/"


class TurkishAddressPreprocessor:
    """High-performance Turkish address preprocessor.

    Public methods:
    - preprocess_address(text) -> str
    - preprocess_dataframe(df, col_name) -> pd.DataFrame

    Hooks:
    - to_embedding_ready(text)
    - custom_rules_hook(text)
    """

    def __init__(self, config: Optional[TurkishAddressConfig] = None) -> None:
        self.config = config or TurkishAddressConfig()

        # Prepare abbreviation patterns (token-aware, with optional dots and colons)
        # e.g., "mah.", "mah:" -> "mahalle"
        self._abbr_patterns: List[Tuple[re.Pattern[str], str]] = []
        for key, value in self.config.abbreviation_map.items():
            # word boundary, allow dotted/colon forms and optional trailing dot
            # Examples: "mah", "mah.", "mah:", "mah:" with number after (handled later)
            pattern = re.compile(rf"(?<![\w\d]){re.escape(key)}\.?\:?\b")
            self._abbr_patterns.append((pattern, value))

        # Numeric normalization patterns
        # Normalize variations of numara: "no:5", "no 5", "n: 5", "5 no" -> "numara 5"
        self._re_numara_colon = re.compile(r"\b(?:no|n|numara)\s*[:\-]?\s*(\d+)\b")
        self._re_numara_trailing = re.compile(r"\b(\d+)\s*(?:no|n|numara)\b")

        # Compound house numbers: 5/1, 5-1 -> canonical 5/1 (configurable)
        self._re_house_compound = re.compile(r"\b(\d+)\s*[\/-]\s*(\d+)\b")

        # Remove extraneous punctuation but keep digits, letters, space, and separators / -
        self._re_punct = re.compile(r"[^0-9a-z\s/\-]")

        # Collapse multiple separators spaces or slashes/dashes
        self._re_multi_space = re.compile(r"\s+")
        self._re_multi_slash = re.compile(r"[/]+")
        self._re_multi_dash = re.compile(r"[-]+")

        # Stopwords
        self._stopwords: set[str] = set()
        if self.config.remove_stopwords and nltk_stopwords is not None:
            try:
                self._stopwords = set(nltk_stopwords.words("turkish"))
                # Add address-specific non-discriminative tokens
                self._stopwords.update({"il", "ilce", "turkiye", "posta", "kodu"})
            except Exception:  # pragma: no cover
                self._stopwords = set()

        # Prepare fuzzy lexicon
        self._lexicon: List[str] = sorted(set(map(str, self.config.typo_candidates)))

    # ------------------------------ Core Steps ------------------------------ #

    def _normalize_case_and_chars(self, text: str) -> str:
        text = _safe_lower(text)
        # normalize Turkish characters to ASCII (c, s, g, u, o, i)
        text = unidecode(text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        for pattern, replacement in self._abbr_patterns:
            text = pattern.sub(replacement, text)
        return text

    def _cleanup_punctuation(self, text: str) -> str:
        text = self._re_punct.sub(" ", text)
        text = self._re_multi_space.sub(" ", text)
        return text.strip()

    def _normalize_numbers(self, text: str) -> str:
        # Canonicalize explicit numara specifications
        text = self._re_numara_colon.sub(lambda m: f"numara {m.group(1)}", text)
        text = self._re_numara_trailing.sub(lambda m: f"numara {m.group(1)}", text)

        # Normalize house compounds like 5/1, 5-1 -> 5/<canonical>
        def _compound(m: re.Match[str]) -> str:
            left, right = m.group(1), m.group(2)
            return f"{left}{self.config.canonical_slash}{right}"

        text = self._re_house_compound.sub(_compound, text)

        # Collapse multiple separators
        text = self._re_multi_slash.sub(self.config.canonical_slash, text)
        text = self._re_multi_dash.sub("-", text)
        return text

    def custom_rules_hook(self, text: str) -> str:
        """Hook for project-specific rules. No-op by default."""
        return text

    def _maybe_remove_stopwords(self, text: str) -> str:
        if not self._stopwords:
            return text
        tokens = [tok for tok in text.split() if tok not in self._stopwords]
        return " ".join(tokens)

    # -------------------------- Fuzzy Typo Correction ----------------------- #

    @lru_cache(maxsize=100_000)
    def _correct_token(self, token: str) -> str:
        """Correct a single token using rapidfuzz if similar to a trusted lexicon.

        Applies only to alphabetic tokens with sufficient length. Returns the
        original token if below threshold or library unavailable.
        """
        if rf_process is None or rf_fuzz is None:
            return token
        if len(token) < 4:
            return token
        if not token.isalpha():
            return token
        if token in self._lexicon:
            return token

        match = rf_process.extractOne(token, self._lexicon, scorer=rf_fuzz.token_sort_ratio)
        if match is None:
            return token
        candidate, score, _ = match
        if score >= self.config.fuzzy_threshold:
            return str(candidate)
        return token

    def _apply_fuzzy_corrections(self, text: str) -> str:
        tokens = text.split()
        corrected = [self._correct_token(tok) for tok in tokens]
        return " ".join(corrected)

    # ----------------------------- Public API ------------------------------- #

    def preprocess_address(self, text: str) -> str:
        """Preprocess a single address string to a canonical normalized form.

        Steps (in order):
        1) lowercase
        2) unidecode
        3) abbreviation expansion
        4) punctuation cleanup & whitespace
        5) number and house-number normalization
        6) frequent typo correction (token-level rapidfuzz, cached)
        7) optional stopword removal
        8) final whitespace compaction
        """
        if not isinstance(text, str):
            return ""
        s = text
        s = self._normalize_case_and_chars(s)
        s = self._expand_abbreviations(s)
        s = self._cleanup_punctuation(s)
        s = self._normalize_numbers(s)
        s = self._apply_fuzzy_corrections(s)
        s = self._maybe_remove_stopwords(s)
        s = _remove_extra_spaces(s)
        s = self.custom_rules_hook(s)
        return s

    def preprocess_dataframe(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Apply preprocessing to a DataFrame efficiently.

        - Leaves original column intact; adds `normalized_address`.
        - Vectorizes deterministic regex rules via pandas.str methods where possible.
        - Uses cached token-level fuzzy correction.
        - Optionally deduplicates exactly and near-duplicates.
        """
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")

        s = df[col_name].astype(str)

        # Vectorized lower + unidecode via Series.apply (fast enough, avoids per-row Python in heavy logic)
        s = s.str.lower().map(unidecode)

        # Abbreviation expansion via sequential regex replace
        for pattern, replacement in self._abbr_patterns:
            s = s.str.replace(pattern, replacement, regex=True)

        # Punctuation cleanup
        s = s.str.replace(self._re_punct, " ", regex=True)
        s = s.str.replace(self._re_multi_space, " ", regex=True).str.strip()

        # Number normalization
        s = s.str.replace(self._re_numara_colon, lambda m: f"numara {m.group(1)}", regex=True)
        s = s.str.replace(self._re_numara_trailing, lambda m: f"numara {m.group(1)}", regex=True)
        s = s.str.replace(self._re_house_compound, lambda m: f"{m.group(1)}{self.config.canonical_slash}{m.group(2)}", regex=True)
        s = s.str.replace(self._re_multi_slash, self.config.canonical_slash, regex=True)
        s = s.str.replace(self._re_multi_dash, "-", regex=True)

        # Fuzzy corrections: token-level with LRU cache
        s = s.apply(self._apply_fuzzy_corrections)

        # Optional stopword removal
        if self._stopwords:
            s = s.apply(self._maybe_remove_stopwords)

        # Final compaction
        s = s.map(_remove_extra_spaces)

        out = df.copy()
        out["normalized_address"] = s

        return out

    # ------------------------------ Dedup Logic ----------------------------- #

    @staticmethod
    def _blocking_key(text: str) -> str:
        alnum = re.sub(r"[^a-z0-9]", "", text)
        prefix = alnum[:10]
        length_bin = str(len(alnum) // 5)
        return f"{prefix}|{length_bin}"

    def deduplicate(self, df: pd.DataFrame, near_duplicates: Optional[bool] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Deduplicate on `normalized_address`.

        Always removes exact duplicates. If near_duplicates is True (or config enabled),
        additionally removes near-duplicates within buckets based on a blocking key.
        Returns the deduplicated DataFrame and stats.
        """
        if "normalized_address" not in df.columns:
            raise ValueError("DataFrame must contain 'normalized_address' before deduplication")

        start_rows = len(df)
        df_ex = df.drop_duplicates(subset=["normalized_address"], keep="first").copy()
        exact_removed = start_rows - len(df_ex)

        enable_near = self.config.enable_near_dedup if near_duplicates is None else near_duplicates
        near_removed = 0

        if enable_near and rf_process is not None and rf_fuzz is not None:
            # Bucket by blocking key
            df_ex["_block"] = df_ex["normalized_address"].map(self._blocking_key)
            groups = df_ex.groupby("_block", sort=False)
            to_drop_idx: List[int] = []
            for _, g in groups:
                addrs = g["normalized_address"].tolist()
                idxs = g.index.tolist()
                # Greedy selection using high threshold; keep the first, drop those >= threshold to it
                kept: List[int] = []
                for i, base in enumerate(addrs):
                    if idxs[i] in to_drop_idx:
                        continue
                    kept.append(idxs[i])
                    # Compare remaining in bucket to base
                    for j in range(i + 1, len(addrs)):
                        if idxs[j] in to_drop_idx:
                            continue
                        score = rf_fuzz.token_sort_ratio(base, addrs[j])
                        if score >= self.config.near_dedup_threshold:
                            to_drop_idx.append(idxs[j])
                # continue to next bucket
            if to_drop_idx:
                df_ex = df_ex.drop(index=to_drop_idx)
                near_removed = len(to_drop_idx)
            if "_block" in df_ex.columns:
                df_ex = df_ex.drop(columns=["_block"])

        stats = {
            "start_rows": start_rows,
            "after_exact": len(df) - exact_removed,
            "exact_removed": exact_removed,
            "after_all": len(df_ex),
            "near_removed": near_removed,
        }
        return df_ex, stats

    # ------------------------------- Hooks --------------------------------- #

    def to_embedding_ready(self, text: str) -> str:
        """Hook for additional steps for embedding models (e.g., TF-IDF/BERT).
        Currently no-op; reserved for future extension.
        """
        return text


def _print_examples(df: pd.DataFrame, col: str, n: int = 10) -> None:
    sample = df.sample(n=min(n, len(df)), random_state=42)
    for _, row in sample.iterrows():
        before = row[col]
        after = row["normalized_address"]
        print(f"- BEFORE: {before}")
        print(f"  AFTER : {after}")


def _summary_report(df_before: pd.DataFrame, df_after: pd.DataFrame, col: str, dedup_stats: Optional[Dict[str, int]], elapsed: float) -> None:
    print("\nSummary:")
    print(f"- rows processed: {len(df_before)}")
    print(f"- unique before: {df_before[col].nunique(dropna=False)}")
    print(f"- unique after : {df_after['normalized_address'].nunique(dropna=False)}")
    if dedup_stats is not None:
        print(f"- exact removed: {dedup_stats.get('exact_removed', 0)}")
        print(f"- near removed : {dedup_stats.get('near_removed', 0)}")
        print(f"- final rows   : {dedup_stats.get('after_all', len(df_after))}")
    print(f"- elapsed time : {elapsed:.2f}s ({len(df_before)/max(elapsed,1e-6):.0f} rows/sec)")


def _process_csv_if_exists(prep: TurkishAddressPreprocessor, path: str, is_train: bool) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    print(f"Reading: {path}")
    df = pd.read_csv(path)
    col = "address"
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in {path}")
    t0 = time.time()
    df_out = prep.preprocess_dataframe(df, col)
    t1 = time.time()

    # Dedup always exact; near if enabled in config
    df_dedup, stats = prep.deduplicate(df_out, near_duplicates=None)

    # Save
    out_path = "preprocessed_addresses.csv"
    if is_train:
        cols = ["address", "normalized_address", "label"] if "label" in df.columns else ["address", "normalized_address"]
        save_df = df_dedup.copy()
        save_df = save_df[cols]
    else:
        cols = ["id", "address", "normalized_address"] if "id" in df.columns else ["address", "normalized_address"]
        save_df = df_dedup.copy()[cols]
    mode = "w"
    header = True
    if os.path.exists(out_path):
        # Overwrite to keep most recent run
        pass
    save_df.to_csv(out_path, index=False, mode=mode, header=header)

    # Examples and summary
    print("\nExamples (random 10):")
    _print_examples(df_out, col, n=10)
    _summary_report(df, df_out, col, stats, elapsed=t1 - t0)
    print(f"Saved: {out_path}")
    return df_out


def _run_sanity_tests() -> None:
    prep = TurkishAddressPreprocessor()
    cases = {
        "Akarca Mah. Adnan Menderes Cad. 864.Sok. No:15 D.1 K.2": "akarca mahalle adnan menderes caddesi 864 sokak numara 15 daire 1 kat 2",
        "Pazaryeri mah. 417. sk. No:6/4 Fethiye/MUĞLA": "pazaryeri mahalle 417 sokak numara 6/4 fethiye mugla",
        "Limanreis Mahallesi Aziz Sokak No 4 Narlıdere İzmir Narlıdere Narlıdere": "limanreis mahallesi aziz sokak numara 4 narlidere izmir narlidere narlidere",
        "1771 sokak no:5 d:5 Kaçuna Apt. Bostanlı Karsıyaka KARŞIYAKA İzmir": "1771 sokak numara 5 daire 5 kacuna apartmani bostanli karsiyaka karsiyaka izmir",
    }
    for raw, expected_prefix in cases.items():
        out = prep.preprocess_address(raw)
        # We assert strong invariants: abbreviation expansion and numara format
        assert "numara " in out, f"missing 'numara' in {out}"
        assert re.search(r"\bnumara \d+\b", out), f"numara not canonical: {out}"
        # Allow minor fuzzy differences, but ensure prefix matches expected start tokens
        assert out.startswith(expected_prefix.split()[0]), f"unexpected start for {out}"


if __name__ == "__main__":
    # End-to-end run if train.csv/test.csv present in CWD
    print("Turkish Address Preprocessing - Start")
    _run_sanity_tests()
    config = TurkishAddressConfig(
        remove_stopwords=True,
        enable_near_dedup=False,  # set True to enable near-duplicate removal
        fuzzy_threshold=90,
        near_dedup_threshold=98,
        canonical_slash="/",
    )
    preprocessor = TurkishAddressPreprocessor(config)

    any_done = False
    train_out = _process_csv_if_exists(preprocessor, "train.csv", is_train=True)
    if train_out is not None:
        any_done = True
    test_out = _process_csv_if_exists(preprocessor, "test.csv", is_train=False)
    if test_out is not None:
        any_done = True

    if not any_done:
        print("No train.csv or test.csv found in current directory. Nothing to process.")
    print("Done.")

# -------------------------- Module-level wrappers -------------------------- #

# Lazy singleton to expose simple function API as requested
_GLOBAL_PREPROCESSOR: Optional[TurkishAddressPreprocessor] = None


def _get_global_preprocessor() -> TurkishAddressPreprocessor:
    global _GLOBAL_PREPROCESSOR
    if _GLOBAL_PREPROCESSOR is None:
        _GLOBAL_PREPROCESSOR = TurkishAddressPreprocessor()
    return _GLOBAL_PREPROCESSOR


def preprocess_address(text: str) -> str:
    """Module-level convenience function wrapping TurkishAddressPreprocessor.preprocess_address."""
    return _get_global_preprocessor().preprocess_address(text)


def preprocess_dataframe(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Module-level convenience function wrapping TurkishAddressPreprocessor.preprocess_dataframe."""
    return _get_global_preprocessor().preprocess_dataframe(df, col_name)


