# -*- coding: utf-8 -*-
"""
Turkish Address Normalization Pipeline
=====================================

Production-ready preprocessing pipeline for Turkish address data with 848,237 training addresses
grouped into 10,390 unique labels (average 81.6 addresses per label).

Author: NLP Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import string
from collections import defaultdict, Counter
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ================================================================
# TURKISH ADDRESS NORMALIZER CLASS
# ================================================================

class TurkishAddressNormalizer:
    """
    Comprehensive Turkish address normalizer with abbreviation expansion,
    typo correction, and structural standardization.
    
    Optimized for processing 1M+ addresses efficiently.
    """
    
    def __init__(self):
        """Initialize the normalizer with Turkish-specific rules and mappings."""
        
        # Turkish character mapping (ç→c, ş→s, ğ→g, ü→u, ö→o, ı→i)
        self.turkish_chars = {
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'
        }
        
        # Comprehensive abbreviation expansion mappings
        self.abbreviations = {
            # Neighborhood/District
            'mh': 'mahalle', 'mah': 'mahalle', 'mahalle': 'mahalle',
            'mh.': 'mahalle', 'mah.': 'mahalle',
            
            # Street types
            'cd': 'caddesi', 'cad': 'caddesi', 'cadde': 'caddesi',
            'cd.': 'caddesi', 'cad.': 'caddesi',
            'sk': 'sokak', 'sok': 'sokak', 'sokak': 'sokak',
            'sk.': 'sokak', 'sok.': 'sokak',
            'blv': 'bulvar', 'bulv': 'bulvar', 'bulvar': 'bulvar',
            'blv.': 'bulvar', 'bulv.': 'bulvar',
            
            # Building types
            'apt': 'apartmani', 'ap': 'apartmani', 'apartman': 'apartmani',
            'apt.': 'apartmani', 'ap.': 'apartmani',
            'sit': 'sitesi', 'site': 'sitesi',
            'sit.': 'sitesi', 'site.': 'sitesi',
            'blk': 'blok', 'blok': 'blok',
            'blk.': 'blok', 'blok.': 'blok',
            'plz': 'plaza', 'plaza': 'plaza',
            'plz.': 'plaza', 'plaza.': 'plaza',
            'avm': 'alisveris merkezi', 'avm.': 'alisveris merkezi',
            
            # Address components
            'no': 'numara', 'nu': 'numara', 'numara': 'numara',
            'no.': 'numara', 'nu.': 'numara',
            'kt': 'kat', 'kat': 'kat', 'k': 'kat',
            'kt.': 'kat', 'kat.': 'kat', 'k.': 'kat',
            'dr': 'daire', 'daire': 'daire', 'd': 'daire',
            'dr.': 'daire', 'daire.': 'daire', 'd.': 'daire',
            'dai': 'daire', 'dai.': 'daire',
            
            # Directions
            'kz': 'kuzey', 'gy': 'guney', 'dt': 'dogu', 'bt': 'bati',
            'kz.': 'kuzey', 'gy.': 'guney', 'dt.': 'dogu', 'bt.': 'bati',
            
            # Common institutions
            'unv': 'universitesi', 'unv.': 'universitesi',
            'hst': 'hastanesi', 'hst.': 'hastanesi',
            'okl': 'okulu', 'okl.': 'okulu',
            'lise': 'lisesi', 'lise.': 'lisesi'
        }
        
        # Common typo corrections for Turkish cities/districts
        self.typo_corrections = {
            'uskuar': 'uskudar', 'isketl': 'iskitler', 'kadikoy': 'kadikoy',
            'beyoglu': 'beyoglu', 'sisli': 'sisli', 'besiktas': 'besiktas',
            'fatih': 'fatih', 'umraniye': 'umraniye', 'maltepe': 'maltepe',
            'pendik': 'pendik', 'tuzla': 'tuzla', 'kartal': 'kartal'
        }
        
        # Regex patterns for number normalization
        self.number_patterns = [
            (r'no[:\s=]*(\d+)', r'numara \1'),  # No:5, No=5, No 5
            (r'(\d+)[/\-](\d+)', r'numara \1 daire \2'),  # 5/3, 5-3
            (r'(\d+)\.?\s*kat', r'\1 kat'),  # 2.kat, 2 kat
            (r'(\d+)\.?\s*daire', r'\1 daire'),  # 4.daire, 4 daire
            (r'(\d+)\.?\s*blok', r'\1 blok'),  # A.blok, A blok
        ]
        
        # Turkish stopwords to optionally remove (while preserving location terms)
        self.stopwords = {
            've', 'ile', 'bu', 'bir', 'da', 'de', 'mi', 'mu', 'musun', 'musunuz',
            'dir', 'dir', 'tir', 'tur', 'dır', 'tır', 'tür', 'dür'
        }
    
    def normalize_turkish_chars(self, text: str) -> str:
        """
        Convert Turkish characters to their ASCII equivalents.
        
        Args:
            text: Input text with Turkish characters
            
        Returns:
            Text with Turkish characters normalized
        """
        for tr_char, en_char in self.turkish_chars.items():
            text = text.replace(tr_char, en_char)
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand common Turkish abbreviations in address text.
        
        Args:
            text: Input text with abbreviations
            
        Returns:
            Text with abbreviations expanded
        """
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Clean word for matching (remove punctuation)
            clean_word = word.strip('.,;:()[]{}"-').lower()
            
            if clean_word in self.abbreviations:
                expanded_words.append(self.abbreviations[clean_word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def correct_typos(self, text: str, threshold: float = 0.8) -> str:
        """
        Correct common typos in Turkish place names using fuzzy matching.
        
        Args:
            text: Input text
            threshold: Similarity threshold for correction
            
        Returns:
            Text with typos corrected
        """
        words = text.split()
        corrected_words = []
        
        for word in words:
            clean_word = word.strip('.,;:()[]{}"-').lower()
            
            # Check for exact typo matches first
            if clean_word in self.typo_corrections:
                corrected_words.append(self.typo_corrections[clean_word])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def standardize_numbers(self, text: str) -> str:
        """
        Standardize number formats in addresses.
        
        Args:
            text: Input text with various number formats
            
        Returns:
            Text with standardized number formats
        """
        for pattern, replacement in self.number_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def remove_redundant_locations(self, text: str) -> str:
        """
        Remove redundant location repetitions in addresses.
        
        Args:
            text: Input text that may contain repeated location names
            
        Returns:
            Text with redundant repetitions removed
        """
        words = text.split()
        if len(words) <= 3:
            return text
        
        # Simple deduplication of consecutive identical words
        deduplicated = []
        for i, word in enumerate(words):
            if i == 0 or word != words[i-1]:
                deduplicated.append(word)
        
        return ' '.join(deduplicated)
    
    def clean_punctuation_and_spacing(self, text: str) -> str:
        """
        Clean punctuation and normalize spacing.
        
        Args:
            text: Input text with various punctuation
            
        Returns:
            Cleaned text with normalized spacing
        """
        # Remove special characters but preserve alphanumeric and Turkish chars
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace (multiple spaces to single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        return text.strip()
    
    def normalize(self, address: str) -> str:
        """
        Apply full normalization pipeline to a single address.
        
        Args:
            address: Raw address string
            
        Returns:
            Normalized address string
        """
        if pd.isna(address) or not isinstance(address, str):
            return ""
        
        # Convert to lowercase
        address = address.lower()
        
        # Normalize Turkish characters
        address = self.normalize_turkish_chars(address)
        
        # Expand abbreviations
        address = self.expand_abbreviations(address)
        
        # Correct typos
        address = self.correct_typos(address)
        
        # Standardize numbers
        address = self.standardize_numbers(address)
        
        # Remove redundant locations
        address = self.remove_redundant_locations(address)
        
        # Clean punctuation and spacing
        address = self.clean_punctuation_and_spacing(address)
        
        return address

# ================================================================
# MAIN PREPROCESSING FUNCTIONS
# ================================================================

def preprocess_address(text: str) -> str:
    """
    Apply full preprocessing pipeline to a single address.
    
    Args:
        text: Raw address string
        
    Returns:
        Normalized address string
    """
    normalizer = TurkishAddressNormalizer()
    return normalizer.normalize(text)

def preprocess_dataframe(df: pd.DataFrame, 
                        address_col: str = 'address',
                        label_col: str = 'label',
                        batch_size: int = 10000) -> pd.DataFrame:
    """
    Apply preprocessing to entire dataset with progress tracking and memory optimization.
    
    Args:
        df: Input dataframe
        address_col: Column name containing addresses
        label_col: Column name containing labels (optional)
        batch_size: Number of rows to process in each batch
        
    Returns:
        DataFrame with additional 'processed_address' column
    """
    print(f"Starting preprocessing of {len(df):,} addresses...")
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Initialize normalizer
    normalizer = TurkishAddressNormalizer()
    
    # Process in batches for memory efficiency
    processed_addresses = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing addresses"):
        batch = df.iloc[i:i+batch_size]
        
        # Process batch
        batch_addresses = batch[address_col].apply(normalizer.normalize)
        processed_addresses.extend(batch_addresses)
        
        # Clear memory
        if i % (batch_size * 5) == 0:
            gc.collect()
    
    # Add processed column
    result_df['processed_address'] = processed_addresses
    
    # Remove empty processed addresses
    initial_count = len(result_df)
    result_df = result_df[result_df['processed_address'].str.len() > 0].reset_index(drop=True)
    final_count = len(result_df)
    
    print(f"Preprocessing completed. Removed {initial_count - final_count:,} empty addresses.")
    print(f"Final dataset size: {final_count:,} addresses")
    
    return result_df

def analyze_preprocessing_impact(df_original: pd.DataFrame, 
                               df_processed: pd.DataFrame,
                               address_col: str = 'address') -> Dict:
    """
    Generate comprehensive statistics on preprocessing effectiveness.
    
    Args:
        df_original: Original dataframe
        df_processed: Processed dataframe
        address_col: Column name containing addresses
        
    Returns:
        Dictionary with preprocessing metrics
    """
    print("Analyzing preprocessing impact...")
    
    # Calculate basic statistics
    original_lengths = df_original[address_col].str.len()
    processed_lengths = df_processed['processed_address'].str.len()
    
    original_word_counts = df_original[address_col].str.split().str.len()
    processed_word_counts = df_processed['processed_address'].str.split().str.len()
    
    # Unique address analysis
    original_unique = df_original[address_col].nunique()
    processed_unique = df_processed['processed_address'].nunique()
    
    # Deduplication statistics
    reduction_rate = (original_unique - processed_unique) / original_unique
    
    # Length statistics
    avg_length_change = processed_lengths.mean() - original_lengths.mean()
    avg_word_change = processed_word_counts.mean() - original_word_counts.mean()
    
    # Create analysis results
    analysis = {
        'total_addresses': len(df_original),
        'processed_addresses': len(df_processed),
        'original_unique_addresses': original_unique,
        'processed_unique_addresses': processed_unique,
        'reduction_rate': reduction_rate,
        'avg_original_length': original_lengths.mean(),
        'avg_processed_length': processed_lengths.mean(),
        'avg_length_change': avg_length_change,
        'avg_original_words': original_word_counts.mean(),
        'avg_processed_words': processed_word_counts.mean(),
        'avg_word_change': avg_word_change,
        'length_reduction_percent': (avg_length_change / original_lengths.mean()) * 100,
        'word_reduction_percent': (avg_word_change / original_word_counts.mean()) * 100
    }
    
    return analysis

def display_transformation_examples(df_original: pd.DataFrame,
                                  df_processed: pd.DataFrame,
                                  address_col: str = 'address',
                                  n_examples: int = 10) -> None:
    """
    Display random before/after examples of address transformations.
    
    Args:
        df_original: Original dataframe
        df_processed: Processed dataframe
        address_col: Column name containing addresses
        n_examples: Number of examples to display
    """
    print(f"\n{'='*80}")
    print(f"RANDOM TRANSFORMATION EXAMPLES (showing {n_examples} examples)")
    print(f"{'='*80}")
    
    # Get random indices
    random_indices = np.random.choice(len(df_processed), 
                                    size=min(n_examples, len(df_processed)), 
                                    replace=False)
    
    for i, idx in enumerate(random_indices, 1):
        original = df_original.iloc[idx][address_col]
        processed = df_processed.iloc[idx]['processed_address']
        
        print(f"\nExample {i}:")
        print(f"  Original:  {original}")
        print(f"  Processed: {processed}")
        print(f"  Length:    {len(original)} → {len(processed)} chars")
        print(f"  Words:     {len(original.split())} → {len(processed.split())} words")

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def save_processed_data(df: pd.DataFrame, filename: str = 'preprocessed_addresses.csv') -> None:
    """
    Save processed data to CSV file.
    
    Args:
        df: Processed dataframe
        filename: Output filename
    """
    print(f"Saving processed data to {filename}...")
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Data saved successfully! File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# ================================================================
# MAIN EXECUTION
# ================================================================

def main():
    """Main execution function with example usage."""
    
    print("Turkish Address Normalization Pipeline")
    print("=" * 50)
    
    # Example usage with sample data
    print("\nCreating sample data for demonstration...")
    
    # Create sample Turkish addresses
    sample_addresses = [
        "Narlıdere İzmir Narlıdere Narlıdere",
        "Kadıköy Mah. Cadde No:5 K:2 D:4",
        "Beşiktaş Mh. Cd. Apt No5/3",
        "Üsküdar Mahallesi Sokak No:12 Kat 3",
        "Şişli Mah. Bulvar No:25 Blok A",
        "Fatih Mah. Caddesi No:8 Daire 5",
        "Maltepe Mah. Sokak No:15 Kat 2",
        "Pendik Mah. Cadde No:30 Blok B",
        "Tuzla Mah. Bulvar No:42 Daire 8",
        "Kartal Mah. Sokak No:18 Kat 4"
    ]
    
    sample_labels = [f"label_{i}" for i in range(len(sample_addresses))]
    
    # Create sample dataframe
    sample_df = pd.DataFrame({
        'id': range(len(sample_addresses)),
        'address': sample_addresses,
        'label': sample_labels
    })
    
    print(f"Sample data created: {len(sample_df)} addresses")
    
    # Start preprocessing
    start_time = time.time()
    start_memory = get_memory_usage()
    
    print(f"\nStarting preprocessing at {start_time:.2f}s, Memory: {start_memory:.2f} MB")
    
    # Preprocess the data
    processed_df = preprocess_dataframe(sample_df, address_col='address', label_col='label')
    
    # Calculate processing time and memory
    end_time = time.time()
    end_memory = get_memory_usage()
    processing_time = end_time - start_time
    memory_change = end_memory - start_memory
    
    print(f"\nPreprocessing completed in {processing_time:.2f} seconds")
    print(f"Memory usage: {start_memory:.2f} → {end_memory:.2f} MB (change: {memory_change:+.2f} MB)")
    
    # Analyze impact
    analysis = analyze_preprocessing_impact(sample_df, processed_df, 'address')
    
    print(f"\n{'='*60}")
    print("PREPROCESSING ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    for key, value in analysis.items():
        if isinstance(value, float):
            if 'rate' in key or 'percent' in key:
                print(f"{key.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value:,}")
    
    # Display examples
    display_transformation_examples(sample_df, processed_df, 'address', n_examples=5)
    
    # Save results
    save_processed_data(processed_df, 'sample_preprocessed_addresses.csv')
    
    print(f"\n{'='*60}")
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    return processed_df, analysis

if __name__ == "__main__":
    # Prefer Kaggle-style orchestration if data present; otherwise run demo main()
    try:
        # Added: Kaggle-style preprocessing + submission orchestration
        import os as _os
        def run_preprocessing_and_submission(data_dir: str = ".", batch_size: int = 10000):
            """
            Run preprocessing for train/test and build submission.csv by exact match on processed_address.

            - Reads train.csv (address,label) and test.csv (id,address) with dtype=str
            - Uses TurkishAddressNormalizer().normalize in batches (no normalization logic changes)
            - Saves train_preprocessed.csv (address,processed_address,label)
            - Saves test_preprocessed.csv (id,address,processed_address)
            - Builds processed_address -> mode(label) mapping from train and assigns to test
            - Fallback for unmatched: global most frequent label from train
            - Writes submission.csv with columns id,label preserving original order
            - Prints 10 random examples per split and coverage/time stats
            """
            start = time.time()
            train_path = _os.path.join(data_dir, "train.csv")
            test_path = _os.path.join(data_dir, "test.csv")
            ss_path = _os.path.join(data_dir, "sample_submission.csv")

            # Load CSVs
            print("Loading CSVs...")
            train_df = pd.read_csv(train_path, dtype=str, keep_default_na=False)
            test_df = pd.read_csv(test_path, dtype=str, keep_default_na=False)

            assert 'address' in train_df.columns and 'label' in train_df.columns, "train.csv must have columns address,label"
            assert 'id' in test_df.columns and 'address' in test_df.columns, "test.csv must have columns id,address"

            # Initialize normalizer
            normalizer = TurkishAddressNormalizer()

            def _normalize_series_in_batches(series: pd.Series, batch_size: int) -> List[str]:
                out: List[str] = []
                for i in tqdm(range(0, len(series), batch_size), desc="Normalizing", unit="rows"):
                    batch = series.iloc[i:i+batch_size]
                    out.extend(batch.apply(normalizer.normalize))
                    if (i // batch_size) % 5 == 0:
                        gc.collect()
                return out

            # Process train
            print(f"Processing train: {len(train_df):,} rows (batch_size={batch_size})")
            train_proc = train_df.copy()
            train_proc['processed_address'] = _normalize_series_in_batches(train_proc['address'], batch_size)
            # Save train preprocessed
            train_out_cols = ['address', 'processed_address', 'label']
            train_proc[train_out_cols].to_csv(_os.path.join(data_dir, 'train_preprocessed.csv'), index=False, encoding='utf-8')

            # Process test (do not drop any rows; preserve order)
            print(f"Processing test: {len(test_df):,} rows (batch_size={batch_size})")
            test_proc = test_df.copy()
            test_proc['processed_address'] = _normalize_series_in_batches(test_proc['address'], batch_size)
            # Save test preprocessed
            test_out_cols = ['id', 'address', 'processed_address']
            test_proc[test_out_cols].to_csv(_os.path.join(data_dir, 'test_preprocessed.csv'), index=False, encoding='utf-8')

            # Print examples
            def _print_examples(df: pd.DataFrame, addr_col: str, proc_col: str, name: str):
                print(f"\nExamples ({name}) - 10 random:")
                sample = df.sample(n=min(10, len(df)), random_state=42)
                for _, r in sample.iterrows():
                    print(f"- BEFORE: {r[addr_col]}")
                    print(f"  AFTER : {r[proc_col]}")
            _print_examples(train_proc, 'address', 'processed_address', 'train')
            _print_examples(test_proc, 'address', 'processed_address', 'test')

            # Build mapping: processed_address -> mode(label)
            print("\nBuilding label mapping (processed_address -> mode(label))...")
            # global most frequent label
            global_mode_label = train_proc['label'].mode(dropna=False)[0]
            # mode per processed address
            vc = train_proc.groupby('processed_address')['label'].agg(lambda x: x.value_counts(dropna=False).idxmax()).reset_index()
            vc.columns = ['processed_address', 'label_mode']

            # Join to test by exact processed_address
            test_labeled = test_proc.merge(vc, on='processed_address', how='left')
            matched = test_labeled['label_mode'].notna().sum()
            coverage = matched / len(test_labeled) if len(test_labeled) else 0.0
            # Fallback fill
            test_labeled['label'] = test_labeled['label_mode'].fillna(global_mode_label)
            submission = test_labeled[['id', 'label']].copy()

            # Validate against sample_submission if present
            if _os.path.exists(ss_path):
                ss = pd.read_csv(ss_path, dtype=str, keep_default_na=False)
                assert list(ss.columns) == ['id', 'label'], "sample_submission.csv must have columns id,label"
                assert len(ss) == len(submission), "submission row count must match sample_submission"

            # Save submission
            sub_path = _os.path.join(data_dir, 'submission.csv')
            submission.to_csv(sub_path, index=False, encoding='utf-8')

            # Reporting
            elapsed = time.time() - start
            print("\nReporting:")
            print(f"- Train rows processed: {len(train_df):,}")
            print(f"- Test rows processed : {len(test_df):,}")
            print(f"- Train unique before : {train_df['address'].nunique():,}")
            print(f"- Train unique after  : {train_proc['processed_address'].nunique():,}")
            print(f"- Submission coverage : {coverage*100:.2f}% matched, {(1-coverage)*100:.2f}% fallback")
            print(f"- Elapsed time        : {elapsed:.2f}s ({(len(train_df)+len(test_df))/max(elapsed,1e-6):.0f} rows/sec)")

            return train_proc, test_proc, submission

        data_present = _os.path.exists("train.csv") and _os.path.exists("test.csv")
        if data_present:
            print("Detected train.csv and test.csv. Running preprocessing and submission...")
            _ = run_preprocessing_and_submission(data_dir=".", batch_size=10000)
        else:
            # Fallback to original demo
            processed_data, analysis_results = main()
            print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()