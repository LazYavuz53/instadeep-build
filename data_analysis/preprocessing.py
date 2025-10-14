#!/usr/bin/env python3

"""
01_preprocessing.py
-------------------
This script:
 - Loads train folds and test.csv from 'dataset/' directory
 - Cleans, validates, and saves processed versions
 - Generates distribution plots and metadata
"""

import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------
# Configuration
# -----------------------
DATA_DIR = "datasets"
TRAIN_FOLDS = [f"fold_{i}.csv" for i in range(5)]
TEST_FILE = "test.csv"

TRAIN_PATHS = [os.path.join(DATA_DIR, f) for f in TRAIN_FOLDS]
TEST_PATH = os.path.join(DATA_DIR, TEST_FILE)

# Output directory
CLEAN_DIR = "clean"
os.makedirs(CLEAN_DIR, exist_ok=True)

# Valid amino acids (standard 20)
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_VOCAB)


# -----------------------
# Allele name correction utilities
# -----------------------
def fix_allele_format(allele: str) -> str:
    """Fix allele name to follow standard HLA format: HLA-A*02:101"""
    allele = str(allele).strip().upper()
    valid_pattern = re.compile(r"^HLA-[A-Z]\*\d{2}:\d{2,3}$")

    if valid_pattern.match(allele):
        return allele

    allele = allele.replace(" ", "").replace("_", "").replace("--", "-")

    if not allele.startswith("HLA"):
        allele = "HLA-" + allele
    elif not allele.startswith("HLA-"):
        allele = allele.replace("HLA", "HLA-", 1)

    if "*" not in allele:
        allele = re.sub(r"^(HLA-[A-Z])(\d+)", r"\1*\2", allele)

    allele = re.sub(r"(\*\d{2})(\d{2,3})$", r"\1:\2", allele)
    return allele


def clean_allele_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    print(f"[{name}] Checking and fixing allele formats...")
    valid_pattern = re.compile(r"^HLA-[A-Z]\*\d{2}:\d{2,3}$")
    corrections_made = False
    corrected_alleles = []

    for allele in df["allele"]:
        corrected = fix_allele_format(allele)
        if corrected != allele:
            corrections_made = True
        corrected_alleles.append(corrected)

    if corrections_made:
        print("[DEBUG] Correcting allele naming...")

    df["allele"] = corrected_alleles

    invalid_mask = ~df["allele"].str.match(valid_pattern)
    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        print(f"[{name}] WARNING: {invalid_count} allele entries still invalid after correction.")
    elif corrections_made:
        print(f"[{name}] All allele names corrected successfully.")
    else:
        print(f"[{name}] No mismatch in allele found.")

    return df


# -----------------------
# Utils
# -----------------------
def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    expected_cols = {"peptide", "allele", "hit"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def dataset_stats(df: pd.DataFrame, name: str):
    print(f"\n=== Dataset Stats: {name} ===")
    print(f"Rows: {len(df):,}")
    print("Columns:", list(df.columns))
    print("Unique alleles:", df["allele"].nunique())
    if "hit" in df.columns:
        print("Label distribution (hit):")
        print(df["hit"].value_counts(dropna=False))
    lens = df["peptide"].astype(str).str.len()
    print("Peptide length stats:")
    print(lens.describe())


def remove_exact_duplicates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = len(df)
    df2 = df.drop_duplicates()
    print(f"[{name}] Removed exact duplicate rows: {before - len(df2)}")
    return df2


def remove_conflicting_duplicates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    grp = df.groupby(["peptide", "allele"])["hit"].nunique()
    conflicts = grp[grp > 1]
    if conflicts.empty:
        print(f"[{name}] Conflicting duplicates: none.")
        return df
    before = len(df)
    bad_keys = set(conflicts.index)
    mask = df.set_index(["peptide", "allele"]).index.isin(bad_keys)
    df2 = df[~mask].copy()
    print(f"[{name}] Conflicting (peptide, allele) pairs: {len(conflicts)} | Rows removed: {before - len(df2)}")
    return df2


def handle_missing(df: pd.DataFrame, name: str) -> pd.DataFrame:
    miss = df.isna().sum()
    if miss.sum() == 0:
        print(f"[{name}] Missing values: none.")
        return df
    print(f"[{name}] Missing values per column:\n{miss}")
    before = len(df)
    df2 = df.dropna(subset=["peptide", "allele", "hit"]).copy()
    print(f"[{name}] Rows dropped due to missing peptide/allele/hit: {before - len(df2)}")
    return df2


def is_valid_peptide(seq: str) -> bool:
    s = str(seq).strip().upper()
    if len(s) == 0:
        return False
    return set(s).issubset(AA_SET)


def clean_invalid_peptides(df: pd.DataFrame, name: str) -> pd.DataFrame:
    lens_before = df["peptide"].astype(str).str.len().describe()
    valid_mask = df["peptide"].astype(str).str.upper().apply(is_valid_peptide)
    invalid = (~valid_mask).sum()
    df2 = df[valid_mask].copy()
    print(f"[{name}] Non-standard/invalid peptide rows removed: {invalid}")
    print(f"[{name}] Sequence length stats BEFORE removal:\n{lens_before}")
    print(f"[{name}] Sequence length stats AFTER removal:\n{df2['peptide'].astype(str).str.len().describe()}")
    return df2


def filter_test_alleles_in_train(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_alleles = set(train["allele"].unique())
    test_alleles = set(test["allele"].unique())
    unknown = sorted(list(test_alleles - train_alleles))
    if unknown:
        before = len(test)
        test2 = test[test["allele"].isin(train_alleles)].copy()
        print(f"[TEST] Alleles not present in TRAIN: {unknown}")
        print(f"[TEST] Removed rows with unknown alleles: {before - len(test2)}")
        return test2
    print("[TEST] All test alleles exist in training.")
    return test


def compute_max_len(train_df: pd.DataFrame) -> int:
    max_len = int(train_df["peptide"].astype(str).str.len().max())
    print(f"[FE] Max sequence length (from CLEANED TRAIN): {max_len}")
    return max_len


# -----------------------
# Visualization helpers
# -----------------------
def plot_class_distribution(df, out_prefix, top_n_alleles=15):
    plt.figure(figsize=(4, 4))
    sns.countplot(x="hit", data=df)
    plt.title("General Class Distribution (hit)")
    plt.xlabel("Class (hit)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(CLEAN_DIR, f"{out_prefix}_class_distribution.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    allele_counts = (
        df.groupby(["allele", "hit"])
        .size()
        .reset_index(name="count")
    )
    top_alleles = df["allele"].value_counts().head(top_n_alleles).index
    sns.barplot(
        x="allele", y="count", hue="hit",
        data=allele_counts[allele_counts["allele"].isin(top_alleles)]
    )
    plt.title(f"Class Distribution per Allele (top {top_n_alleles})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(CLEAN_DIR, f"{out_prefix}_class_distribution_per_allele.png"))
    plt.close()


def plot_length_distribution(df, out_prefix, top_n_alleles=15):
    df["length"] = df["peptide"].astype(str).str.len()

    plt.figure(figsize=(5, 4))
    sns.histplot(df["length"], bins=20, kde=True, color="steelblue")
    plt.title("General Peptide Length Distribution")
    plt.xlabel("Peptide length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(CLEAN_DIR, f"{out_prefix}_length_distribution.png"))
    plt.close()

    top_alleles = df["allele"].value_counts().head(top_n_alleles).index
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="allele", y="length",
        data=df[df["allele"].isin(top_alleles)],
        showfliers=False
    )
    plt.title(f"Peptide Length Distribution per Allele (top {top_n_alleles})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(CLEAN_DIR, f"{out_prefix}_length_distribution_per_allele.png"))
    plt.close()


# -----------------------
# Main
# -----------------------
def main():
    train_parts = []
    for p in TRAIN_PATHS:
        df = load_csv_safe(p)
        df["peptide"] = df["peptide"].astype(str).str.upper().str.strip()
        df["allele"] = df["allele"].astype(str).str.strip()
        df = clean_allele_column(df, f"TRAIN file {os.path.basename(p)}")
        train_parts.append(df)
    train_raw = pd.concat(train_parts, ignore_index=True)

    test_raw = load_csv_safe(TEST_PATH)
    test_raw["peptide"] = test_raw["peptide"].astype(str).str.upper().str.strip()
    test_raw["allele"] = test_raw["allele"].astype(str).str.strip()
    test_raw = clean_allele_column(test_raw, "TEST")

    # Stats
    dataset_stats(train_raw, "TRAIN (raw)")
    dataset_stats(test_raw, "TEST (raw)")

    # Cleaning
    train = remove_exact_duplicates(train_raw, "TRAIN")
    test = remove_exact_duplicates(test_raw, "TEST")

    train = remove_conflicting_duplicates(train, "TRAIN")
    if "hit" in test.columns:
        test = remove_conflicting_duplicates(test, "TEST")

    train = handle_missing(train, "TRAIN")
    test = handle_missing(test, "TEST")

    train = clean_invalid_peptides(train, "TRAIN")
    test = clean_invalid_peptides(test, "TEST")

    test = filter_test_alleles_in_train(train, test)

    # Stats after cleaning
    dataset_stats(train, "TRAIN (cleaned)")
    dataset_stats(test, "TEST (cleaned)")

    # Visualization
    print("\n[INFO] Generating distribution plots...")
    plot_class_distribution(train, out_prefix="train")
    plot_length_distribution(train, out_prefix="train")
    print(f"[INFO] Figures saved to {CLEAN_DIR}")

    # Compute max length
    max_seq_len = compute_max_len(train)

    # Save cleaned data & metadata
    train_out = os.path.join(CLEAN_DIR, "train_clean.csv")
    test_out = os.path.join(CLEAN_DIR, "test_clean.csv")
    meta_out = os.path.join(CLEAN_DIR, "metadata.json")

    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)

    metadata = {
        "aa_vocab": AA_VOCAB,
        "max_seq_len": max_seq_len,
        "train_alleles": sorted(train["allele"].unique())
    }
    with open(meta_out, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved cleaned TRAIN to: {train_out}")
    print(f"Saved cleaned TEST  to: {test_out}")
    print(f"Saved metadata to: {meta_out}")
    print(f"Saved figures to: {CLEAN_DIR}")


if __name__ == "__main__":
    main()
