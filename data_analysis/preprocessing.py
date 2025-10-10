"""
Comprehensive EDA and Dataset Issue Detection for MHC-Peptide Binding

This script provides:
1. Dataset statistics and distributions
2. Automated issue detection
3. Visualizations
4. Recommendations for handling problems
"""

import re
from typing import Optional, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)

# ================================
# HLA Allele Normalization Helpers
# ================================
# Please note that As of September 2025 no alleles have been named with the `C` or `A` suffixes.
EXPR_SUFFIXES = {"N", "L", "S", "Q"}

# Flexible parser for common variants:
# Important note: the test set does not include the seperator '*' character
ALLELE_RE = re.compile(
    r"""
    ^\s*
    (?:HLA[-\s]?)?                         # optional HLA- prefix
    (?P<locus>[A-Za-z]{1,3}\d?[A-Za-z]?)   # locus: A, B, C, DRB1, etc.
    \s*\*?\s*                              # Handle missing "*"
    (?P<digits>[\d:]+|\d+)
    (?P<suffix>[NLSQ])?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE
)


def _split_digits(d: str) -> Tuple[str, ...]:
    """Turn '0201' or '02:01:01:02' into ('02','01',...)."""
    parts = d.split(":") if ":" in d else [d[i:i + 2] for i in range(0, len(d), 2)]
    return tuple(p.zfill(2) for p in parts if p)


def normalize_allele(
    allele_raw: str,
    level: int = 2,
    keep_suffix: bool = False,
    force_prefix: bool = True
) -> Optional[Dict[str, str]]:
    """
    Normalize HLA allele to canonical 'HLA-Locus*XX:YY' form (two-field by default).
    Returns dict with canonical strings and components, or None if unparsable.
    """
    if not isinstance(allele_raw, str) or not allele_raw.strip():
        return None
    m = ALLELE_RE.match(allele_raw)
    if not m:
        return None

    locus = m.group("locus").upper()
    digits = m.group("digits")
    suffix = (m.group("suffix") or "").upper()

    groups = _split_digits(digits)
    if not groups:
        return None
    level = max(1, min(level, 4))
    groups = groups[:level]

    canonical_digits = ":".join(groups)
    prefix = "HLA-" if force_prefix else ""
    core = f"{prefix}{locus}*{canonical_digits}"
    full = f"{core}{suffix if (keep_suffix and suffix in EXPR_SUFFIXES) else ''}"

    return {
        "raw": allele_raw,
        "locus": locus,
        "fields": canonical_digits,            # e.g., '02:01'
        "suffix": suffix,                      # N/L/S/Q or ''
        "canonical": core,                     # 'HLA-A*02:01'
        "canonical_with_suffix": full,         # 'HLA-A*02:01N' if kept
    }


def apply_allele_normalization(df: pd.DataFrame, allele_col: str = "allele") -> pd.DataFrame:
    """
    Adds:
      - allele_raw: original string
      - allele_canonical: normalized two-field 'HLA-X*XX:YY'
      - allele_suffix: expression suffix (N/L/S/Q or '')
      - allele_parse_ok: boolean
      - gene: extracted A/B/C from canonical
    """
    def canon2(a):
        out = normalize_allele(str(a), level=2, keep_suffix=False)
        return out["canonical"] if out else None

    def suff(a):

        out = normalize_allele(str(a), level=4, keep_suffix=True)
        return out["suffix"] if out else ""

    df = df.copy()
    df["allele_raw"] = df[allele_col].astype(str)
    df["allele_canonical"] = df[allele_col].astype(str).map(canon2)
    df["allele_suffix"] = df[allele_col].astype(str).map(suff)
    df["allele_parse_ok"] = df["allele_canonical"].notna()
    df["gene"] = df["allele_canonical"].str.extract(r"HLA-([ABC])", expand=False)
    return df


class MHCDatasetAnalyzer:
    """
    Comprehensive analyzer for MHC-peptide binding datasets.
    """

    def __init__(self, train_df, test_df):
        """
        Args:
            train_df: Training dataframe with columns ['peptide', 'allele', 'hit']
            test_df: Test dataframe with same columns
        """
        # Normalize alleles first (train & test)
        self.train_df = apply_allele_normalization(train_df.copy(), allele_col="allele")
        self.test_df = apply_allele_normalization(test_df.copy(), allele_col="allele")

        # ============================================================
        # Remove duplicates
        # ============================================================
        initial_train_len = len(self.train_df)
        initial_test_len = len(self.test_df)

        self.train_df = self.train_df.drop_duplicates(subset=["peptide", "allele_canonical"])
        self.test_df = self.test_df.drop_duplicates(subset=["peptide", "allele_canonical"])

        removed_train = initial_train_len - len(self.train_df)
        removed_test = initial_test_len - len(self.test_df)

        if removed_train > 0 or removed_test > 0:
            print(f"Removed {removed_train} duplicate samples from training data.")
            print(f"Removed {removed_test} duplicate samples from test data.")
        else:
            print(" No duplicate samples found.")
        # ============================================================

        # Add derived columns
        self.train_df["length"] = self.train_df["peptide"].str.len()
        self.test_df["length"] = self.test_df["peptide"].str.len()

        # Store issues found
        self.issues = []
        self.warnings = []
        self.info = []

    def print_section(self, title, symbol="="):
        """Print formatted section header."""
        print(f"\n{symbol * 80}")
        print(f"{title:^80}")
        print(f"{symbol * 80}\n")

    # ========================================================================
    # BASIC STATISTICS
    # ========================================================================
    def analyze_basic_statistics(self):
        """Analyze basic dataset statistics."""
        self.print_section("BASIC DATASET STATISTICS")

        print(f"{'Metric':<40} {'Training':<20} {'Test':<20}")
        print("-" * 80)

        # Sample counts
        print(f"{'Total samples':<40} {len(self.train_df):<20,} {len(self.test_df):<20,}")

        # Positive/Negative counts
        train_pos = self.train_df["hit"].sum()
        train_neg = len(self.train_df) - train_pos
        test_pos = self.test_df["hit"].sum()
        test_neg = len(self.test_df) - test_pos

        print(f"{'Positive samples (binders)':<40} {train_pos:<20,} {test_pos:<20,}")
        print(f"{'Negative samples (non-binders)':<40} {train_neg:<20,} {test_neg:<20,}")

        # Positive ratio
        train_ratio = self.train_df["hit"].mean()
        test_ratio = self.test_df["hit"].mean()
        print(f"{'Positive ratio':<40} {train_ratio:<20.4f} {test_ratio:<20.4f}")

        # Check for class imbalance
        if train_ratio < 0.3 or train_ratio > 0.7:
            severity = "SEVERE" if train_ratio < 0.15 or train_ratio > 0.85 else "MODERATE"
            self.issues.append(f"{severity} class imbalance: {train_ratio:.2%} positive samples")

        # Unique values
        print(
            f"{'Unique peptides':<40} "
            f"{self.train_df['peptide'].nunique():<20,} "
            f"{self.test_df['peptide'].nunique():<20,}"
        )
        print(
            f"{'Unique alleles (raw)':<40} "
            f"{self.train_df['allele'].nunique():<20,} "
            f"{self.test_df['allele'].nunique():<20,}"
        )

        print(
            f"{'Unique alleles (canonical 2-field)':<40} "
            f"{self.train_df['allele_canonical'].nunique():<20,} "
            f"{self.test_df['allele_canonical'].nunique():<20,}"
        )

        # Missing values
        train_missing = self.train_df.isnull().sum().sum()
        test_missing = self.test_df.isnull().sum().sum()
        print(f"{'Missing values':<40} {train_missing:<20,} {test_missing:<20,}")

        if train_missing > 0 or test_missing > 0:
            self.issues.append(
                f"Missing values detected: "
                f"train={train_missing}, test={test_missing}"
            )

        # Duplicates (use canonical allele to avoid format-artifact dupes)
        train_dupes = self.train_df.duplicated(subset=["peptide", "allele_canonical"]).sum()
        test_dupes = self.test_df.duplicated(subset=["peptide", "allele_canonical"]).sum()
        print(f"{'Duplicate (peptide, allele) pairs (canonical)':<40} {train_dupes:<20,} {test_dupes:<20,}")

        if train_dupes > 0:
            self.warnings.append(f"Duplicate training samples detected (canonical): {train_dupes}")

    # ========================================================================
    # PEPTIDE LENGTH ANALYSIS
    # ========================================================================
    def analyze_peptide_lengths(self):
        """Analyze peptide length distribution."""
        self.print_section("PEPTIDE LENGTH ANALYSIS")

        # Basic statistics
        print("Length Statistics:")
        print(f"{'Dataset':<15} {'Min':<8} {'Max':<8} {'Mean':<8} {'Median':<8} {'Std':<8}")
        print("-" * 80)

        train_stats = self.train_df["length"].describe()
        test_stats = self.test_df["length"].describe()

        print(f"{'Training':<15} {train_stats['min']:<8.0f} {train_stats['max']:<8.0f} "
              f"{train_stats['mean']:<8.2f} {train_stats['50%']:<8.0f} {train_stats['std']:<8.2f}")
        print(f"{'Test':<15} {test_stats['min']:<8.0f} {test_stats['max']:<8.0f} "
              f"{test_stats['mean']:<8.2f} {test_stats['50%']:<8.0f} {test_stats['std']:<8.2f}")

        # Check for unusual length distribution
        if train_stats["std"] > 2:
            self.warnings.append(f"High length variability (std={train_stats['std']:.2f})")

        # Check if test lengths are outside training range
        train_min, train_max = self.train_df["length"].min(), self.train_df["length"].max()
        test_min, test_max = self.test_df["length"].min(), self.test_df["length"].max()

        if test_min < train_min or test_max > train_max:
            self.issues.append(
                f"Test lengths outside training range: train=[{train_min},{train_max}], test=[{test_min},{test_max}]")

        # Identify lengths with few samples
        for length in sorted(self.train_df["length"].unique()):
            count = (self.train_df["length"] == length).sum()
            if count < 500:
                print(f"  Length {length}: {count} samples")
                self.warnings.append(f"Length {length} has only {count} training samples")

    # ========================================================================
    # DATA QUALITY CHECKS
    # ========================================================================

    def check_data_quality(self):
        """Check for data quality issues."""
        self.print_section("DATA QUALITY CHECKS")

        checks_passed = 0
        checks_total = 0

        # Check 1: Valid peptide sequences
        checks_total += 1
        print("Check 1: Valid peptide sequences")
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

        invalid_peptides = []
        for idx, peptide in enumerate(self.train_df["peptide"]):
            if not all(aa in valid_aa for aa in peptide):
                invalid_peptides.append((idx, peptide))
                if len(invalid_peptides) <= 5:
                    print(f"  Invalid: {peptide}")

        if len(invalid_peptides) == 0:
            print(" PASS: All peptides contain valid amino acids")
            checks_passed += 1
        else:
            print(f"  FAIL: {len(invalid_peptides)} peptides contain invalid characters")
            self.issues.append(f"{len(invalid_peptides)} peptides have invalid amino acids")

        # Check 2: Consistent allele naming (normalization success)
        checks_total += 1
        print("\nCheck 2: Consistent allele naming (normalization)")

        train_parse_fail = (~self.train_df["allele_parse_ok"]).sum()
        test_parse_fail = (~self.test_df["allele_parse_ok"]).sum()

        if train_parse_fail == 0 and test_parse_fail == 0:
            print("  PASS: All alleles parsed and normalized to canonical 2-field form")
            checks_passed += 1
        else:
            print(f"  FAIL: Unparsable alleles — train={train_parse_fail}, test={test_parse_fail}")
            bad_examples = self.train_df.loc[~self.train_df["allele_parse_ok"], "allele"].head(5).tolist()
            if bad_examples:
                print("  Examples (train):", bad_examples)
            self.warnings.append(f"Unparsable allele names: train={train_parse_fail}, test={test_parse_fail}")

        # Check 3: Label consistency
        checks_total += 1
        print("\nCheck 3: Label consistency")

        valid_labels = {0, 1, 0.0, 1.0}
        invalid_labels = self.train_df[~self.train_df["hit"].isin(valid_labels)]

        if len(invalid_labels) == 0:
            print("   PASS: All labels are 0 or 1")
            checks_passed += 1
        else:
            print(f"   FAIL: {len(invalid_labels)} samples have invalid labels")
            print("  Unique invalid values:", invalid_labels["hit"].unique())
            self.issues.append(f"{len(invalid_labels)} samples have invalid labels")

        # Check 4: Peptide length range
        checks_total += 1
        print("\nCheck 4: Peptide length within expected range (According to Reynisson et al. 2020) (8-15)")

        unusual_lengths = self.train_df[(self.train_df["length"] < 8) | (self.train_df["length"] > 15)]

        if len(unusual_lengths) == 0:
            print("  ✓ PASS: All peptides are 8-15 amino acids")
            checks_passed += 1
        else:
            print(f"   WARNING: {len(unusual_lengths)} peptides outside 8-15 range")
            length_dist = unusual_lengths["length"].value_counts()
            print("  Distribution:", dict(length_dist))
            self.warnings.append(f"{len(unusual_lengths)} peptides outside typical 8-15 length")

        # Check 5: Train-test consistency
        checks_total += 1
        print("\nCheck 5: Train-test distribution consistency")

        train_pos_ratio = self.train_df["hit"].mean()
        test_pos_ratio = self.test_df["hit"].mean()
        ratio_diff = abs(train_pos_ratio - test_pos_ratio)

        if ratio_diff < 0.05:
            print(f"   PASS: Similar positive ratios (train={train_pos_ratio:.3f}, test={test_pos_ratio:.3f})")
            checks_passed += 1
        else:
            print(f"   WARNING: Different positive ratios (train={train_pos_ratio:.3f}, test={test_pos_ratio:.3f})")
            self.warnings.append(f"Train-test positive ratio differs by {ratio_diff:.3f}")

        # Summary
        print(f"\n{'='*80}")
        print(f"Quality Checks: {checks_passed}/{checks_total} passed")
        print(f"{'='*80}")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    def create_visualizations(self, save_path="detailed_eda_plots.png"):
        """Create comprehensive visualizations."""
        self.print_section("CREATING VISUALIZATIONS")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Class distribution
        ax1 = fig.add_subplot(gs[0, 0])
        class_counts = self.train_df["hit"].value_counts()
        colors = ["#3498db", "#e74c3c"]
        ax1.bar(["Non-binder", "Binder"], class_counts.values, color=colors, alpha=0.7)
        ax1.set_title("Class Distribution (Training)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Count")
        for i, v in enumerate(class_counts.values):
            ax1.text(i, v, f"{v:,}\n({v/len(self.train_df)*100:.1f}%)",
                     ha="center", va="bottom")

        # 3. Top 15 alleles (canonical)
        ax3 = fig.add_subplot(gs[0, 2])
        top_alleles = self.train_df["allele_canonical"].value_counts().head(15)
        ax3.barh(range(len(top_alleles)), top_alleles.values, color="#2ecc71", alpha=0.7)
        ax3.set_yticks(range(len(top_alleles)))
        ax3.set_yticklabels(top_alleles.index, fontsize=9)
        ax3.set_xlabel("Count")
        ax3.set_title("Top 15 Alleles (Canonical)", fontsize=12, fontweight="bold")
        ax3.invert_yaxis()
        ax3.grid(axis="x", alpha=0.3)

        # 4. Binding rate by length
        ax4 = fig.add_subplot(gs[1, 0])
        binding_by_length = self.train_df.groupby("length")["hit"].agg(["mean", "count"])
        binding_by_length = binding_by_length[binding_by_length["count"] > 50]  # Only lengths with >50 samples
        ax4.bar(binding_by_length.index, binding_by_length["mean"] * 100,
                color="#9b59b6", alpha=0.7)
        ax4.set_xlabel("Peptide Length")
        ax4.set_ylabel("Binding Rate (%)")
        ax4.set_title("Binding Rate by Peptide Length", fontsize=12, fontweight="bold")
        ax4.axhline(y=self.train_df["hit"].mean() * 100, color="r",
                    linestyle="--", label="Overall mean")
        ax4.legend()
        ax4.grid(alpha=0.3)

        # 8. Allele gene distribution
        ax8 = fig.add_subplot(gs[2, 1])
        gene_dist = self.train_df["gene"].value_counts()
        colors_gene = ["#e74c3c", "#3498db", "#2ecc71"]
        if len(gene_dist) > 0:
            ax8.pie(gene_dist.values, labels=[f"HLA-{g}" for g in gene_dist.index],
                    autopct="%1.1f%%", colors=colors_gene[:len(gene_dist)], startangle=90)
            ax8.set_title("Allele Gene Distribution", fontsize=12, fontweight="bold")
        else:
            ax8.text(0.5, 0.5, "No gene data", ha="center", va="center")
            ax8.axis("off")

        # 9. Heatmap: Binding rate by length and gene
        ax9 = fig.add_subplot(gs[2, 2])
        pivot_data = self.train_df.pivot_table(
            values="hit",
            index="length",
            columns="gene",
            aggfunc="mean"
        ) * 100
        if pivot_data.size > 0:
            sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="RdYlGn",
                        ax=ax9, cbar_kws={"label": "Binding Rate (%)"})
            ax9.set_title("Binding Rate by Length and Gene", fontsize=12, fontweight="bold")
            ax9.set_xlabel("HLA Gene")
            ax9.set_ylabel("Peptide Length")
        else:
            ax9.text(0.5, 0.5, "No data for heatmap", ha="center", va="center")
            ax9.axis("off")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Visualizations saved to '{save_path}'")
        plt.close()

    # ========================================================================
    # MAIN ANALYSIS RUNNER
    # ========================================================================
    def run_full_analysis(self, create_plots=True):
        """Run complete analysis pipeline."""
        print("\n" + "=" * 80)
        print(" " * 20 + "MHC DATASET COMPREHENSIVE ANALYSIS")
        print("=" * 80)

        # Run all analyses
        self.analyze_basic_statistics()
        self.analyze_peptide_lengths()
        self.check_data_quality()

        # Create visualizations
        if create_plots:
            self.create_visualizations()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80 + "\n")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================
def analyze_mhc_dataset(train_paths, test_path, create_plots=True):
    """
    Convenience function to analyze MHC dataset.

    Args:
        train_paths: List of paths to training fold CSV files
        test_path: Path to test CSV file
        create_plots: Whether to create visualization plots

    Returns:
        analyzer: MHCDatasetAnalyzer object with results
    """
    print("Loading data...")

    # Load training folds
    train_dfs = []
    for i, path in enumerate(train_paths):
        df = pd.read_csv(path)
        df["fold"] = i
        train_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True)

    # Load test data
    test_df = pd.read_csv(test_path)

    # Create analyzer and run analysis
    analyzer = MHCDatasetAnalyzer(train_df, test_df)
    analyzer.run_full_analysis(create_plots=create_plots)

    return analyzer


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    """
    Example usage - adjust file paths to match your data.
    """

    # Define your file paths
    train_paths = [
        "datasets/fold_0.csv",
        "datasets/fold_1.csv",
        "datasets/fold_2.csv",
        "datasets/fold_3.csv",
        "datasets/fold_4.csv"
    ]
    test_path = "datasets/test.csv"

    # Run analysis
    analyzer = analyze_mhc_dataset(
        train_paths=train_paths,
        test_path=test_path,
        create_plots=True  # Set False to skip visualizations
    )

    print(" Visualizations saved to 'detailed_eda_plots.png'")
