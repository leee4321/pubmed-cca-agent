"""
Data Loader module for loading and processing CCA bootstrap results,
FreeSurfer labels, and paper context files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class CCALoadingResult:
    """Represents a single variable's loading result from CCA bootstrap."""
    variable_name: str
    loading_estimate: float
    ci_68_low: float
    ci_68_upper: float
    ci_95_low: float
    ci_95_upper: float
    ci_99_low: float
    ci_99_upper: float
    ci_997_low: float
    ci_997_upper: float
    consistency: float

    @property
    def is_significant_95(self) -> bool:
        """Check if loading is significant (95% CI doesn't cross zero)."""
        return (self.ci_95_low > 0 and self.ci_95_upper > 0) or \
               (self.ci_95_low < 0 and self.ci_95_upper < 0)

    @property
    def is_significant_99(self) -> bool:
        """Check if loading is significant (99% CI doesn't cross zero)."""
        return (self.ci_99_low > 0 and self.ci_99_upper > 0) or \
               (self.ci_99_low < 0 and self.ci_99_upper < 0)

    @property
    def direction(self) -> str:
        """Return the direction of the loading."""
        if self.loading_estimate > 0:
            return "positive"
        elif self.loading_estimate < 0:
            return "negative"
        return "zero"


@dataclass
class FreeSurferLabel:
    """Represents a FreeSurfer brain region label."""
    index: int
    abbreviation: str
    full_name: str
    is_cortex: bool


@dataclass
class CCAResults:
    """Container for all CCA results and related data."""
    x_loadings: List[CCALoadingResult]
    y_loadings: List[CCALoadingResult]
    freesurfer_labels: Dict[str, FreeSurferLabel]
    analysis_description: str
    introduction: str
    methods: str

    def get_significant_x_loadings(self, threshold: str = "95") -> List[CCALoadingResult]:
        """Get X loadings that are statistically significant."""
        if threshold == "95":
            return [l for l in self.x_loadings if l.is_significant_95]
        elif threshold == "99":
            return [l for l in self.x_loadings if l.is_significant_99]
        return self.x_loadings

    def get_significant_y_loadings(self, threshold: str = "95") -> List[CCALoadingResult]:
        """Get Y loadings that are statistically significant."""
        if threshold == "95":
            return [l for l in self.y_loadings if l.is_significant_95]
        elif threshold == "99":
            return [l for l in self.y_loadings if l.is_significant_99]
        return self.y_loadings

    def get_top_x_loadings(self, n: int = 10, by_absolute: bool = True) -> List[CCALoadingResult]:
        """Get top N X loadings by loading magnitude."""
        if by_absolute:
            sorted_loadings = sorted(self.x_loadings, key=lambda x: abs(x.loading_estimate), reverse=True)
        else:
            sorted_loadings = sorted(self.x_loadings, key=lambda x: x.loading_estimate, reverse=True)
        return sorted_loadings[:n]

    def get_top_y_loadings(self, n: int = 10, by_absolute: bool = True) -> List[CCALoadingResult]:
        """Get top N Y loadings by loading magnitude."""
        if by_absolute:
            sorted_loadings = sorted(self.y_loadings, key=lambda x: abs(x.loading_estimate), reverse=True)
        else:
            sorted_loadings = sorted(self.y_loadings, key=lambda x: x.loading_estimate, reverse=True)
        return sorted_loadings[:n]

    def get_positive_x_loadings(self, min_value: float = 0.1) -> List[CCALoadingResult]:
        """Get X loadings with positive values above threshold."""
        return [l for l in self.x_loadings if l.loading_estimate >= min_value]

    def get_negative_x_loadings(self, max_value: float = -0.1) -> List[CCALoadingResult]:
        """Get X loadings with negative values below threshold."""
        return [l for l in self.x_loadings if l.loading_estimate <= max_value]

    def get_brain_region_name(self, abbrev: str) -> str:
        """Get full brain region name from abbreviation."""
        if abbrev in self.freesurfer_labels:
            return self.freesurfer_labels[abbrev].full_name
        return abbrev


def load_bootstrap_csv(filepath: str) -> List[CCALoadingResult]:
    """Load bootstrap result CSV file and return list of CCALoadingResult."""
    df = pd.read_csv(filepath, index_col=0)

    results = []
    for idx, row in df.iterrows():
        result = CCALoadingResult(
            variable_name=str(idx),
            loading_estimate=row['loading_comp1_estimate'],
            ci_68_low=row['loading_comp1_68%_low'],
            ci_68_upper=row['loading_comp1_68%_upper'],
            ci_95_low=row['loading_comp1_95%_low'],
            ci_95_upper=row['loading_comp1_95%_upper'],
            ci_99_low=row['loading_comp1_99%_low'],
            ci_99_upper=row['loading_comp1_99%_upper'],
            ci_997_low=row['loading_comp1_99.7%_low'],
            ci_997_upper=row['loading_comp1_99.7%_upper'],
            consistency=row['loading_comp1_consistency']
        )
        results.append(result)

    return results


def load_freesurfer_labels(filepath: str) -> Dict[str, FreeSurferLabel]:
    """Load FreeSurfer label CSV and return dictionary mapping abbreviation to label."""
    df = pd.read_csv(filepath, index_col=0)

    labels = {}
    for idx, row in df.iterrows():
        label = FreeSurferLabel(
            index=int(idx),
            abbreviation=row['feat'],
            full_name=row['label_freesurfer(Full name)'],
            is_cortex=bool(row['cortex'])
        )
        labels[label.abbreviation] = label

    return labels


def load_text_file(filepath: str) -> str:
    """Load a text file and return its contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_cca_results(
    x_loading_path: str,
    y_loading_path: str,
    freesurfer_label_path: str,
    analysis_description_path: str,
    introduction_path: str,
    methods_path: str
) -> CCAResults:
    """
    Load all CCA results and context files.

    Args:
        x_loading_path: Path to X loading bootstrap results CSV
        y_loading_path: Path to Y loading bootstrap results CSV
        freesurfer_label_path: Path to FreeSurfer labels CSV
        analysis_description_path: Path to analysis description text file
        introduction_path: Path to Introduction text file
        methods_path: Path to Methods text file

    Returns:
        CCAResults object containing all loaded data
    """
    x_loadings = load_bootstrap_csv(x_loading_path)
    y_loadings = load_bootstrap_csv(y_loading_path)
    freesurfer_labels = load_freesurfer_labels(freesurfer_label_path)
    analysis_description = load_text_file(analysis_description_path)
    introduction = load_text_file(introduction_path)
    methods = load_text_file(methods_path)

    return CCAResults(
        x_loadings=x_loadings,
        y_loadings=y_loadings,
        freesurfer_labels=freesurfer_labels,
        analysis_description=analysis_description,
        introduction=introduction,
        methods=methods
    )


def load_default_cca_results(base_dir: str = "input") -> CCAResults:
    """
    Load CCA results from default file paths.

    Args:
        base_dir: Base directory containing all the data files (default: input)

    Returns:
        CCAResults object
    """
    base_path = Path(base_dir)

    return load_cca_results(
        x_loading_path=str(base_path / "bootstrap_result_summary_x_loading_comp1.csv"),
        y_loading_path=str(base_path / "bootstrap_result_summary_y_loading_comp1.csv"),
        freesurfer_label_path=str(base_path / "FreeSurfer_label.csv"),
        analysis_description_path=str(base_path / "analysis_results_description.txt"),
        introduction_path=str(base_path / "Introduction.txt"),
        methods_path=str(base_path / "Methods.txt")
    )


# ============================================================
# Helper functions for interpreting Y loadings
# ============================================================

def parse_y_variable_name(var_name: str, freesurfer_labels: Dict[str, FreeSurferLabel]) -> Dict:
    """
    Parse Y variable name to extract metric type and brain region.

    Y variables follow the pattern: {metric}_{region} or just {metric} for global measures.

    Returns:
        Dictionary with 'metric_type', 'region_abbrev', 'region_full_name', 'is_global'
    """
    # Global metrics (no underscore or density)
    global_metrics = ['density', 'modularity', 'norm_modularity', 'norm_avg_clust_coef',
                      'norm_char_path_len', 'global_efficiency', 'norm_global_efficiency',
                      'small_worldness']

    if var_name in global_metrics:
        return {
            'metric_type': var_name,
            'region_abbrev': None,
            'region_full_name': None,
            'is_global': True
        }

    # Regional metrics: {metric}_{region}
    parts = var_name.split('_', 1)
    if len(parts) == 2:
        metric_abbrev = parts[0]
        region_abbrev = parts[1]

        # Map metric abbreviations
        metric_map = {
            'deg': 'degree',
            'stren': 'strength',
            'norm_Cc': 'nodal efficiency',
            'norm_clust_coef': 'clustering coefficient',
            'BC': 'betweenness centrality'
        }

        metric_type = metric_map.get(metric_abbrev, metric_abbrev)

        # Get full region name
        region_full_name = region_abbrev
        if region_abbrev in freesurfer_labels:
            region_full_name = freesurfer_labels[region_abbrev].full_name

        return {
            'metric_type': metric_type,
            'region_abbrev': region_abbrev,
            'region_full_name': region_full_name,
            'is_global': False
        }

    return {
        'metric_type': var_name,
        'region_abbrev': None,
        'region_full_name': None,
        'is_global': True
    }


def categorize_y_loadings(
    y_loadings: List[CCALoadingResult],
    freesurfer_labels: Dict[str, FreeSurferLabel]
) -> Dict[str, List[Tuple[CCALoadingResult, Dict]]]:
    """
    Categorize Y loadings by metric type.

    Returns:
        Dictionary mapping metric type to list of (loading, parsed_info) tuples
    """
    categorized = {}

    for loading in y_loadings:
        parsed = parse_y_variable_name(loading.variable_name, freesurfer_labels)
        metric_type = parsed['metric_type']

        if metric_type not in categorized:
            categorized[metric_type] = []

        categorized[metric_type].append((loading, parsed))

    return categorized


def get_summary_statistics(loadings: List[CCALoadingResult]) -> Dict:
    """
    Get summary statistics for a list of loadings.
    """
    if not loadings:
        return {}

    estimates = [l.loading_estimate for l in loadings]

    return {
        'count': len(loadings),
        'mean': np.mean(estimates),
        'std': np.std(estimates),
        'min': np.min(estimates),
        'max': np.max(estimates),
        'n_positive': sum(1 for e in estimates if e > 0),
        'n_negative': sum(1 for e in estimates if e < 0),
        'n_significant_95': sum(1 for l in loadings if l.is_significant_95),
        'n_significant_99': sum(1 for l in loadings if l.is_significant_99)
    }


if __name__ == "__main__":
    # Test loading
    print("Testing data loader...")

    try:
        results = load_default_cca_results("input")

        print(f"\nLoaded {len(results.x_loadings)} X loadings")
        print(f"Loaded {len(results.y_loadings)} Y loadings")
        print(f"Loaded {len(results.freesurfer_labels)} FreeSurfer labels")

        print("\n--- Top 5 X loadings (by absolute value) ---")
        for loading in results.get_top_x_loadings(5):
            print(f"  {loading.variable_name}: {loading.loading_estimate:.4f} "
                  f"(95% CI: [{loading.ci_95_low:.4f}, {loading.ci_95_upper:.4f}])")

        print("\n--- Significant X loadings (95% CI) ---")
        sig_x = results.get_significant_x_loadings("95")
        print(f"  Count: {len(sig_x)}")

        print("\n--- Y loading summary statistics ---")
        stats = get_summary_statistics(results.y_loadings)
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
