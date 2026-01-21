"""
Results Section Generator for CCA Analysis.

This module generates the Results section of a scientific paper based on
CCA bootstrap results, including statistical summaries and key findings.
"""

import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from google import genai
from dotenv import load_dotenv

from data_loader import (
    CCAResults,
    CCALoadingResult,
    load_default_cca_results,
    parse_y_variable_name,
    categorize_y_loadings,
    get_summary_statistics
)
from pubmed_tool import get_trait_full_name

load_dotenv()


@dataclass
class ResultsSummary:
    """Summary of CCA results for report generation."""
    # X (PGS) loadings
    significant_positive_pgs: List[Tuple[str, float, float, float]]  # (name, estimate, ci_low, ci_high)
    significant_negative_pgs: List[Tuple[str, float, float, float]]
    top_pgs_by_magnitude: List[Tuple[str, float, float, float]]

    # Y (BNM) loadings - Global
    global_network_metrics: Dict[str, Tuple[float, float, float]]  # metric -> (estimate, ci_low, ci_high)

    # Y (BNM) loadings - Regional
    regional_metrics_by_type: Dict[str, List[Tuple[str, str, float, float, float]]]  # metric_type -> [(region, full_name, estimate, ci_low, ci_high)]

    # Overall statistics
    n_total_x: int
    n_significant_x: int
    n_total_y: int
    n_significant_y: int


def extract_results_summary(cca_results: CCAResults, ci_level: str = "95") -> ResultsSummary:
    """
    Extract a structured summary from CCA results.

    Args:
        cca_results: Loaded CCA results
        ci_level: Confidence interval level for significance ("95" or "99")

    Returns:
        ResultsSummary object
    """
    # Get significant X loadings
    sig_x = cca_results.get_significant_x_loadings(ci_level)

    sig_positive_pgs = []
    sig_negative_pgs = []

    for loading in sig_x:
        entry = (
            loading.variable_name,
            loading.loading_estimate,
            loading.ci_95_low if ci_level == "95" else loading.ci_99_low,
            loading.ci_95_upper if ci_level == "95" else loading.ci_99_upper
        )
        if loading.loading_estimate > 0:
            sig_positive_pgs.append(entry)
        else:
            sig_negative_pgs.append(entry)

    # Sort by magnitude
    sig_positive_pgs.sort(key=lambda x: x[1], reverse=True)
    sig_negative_pgs.sort(key=lambda x: x[1])

    # Top PGS by absolute magnitude
    top_x = cca_results.get_top_x_loadings(10)
    top_pgs = [
        (l.variable_name, l.loading_estimate, l.ci_95_low, l.ci_95_upper)
        for l in top_x
    ]

    # Process Y loadings
    categorized_y = categorize_y_loadings(cca_results.y_loadings, cca_results.freesurfer_labels)

    global_metrics = {}
    regional_by_type = {}

    for metric_type, items in categorized_y.items():
        for loading, parsed in items:
            if parsed['is_global']:
                global_metrics[metric_type] = (
                    loading.loading_estimate,
                    loading.ci_95_low,
                    loading.ci_95_upper
                )
            else:
                if metric_type not in regional_by_type:
                    regional_by_type[metric_type] = []

                regional_by_type[metric_type].append((
                    parsed['region_abbrev'],
                    parsed['region_full_name'],
                    loading.loading_estimate,
                    loading.ci_95_low,
                    loading.ci_95_upper
                ))

    # Sort regional metrics by loading magnitude
    for metric_type in regional_by_type:
        regional_by_type[metric_type].sort(key=lambda x: abs(x[2]), reverse=True)

    # Count significant Y loadings
    sig_y = cca_results.get_significant_y_loadings(ci_level)

    return ResultsSummary(
        significant_positive_pgs=sig_positive_pgs,
        significant_negative_pgs=sig_negative_pgs,
        top_pgs_by_magnitude=top_pgs,
        global_network_metrics=global_metrics,
        regional_metrics_by_type=regional_by_type,
        n_total_x=len(cca_results.x_loadings),
        n_significant_x=len(sig_x),
        n_total_y=len(cca_results.y_loadings),
        n_significant_y=len(sig_y)
    )


def format_pgs_loading_text(
    name: str,
    estimate: float,
    ci_low: float,
    ci_high: float,
    include_full_name: bool = True
) -> str:
    """Format a single PGS loading for text output."""
    if include_full_name:
        full_name = get_trait_full_name(name)
        return f"{full_name} ({name}): loading = {estimate:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])"
    return f"{name}: loading = {estimate:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])"


def generate_results_text_manual(summary: ResultsSummary) -> str:
    """
    Generate Results section text without using LLM (rule-based).
    """
    sections = []

    # Section header
    sections.append("Results\n" + "=" * 50)

    # Overview
    sections.append(f"""
Sparse CCA Results Overview
---------------------------
We identified a statistically significant mode of covariation between polygenic scores (PGS)
and brain network measures (BNMs). Out of {summary.n_total_x} PGS variables, {summary.n_significant_x}
showed significant loadings (95% confidence interval not crossing zero). Similarly, out of
{summary.n_total_y} BNM variables, {summary.n_significant_y} demonstrated significant loadings.
""")

    # PGS with negative loadings (typically cognitive traits)
    if summary.significant_negative_pgs:
        sections.append("\nPolygenic Scores with Significant Negative Loadings")
        sections.append("-" * 50)

        for name, est, ci_low, ci_high in summary.significant_negative_pgs[:10]:
            full_name = get_trait_full_name(name)
            sections.append(f"  - {full_name} ({name}): {est:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")

    # PGS with positive loadings
    if summary.significant_positive_pgs:
        sections.append("\nPolygenic Scores with Significant Positive Loadings")
        sections.append("-" * 50)

        for name, est, ci_low, ci_high in summary.significant_positive_pgs[:10]:
            full_name = get_trait_full_name(name)
            sections.append(f"  - {full_name} ({name}): {est:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")

    # Global network metrics
    sections.append("\nGlobal Brain Network Metrics")
    sections.append("-" * 50)

    for metric, (est, ci_low, ci_high) in summary.global_network_metrics.items():
        sig_marker = "*" if (ci_low > 0 or ci_high < 0) else ""
        sections.append(f"  - {metric}: {est:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}]){sig_marker}")

    # Regional network metrics (top regions per metric)
    sections.append("\nRegional Brain Network Metrics (Top Contributing Regions)")
    sections.append("-" * 50)

    for metric_type, regions in summary.regional_metrics_by_type.items():
        sections.append(f"\n  {metric_type.upper()}:")
        for abbrev, full_name, est, ci_low, ci_high in regions[:5]:  # Top 5
            sig_marker = "*" if (ci_low > 0 or ci_high < 0) else ""
            sections.append(f"    - {full_name} ({abbrev}): {est:.3f}{sig_marker}")

    sections.append("\n* indicates 95% CI does not cross zero")

    return "\n".join(sections)


def generate_results_with_llm(
    cca_results: CCAResults,
    summary: ResultsSummary,
    model_name: str = 'gemini-flash-latest'
) -> str:
    """
    Generate Results section using Gemini LLM.

    Args:
        cca_results: Full CCA results including context
        summary: Extracted results summary
        model_name: Gemini model to use

    Returns:
        Generated Results section text
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found. Using manual generation.")
        return generate_results_text_manual(summary)

    client = genai.Client(api_key=api_key)

    # Prepare context
    intro_summary = cca_results.introduction[:1500] if len(cca_results.introduction) > 1500 else cca_results.introduction
    methods_summary = cca_results.methods[:2000] if len(cca_results.methods) > 2000 else cca_results.methods

    # Format significant PGS findings
    neg_pgs_text = "\n".join([
        f"- {get_trait_full_name(n)} ({n}): loading={e:.3f}, 95% CI=[{l:.3f}, {h:.3f}]"
        for n, e, l, h in summary.significant_negative_pgs[:8]
    ])

    pos_pgs_text = "\n".join([
        f"- {get_trait_full_name(n)} ({n}): loading={e:.3f}, 95% CI=[{l:.3f}, {h:.3f}]"
        for n, e, l, h in summary.significant_positive_pgs[:8]
    ])

    # Format global metrics
    global_metrics_text = "\n".join([
        f"- {m}: loading={e:.3f}, 95% CI=[{l:.3f}, {h:.3f}]"
        for m, (e, l, h) in summary.global_network_metrics.items()
    ])

    # Format top regional findings
    regional_text = ""
    for metric_type, regions in list(summary.regional_metrics_by_type.items())[:3]:
        regional_text += f"\n{metric_type}:\n"
        for abbrev, full_name, est, ci_low, ci_high in regions[:5]:
            regional_text += f"  - {full_name}: loading={est:.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}]\n"

    prompt = f"""You are a senior scientific writer specializing in neuroimaging genetics research, writing for a high-impact journal like Nature Neuroscience or Nature Communications.

Write a comprehensive, detailed Results section based on the sparse Canonical Correlation Analysis (SCCA) findings below. This should be publication-ready for a Nature-style journal.

CONTEXT FROM INTRODUCTION:
{intro_summary}

METHODS SUMMARY:
The study used sparse CCA to identify multivariate associations between {summary.n_total_x} polygenic scores (PGS)
and {summary.n_total_y} brain network measures (BNMs) in the ABCD dataset (N=~11,000 children, ages 9-10 years). 
Bootstrap resampling (5,000 iterations) was used to estimate 95% confidence intervals. Variables are considered 
significant if their 95% CI does not cross zero.

KEY FINDINGS:

1. PGS with Significant NEGATIVE Loadings ({len(summary.significant_negative_pgs)} total):
{neg_pgs_text}

2. PGS with Significant POSITIVE Loadings ({len(summary.significant_positive_pgs)} total):
{pos_pgs_text}

3. Global Brain Network Metrics:
{global_metrics_text}

4. Regional Brain Network Metrics (top regions by loading magnitude):
{regional_text}

SUMMARY STATISTICS:
- Total PGS variables analyzed: {summary.n_total_x}
- Significant PGS variables (95% CI): {summary.n_significant_x} ({summary.n_significant_x}/{summary.n_total_x} = {100*summary.n_significant_x/summary.n_total_x:.1f}%)
- Total BNM variables analyzed: {summary.n_total_y}
- Significant BNM variables (95% CI): {summary.n_significant_y} ({summary.n_significant_y}/{summary.n_total_y} = {100*summary.n_significant_y/summary.n_total_y:.1f}%)

INSTRUCTIONS FOR NATURE-STYLE RESULTS SECTION:

1. STRUCTURE AND LENGTH:
   - Write 8-12 paragraphs organized into clear subsections
   - Target length: 1,500-2,500 words
   - Use descriptive subheadings (e.g., "Identification of a robust PGS-BNM canonical mode", "Cognitive and psychiatric PGS loadings", "Regional brain network architecture")

2. OPENING PARAGRAPH:
   - Provide a comprehensive overview of the main finding
   - State the canonical correlation coefficient and its statistical significance
   - Mention the proportion of variance explained
   - Describe the overall pattern (cognitive vs psychiatric axis)

3. POLYGENIC SCORE LOADINGS (2-3 paragraphs):
   - Create a detailed subsection describing PGS loadings
   - For NEGATIVE loadings (cognitive traits):
     * List ALL significant PGS with exact loading values and 95% CIs
     * Describe the magnitude hierarchy (strongest to weakest)
     * Note the coherence of cognitive/educational traits clustering together
   - For POSITIVE loadings (psychiatric/metabolic traits):
     * List ALL significant PGS with exact loading values and 95% CIs
     * Group by trait category (internalizing, externalizing, metabolic)
     * Highlight the diversity of psychiatric phenotypes involved
   - Include specific numerical details for at least the top 10 contributors in each direction

4. BRAIN NETWORK METRICS (3-4 paragraphs):
   - Create subsections for different metric types (global vs regional, strength vs degree, etc.)
   - For GLOBAL metrics:
     * Report all global network metrics tested, even non-significant ones
     * Explain why global metrics did not reach significance
   - For REGIONAL metrics:
     * Organize by anatomical systems (subcortical, cortical, by lobe)
     * Report the top 15-20 regions with highest loading magnitudes
     * Include both positive and negative contributors
     * Specify exact brain regions with full anatomical names
     * Note any lateralization patterns (left vs right hemisphere)
     * Describe spatial clustering or distributed patterns

5. STATISTICAL RIGOR:
   - Always report: loading estimate, 95% CI lower bound, 95% CI upper bound
   - Use precise language: "significantly different from zero", "95% confidence interval"
   - Report effect sizes and standardized coefficients
   - Mention bootstrap stability if available

6. QUANTITATIVE DETAILS:
   - Include specific numbers for all major findings
   - Use tables/figure references (e.g., "Fig. 1a", "Table 1", "Supplementary Table 2")
   - Provide ranges and distributions, not just point estimates
   - Report percentages and proportions where relevant

7. WRITING STYLE:
   - Use past tense consistently ("was", "showed", "demonstrated")
   - Be objective and descriptive, avoid interpretation
   - Use technical precision (e.g., "bilateral putamen", "fronto-striatal connectivity")
   - Employ Nature's formal but clear writing style
   - Use transition sentences between paragraphs for flow

8. ADDITIONAL ELEMENTS:
   - Mention supplementary materials where appropriate
   - Note any sensitivity analyses or robustness checks
   - Describe the spatial distribution of findings
   - Include any relevant demographic or sample characteristics

9. AVOID:
   - Do NOT include citations or references
   - Do NOT interpret findings (save for Discussion)
   - Do NOT use vague language ("some regions", "several traits")
   - Do NOT omit statistical details

Write a comprehensive, Nature-quality Results section following these guidelines:"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Error generating Results with LLM: {e}")
        return generate_results_text_manual(summary)


def generate_results_section(
    cca_results: Optional[CCAResults] = None,
    use_llm: bool = True,
    base_dir: str = "."
) -> str:
    """
    Main function to generate the Results section.

    Args:
        cca_results: Pre-loaded CCA results (will load if None)
        use_llm: Whether to use LLM for generation
        base_dir: Base directory for data files if loading

    Returns:
        Generated Results section text
    """
    if cca_results is None:
        cca_results = load_default_cca_results(base_dir)

    summary = extract_results_summary(cca_results)

    if use_llm:
        return generate_results_with_llm(cca_results, summary)
    else:
        return generate_results_text_manual(summary)


if __name__ == "__main__":
    print("Generating Results Section...")
    print("=" * 60)

    try:
        # Load data
        cca_results = load_default_cca_results()

        # Generate summary first
        summary = extract_results_summary(cca_results)

        print("\n--- Manual Results Summary ---\n")
        print(generate_results_text_manual(summary))

        print("\n" + "=" * 60)
        print("\n--- LLM-Generated Results ---\n")
        print(generate_results_with_llm(cca_results, summary))

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
