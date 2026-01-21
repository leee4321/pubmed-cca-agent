"""
Discussion Section Generator for CCA Analysis.

This module generates the Discussion section of a scientific paper by:
1. Analyzing CCA results to identify key findings
2. Searching PubMed for relevant prior literature
3. Using LLM to synthesize findings with prior research
"""

import os
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from google import genai
from dotenv import load_dotenv

from data_loader import (
    CCAResults,
    CCALoadingResult,
    load_default_cca_results,
    parse_y_variable_name,
    categorize_y_loadings
)
from pubmed_tool import (
    PubMedArticle,
    search_and_fetch_detailed,
    search_for_pgs_brain_association,
    search_for_brain_region_function,
    search_for_network_property,
    format_citation,
    format_reference,
    format_articles_for_context,
    get_trait_full_name,
    build_cca_related_query
)
from results_generator import extract_results_summary, ResultsSummary

load_dotenv()


@dataclass
class LiteratureContext:
    """Container for literature gathered from PubMed."""
    pgs_brain_articles: Dict[str, List[PubMedArticle]] = field(default_factory=dict)
    brain_region_articles: Dict[str, List[PubMedArticle]] = field(default_factory=dict)
    network_metric_articles: Dict[str, List[PubMedArticle]] = field(default_factory=dict)
    general_topic_articles: List[PubMedArticle] = field(default_factory=list)
    all_references: List[PubMedArticle] = field(default_factory=list)


def gather_literature_for_discussion(
    summary: ResultsSummary,
    cca_results: CCAResults,
    max_articles_per_topic: int = 3,
    verbose: bool = True
) -> LiteratureContext:
    """
    Gather relevant literature from PubMed based on CCA findings.

    Args:
        summary: ResultsSummary from results_generator
        cca_results: Full CCA results
        max_articles_per_topic: Maximum articles to fetch per topic
        verbose: Whether to print progress

    Returns:
        LiteratureContext with categorized articles
    """
    context = LiteratureContext()

    # 1. Search for PGS-brain associations (top significant traits)
    if verbose:
        print("\n1. Searching for PGS-brain association literature...")

    # Combine positive and negative significant PGS
    all_sig_pgs = summary.significant_negative_pgs + summary.significant_positive_pgs

    # Focus on top 5 most significant (by magnitude)
    top_pgs = sorted(all_sig_pgs, key=lambda x: abs(x[1]), reverse=True)[:5]

    for pgs_code, estimate, _, _ in top_pgs:
        trait_name = get_trait_full_name(pgs_code)
        if verbose:
            print(f"  - Searching for: {trait_name}")

        articles = search_for_pgs_brain_association(trait_name, max_results=max_articles_per_topic)
        if articles:
            context.pgs_brain_articles[trait_name] = articles
            context.all_references.extend(articles)
        time.sleep(0.5)  # Rate limiting

    # 2. Search for brain region function literature (top loading regions)
    if verbose:
        print("\n2. Searching for brain region function literature...")

    # Get top regions from Y loadings
    top_y_loadings = sorted(cca_results.y_loadings,
                           key=lambda x: abs(x.loading_estimate), reverse=True)[:10]

    searched_regions = set()
    for loading in top_y_loadings:
        parsed = parse_y_variable_name(loading.variable_name, cca_results.freesurfer_labels)
        if not parsed['is_global'] and parsed['region_full_name']:
            region_name = parsed['region_full_name']

            # Avoid duplicate searches
            if region_name in searched_regions:
                continue
            searched_regions.add(region_name)

            if verbose:
                print(f"  - Searching for: {region_name}")

            articles = search_for_brain_region_function(region_name, max_results=max_articles_per_topic)
            if articles:
                context.brain_region_articles[region_name] = articles
                context.all_references.extend(articles)
            time.sleep(0.5)

            if len(searched_regions) >= 5:  # Limit to 5 regions
                break

    # 3. Search for network metric literature
    if verbose:
        print("\n3. Searching for network metric literature...")

    key_metrics = ['global efficiency', 'modularity', 'small-worldness', 'clustering coefficient']
    for metric in key_metrics:
        if verbose:
            print(f"  - Searching for: {metric}")

        articles = search_for_network_property(metric, max_results=max_articles_per_topic)
        if articles:
            context.network_metric_articles[metric] = articles
            context.all_references.extend(articles)
        time.sleep(0.5)

    # 4. General topic searches
    if verbose:
        print("\n4. Searching for general topic literature...")

    general_queries = [
        "polygenic score brain development adolescent",
        "brain network topology genetic heritability",
        "cognitive ability structural connectivity white matter",
        "psychiatric risk brain network"
    ]

    for query in general_queries:
        if verbose:
            print(f"  - Searching for: {query}")

        articles = search_and_fetch_detailed(query, max_results=2)
        if articles:
            context.general_topic_articles.extend(articles)
            context.all_references.extend(articles)
        time.sleep(0.5)

    # Remove duplicates from all_references (by PMID)
    seen_pmids = set()
    unique_refs = []
    for article in context.all_references:
        if article.pmid not in seen_pmids:
            seen_pmids.add(article.pmid)
            unique_refs.append(article)
    context.all_references = unique_refs

    if verbose:
        print(f"\nTotal unique references gathered: {len(context.all_references)}")

    return context


def format_literature_for_prompt(context: LiteratureContext, max_chars: int = 8000) -> str:
    """
    Format gathered literature into a prompt-friendly string.

    Args:
        context: LiteratureContext with articles
        max_chars: Maximum characters to include

    Returns:
        Formatted string of literature summaries
    """
    sections = []

    # PGS-brain associations
    if context.pgs_brain_articles:
        sections.append("## PGS-Brain Association Literature")
        for trait, articles in context.pgs_brain_articles.items():
            sections.append(f"\n### {trait}")
            sections.append(format_articles_for_context(articles, max_abstract_length=300))

    # Brain region literature
    if context.brain_region_articles:
        sections.append("\n## Brain Region Function Literature")
        for region, articles in list(context.brain_region_articles.items())[:3]:
            sections.append(f"\n### {region}")
            sections.append(format_articles_for_context(articles[:2], max_abstract_length=300))

    # Network metric literature
    if context.network_metric_articles:
        sections.append("\n## Network Metric Literature")
        for metric, articles in list(context.network_metric_articles.items())[:2]:
            sections.append(f"\n### {metric}")
            sections.append(format_articles_for_context(articles[:2], max_abstract_length=300))

    # General literature
    if context.general_topic_articles:
        sections.append("\n## General Topic Literature")
        sections.append(format_articles_for_context(context.general_topic_articles[:3], max_abstract_length=300))

    full_text = "\n".join(sections)

    # Truncate if too long
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[Literature truncated due to length...]"

    return full_text


def generate_discussion_with_llm(
    cca_results: CCAResults,
    summary: ResultsSummary,
    literature_context: LiteratureContext,
    model_name: str = 'gemini-1.5-flash'
) -> Tuple[str, List[str]]:
    """
    Generate Discussion section using Gemini LLM.

    Args:
        cca_results: Full CCA results
        summary: Results summary
        literature_context: Gathered literature
        model_name: Gemini model to use

    Returns:
        Tuple of (discussion_text, list_of_references)
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    client = genai.Client(api_key=api_key)

    # Format key findings
    top_neg_pgs = [(get_trait_full_name(n), e) for n, e, _, _ in summary.significant_negative_pgs[:5]]
    top_pos_pgs = [(get_trait_full_name(n), e) for n, e, _, _ in summary.significant_positive_pgs[:5]]

    findings_text = f"""
KEY FINDINGS FROM CCA ANALYSIS:

1. Polygenic Scores with Strong NEGATIVE Loadings (associated with one end of brain network pattern):
{chr(10).join([f'   - {name}: loading={est:.3f}' for name, est in top_neg_pgs])}

2. Polygenic Scores with Strong POSITIVE Loadings (associated with opposite brain network pattern):
{chr(10).join([f'   - {name}: loading={est:.3f}' for name, est in top_pos_pgs])}

3. Brain Network Pattern:
   - The mode captures a pattern where cognitive/educational traits (IQ, educational attainment, cognitive performance)
     show negative loadings, while psychiatric/behavioral traits (BMI, smoking, depression) show positive loadings.
   - This suggests a shared genetic architecture linking cognitive ability with specific brain network configurations.
"""

    # Format literature
    literature_text = format_literature_for_prompt(literature_context)

    # Build reference list for citation
    ref_list = []
    ref_map = {}
    for i, article in enumerate(literature_context.all_references, 1):
        citation = format_citation(article)
        ref_map[citation] = i
        ref_list.append(f"[{i}] {format_reference(article)}")

    prompt = f"""You are a scientific writer specializing in neuroimaging genetics and brain development research.
Write a comprehensive Discussion section for a scientific paper based on the CCA findings and related literature below.

STUDY CONTEXT:
This study used sparse Canonical Correlation Analysis (SCCA) to examine the multivariate relationship between
30 polygenic scores (PGS) for cognitive and psychiatric traits and brain network measures (BNMs) derived from
diffusion MRI tractography in children from the ABCD study (ages 9-10 years).

{findings_text}

RELEVANT PRIOR LITERATURE:
{literature_text}

AVAILABLE CITATIONS (use these in your discussion):
{chr(10).join([f'{format_citation(a)}' for a in literature_context.all_references[:20]])}

INSTRUCTIONS FOR WRITING THE DISCUSSION:

1. STRUCTURE (5-7 paragraphs):
   a) Opening paragraph: Summarize the main finding - the identified mode of covariation between PGS and brain networks
   b) Interpretation of PGS loadings: Discuss what the loading pattern (negative for cognitive traits, positive for
      psychiatric traits) suggests about the genetic architecture of brain network organization
   c) Brain network interpretation: Discuss which brain regions and network properties were most associated
   d) Comparison with prior literature: Compare findings to previous studies (use the provided citations)
   e) Biological/mechanistic interpretation: Propose potential mechanisms linking genetic predisposition to brain networks
   f) Implications: Discuss implications for understanding brain development and psychiatric risk
   g) Limitations and future directions

2. CITATION STYLE:
   - Use author-year format: (Author et al., Year) or Author et al. (Year)
   - Only cite papers from the provided literature list
   - Integrate citations naturally into the text

3. TONE:
   - Academic and formal
   - Cautious interpretation (use "may", "suggests", "appears to")
   - Acknowledge limitations

4. LENGTH: Approximately 800-1200 words

Write the Discussion section:"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        discussion_text = response.text

        return discussion_text, ref_list, literature_text

    except Exception as e:
        raise RuntimeError(f"Error generating Discussion with LLM: {e}")


def generate_discussion_section(
    cca_results: Optional[CCAResults] = None,
    summary: Optional[ResultsSummary] = None,
    gather_literature: bool = True,
    verbose: bool = True,
    base_dir: str = "."
) -> Tuple[str, List[str], str]:
    """
    Main function to generate the Discussion section.

    Args:
        cca_results: Pre-loaded CCA results (will load if None)
        summary: Pre-computed results summary (will compute if None)
        gather_literature: Whether to search PubMed for literature
        verbose: Whether to print progress
        base_dir: Base directory for data files

    Returns:
        Tuple of (discussion_text, list_of_references)
    """
    if cca_results is None:
        if verbose:
            print("Loading CCA results...")
        cca_results = load_default_cca_results(base_dir)

    if summary is None:
        if verbose:
            print("Extracting results summary...")
        summary = extract_results_summary(cca_results)

    if gather_literature:
        if verbose:
            print("\nGathering literature from PubMed...")
        literature_context = gather_literature_for_discussion(
            summary, cca_results,
            max_articles_per_topic=3,
            verbose=verbose
        )
    else:
        literature_context = LiteratureContext()

    if verbose:
        print("\nGenerating Discussion section with LLM...")

    discussion, references, literature_text = generate_discussion_with_llm(
        cca_results, summary, literature_context
    )

    return discussion, references, literature_text


def save_discussion_output(
    discussion: str,
    references: List[str],
    literature_text: str,
    output_dir: str = "results",
    filename_prefix: str = "generated_discussion"
):
    """
    Save generated discussion and references to files.

    Args:
        discussion: Generated discussion text
        references: List of formatted references
        literature_text: Text listing literature gathered using keyword search on PubMed. This text is included in the prompt for LLM to generate discussions.
        output_dir: Directory to save files
        filename_prefix: Prefix for output filenames
    """
    import os

    # Save discussion
    discussion_path = os.path.join(output_dir, f"{filename_prefix}.txt")
    with open(discussion_path, 'w', encoding='utf-8') as f:
        f.write("DISCUSSION\n")
        f.write("=" * 60 + "\n\n")
        f.write(discussion)

    # Save references
    references_path = os.path.join(output_dir, f"{filename_prefix}_references.txt")
    with open(references_path, 'w', encoding='utf-8') as f:
        f.write("REFERENCES\n")
        f.write("=" * 60 + "\n\n")
        for ref in references:
            f.write(ref + "\n\n")
            
    # Save literature context
    literature_path = os.path.join(output_dir, f"{filename_prefix}_literature_text.txt")
    with open(literature_path, 'w', encoding='utf-8') as f:
        f.write("LITERATURE TEXT\n")
        f.write("=" * 60 + "\n\n")
        f.write(literature_text)
    

    print(f"Discussion saved to: {discussion_path}")
    print(f"References saved to: {references_path}")
    print(f"Literature text saved to: {literature_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Discussion Section Generator")
    print("=" * 60)

    try:
        # Generate discussion
        discussion, references, literature_text = generate_discussion_section(
            gather_literature=True,
            verbose=True
        )

        print("\n" + "=" * 60)
        print("GENERATED DISCUSSION")
        print("=" * 60 + "\n")
        print(discussion)

        print("\n" + "=" * 60)
        print("REFERENCES")
        print("=" * 60 + "\n")
        for ref in references[:10]:
            print(ref)
            print()

        if len(references) > 10:
            print(f"... and {len(references) - 10} more references")

        # Optionally save to files
        # save_discussion_output(discussion, references, literature_text)

    except FileNotFoundError as e:
        print(f"Error: Data file not found - {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
