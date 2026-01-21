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
    model_name: str = 'gemini-flash-latest'
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

    prompt = f"""You are a senior scientific writer and neuroscientist specializing in neuroimaging genetics and developmental neuroscience, writing for a premier journal like Nature Neuroscience, Nature Communications, or Nature Human Behaviour.

Write a comprehensive, deeply analytical Discussion section based on the CCA findings and related literature below. This should be publication-ready for a Nature-style journal.

STUDY CONTEXT:
This study used sparse Canonical Correlation Analysis (SCCA) to examine the multivariate relationship between
30 polygenic scores (PGS) for cognitive and psychiatric traits and brain network measures (BNMs) derived from
diffusion MRI tractography in approximately 11,000 children from the ABCD study (ages 9-10 years).

{findings_text}

RELEVANT PRIOR LITERATURE:
{literature_text}

AVAILABLE CITATIONS (integrate these naturally):
{chr(10).join([f'{format_citation(a)}' for a in literature_context.all_references[:20]])}

INSTRUCTIONS FOR NATURE-STYLE DISCUSSION SECTION:

1. STRUCTURE AND LENGTH:
   - Write 10-15 paragraphs organized into clear subsections with descriptive subheadings
   - Target length: 2,500-3,500 words
   - Use a logical flow: main findings → interpretation → mechanisms → literature comparison → implications → limitations → future directions

2. OPENING PARAGRAPH (1-2 paragraphs):
   - Provide a compelling summary of the principal finding
   - State the key novelty and significance of this work
   - Emphasize the multivariate nature and developmental timing (preadolescent)
   - Highlight the bipolar genetic architecture (cognitive vs psychiatric/metabolic)
   - Set up the narrative for the rest of the discussion

3. INTERPRETATION OF THE GENETIC ARCHITECTURE (2-3 paragraphs):
   Subsection: "A shared genetic architecture linking cognitive ability and psychiatric vulnerability"
   - Deeply analyze the bipolar PGS loading pattern
   - Discuss genetic pleiotropy and shared biological pathways
   - Explain what the negative cognitive loadings vs positive psychiatric loadings mean biologically
   - Connect to concepts like the 'p-factor' (general psychopathology factor)
   - Discuss the inclusion of metabolic traits (BMI, smoking) and what this suggests about generalized vulnerability
   - Compare the magnitude and consistency of loadings across trait categories
   - Discuss the implications of finding this pattern in preadolescent children

4. BRAIN NETWORK INTERPRETATION (2-3 paragraphs):
   Subsection: "Structural brain network substrates of polygenic risk"
   - Analyze which brain regions and network properties were most strongly associated
   - Discuss the functional significance of identified regions (e.g., putamen, hippocampus, insula, temporal pole)
   - Explain why these specific regions/networks are biologically plausible given the PGS involved
   - Discuss the absence of significant global network metrics and what this implies
   - Analyze regional vs global findings in terms of developmental neurobiology
   - Consider lateralization patterns and anatomical specificity
   - Relate structural connectivity findings to known functional networks (default mode, salience, executive control)

5. MECHANISTIC INTERPRETATION (2-3 paragraphs):
   Subsection: "Biological mechanisms linking genetic risk to brain network organization"
   - Propose specific molecular and cellular mechanisms
   - Discuss neurodevelopmental processes (myelination, synaptic pruning, axonal guidance, neurogenesis)
   - Connect to known biological pathways (e.g., synaptic signaling, neurotransmitter systems, inflammatory processes)
   - Discuss the developmental timing - why these patterns are visible at ages 9-10
   - Consider gene expression patterns in identified brain regions
   - Discuss how genetic variants might influence white matter microstructure
   - Propose testable hypotheses about causal pathways

6. COMPARISON WITH PRIOR LITERATURE (2-3 paragraphs):
   Subsection: "Convergence with prior neuroimaging-genetic studies"
   - Extensively cite and compare with the provided literature
   - Highlight consistencies with previous findings
   - Discuss discrepancies and potential reasons
   - Compare with adult studies vs other pediatric studies
   - Discuss how multivariate approaches (CCA) provide advantages over univariate approaches
   - Integrate findings from GWAS, imaging genetics, and developmental neuroscience
   - Use at least 10-15 citations naturally integrated throughout

7. CLINICAL AND TRANSLATIONAL IMPLICATIONS (2 paragraphs):
   Subsection: "Implications for early risk identification and intervention"
   - Discuss potential for early biomarkers of psychiatric risk
   - Consider precision psychiatry and personalized medicine applications
   - Discuss preventive interventions during sensitive developmental periods
   - Address the ethical considerations of polygenic risk prediction in children
   - Discuss potential for monitoring intervention efficacy
   - Consider public health implications

8. LIMITATIONS (1-2 paragraphs):
   Subsection: "Limitations and caveats"
   - Provide a thorough, honest discussion of limitations
   - Cross-sectional design and inability to infer causation
   - PGS derived from adult GWAS applied to children
   - Population ancestry considerations and generalizability
   - Measurement limitations of diffusion MRI and network construction
   - Statistical considerations (multiple testing, model selection)
   - Unmeasured confounders and environmental factors
   - Limited functional outcome data at this age
   - Each limitation should be substantive (2-3 sentences)

9. FUTURE DIRECTIONS (1 paragraph):
   Subsection: "Future research directions"
   - Propose specific, concrete future studies
   - Longitudinal follow-up to track developmental trajectories
   - Integration with functional MRI and other modalities
   - Gene-environment interaction studies
   - Validation in independent cohorts and diverse populations
   - Mechanistic studies (animal models, in vitro)
   - Clinical translation and intervention studies

10. WRITING STYLE:
    - Use sophisticated, precise scientific language
    - Balance technical depth with clarity
    - Employ cautious interpretation ("suggests", "may indicate", "is consistent with", "appears to reflect")
    - Use active voice where appropriate for impact
    - Create smooth transitions between paragraphs and sections
    - Build a compelling narrative arc
    - Maintain objectivity while highlighting significance

11. CITATION INTEGRATION:
    - Use author-year format: (Author et al., Year) or "Author et al. (Year) showed..."
    - Integrate citations naturally into the narrative, not as lists
    - Cite multiple papers when discussing a concept
    - Use citations to support claims, provide context, and show convergence/divergence
    - Only cite papers from the provided literature list

12. QUANTITATIVE DEPTH:
    - Reference specific loading values and effect sizes from the results
    - Discuss the magnitude and precision of estimates
    - Compare effect sizes across different PGS and brain regions
    - Relate findings to clinically meaningful outcomes where possible

13. AVOID:
    - Do NOT simply restate results
    - Do NOT make unsupported causal claims
    - Do NOT ignore contradictory evidence
    - Do NOT use vague, hand-waving explanations
    - Do NOT neglect important limitations

Write a comprehensive, Nature-quality Discussion section following these guidelines:"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        discussion_text = response.text

        return discussion_text, ref_list

    except Exception as e:
        raise RuntimeError(f"Error generating Discussion with LLM: {e}")


def generate_discussion_section(
    cca_results: Optional[CCAResults] = None,
    summary: Optional[ResultsSummary] = None,
    gather_literature: bool = True,
    verbose: bool = True,
    base_dir: str = "."
) -> Tuple[str, List[str]]:
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

    discussion, references = generate_discussion_with_llm(
        cca_results, summary, literature_context
    )

    return discussion, references


def save_discussion_output(
    discussion: str,
    references: List[str],
    output_dir: str = "results",
    filename_prefix: str = "generated_discussion"
):
    """
    Save generated discussion and references to files.

    Args:
        discussion: Generated discussion text
        references: List of formatted references
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

    print(f"Discussion saved to: {discussion_path}")
    print(f"References saved to: {references_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Discussion Section Generator")
    print("=" * 60)

    try:
        # Generate discussion
        discussion, references = generate_discussion_section(
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
        # save_discussion_output(discussion, references)

    except FileNotFoundError as e:
        print(f"Error: Data file not found - {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
