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

import google.generativeai as genai
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


def format_literature_for_prompt(context: LiteratureContext, max_chars: Optional[int] = None, max_abstract_length: Optional[int] = None) -> str:
    """
    Format gathered literature into a prompt-friendly string.

    Args:
        context: LiteratureContext with articles
        max_chars: Maximum characters to include (None means no limit)
        max_abstract_length: Maximum characters per abstract (None means no limit)

    Returns:
        Formatted string of literature summaries
    """
    sections = []

    # PGS-brain associations
    if context.pgs_brain_articles:
        sections.append("## PGS-Brain Association Literature")
        for trait, articles in context.pgs_brain_articles.items():
            sections.append(f"\n### {trait}")
            sections.append(format_articles_for_context(articles, max_abstract_length=max_abstract_length))

    # Brain region literature
    if context.brain_region_articles:
        sections.append("\n## Brain Region Function Literature")
        for region, articles in list(context.brain_region_articles.items())[:3]:
            sections.append(f"\n### {region}")
            sections.append(format_articles_for_context(articles[:2], max_abstract_length=max_abstract_length))

    # Network metric literature
    if context.network_metric_articles:
        sections.append("\n## Network Metric Literature")
        for metric, articles in list(context.network_metric_articles.items())[:2]:
            sections.append(f"\n### {metric}")
            sections.append(format_articles_for_context(articles[:2], max_abstract_length=max_abstract_length))

    # General literature
    if context.general_topic_articles:
        sections.append("\n## General Topic Literature")
        sections.append(format_articles_for_context(context.general_topic_articles[:3], max_abstract_length=max_abstract_length))

    full_text = "\n".join(sections)

    # Truncate if too long (explicit None check allows 0 to be a valid value)
    if max_chars is not None and len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[Literature truncated due to length...]"

    return full_text


def generate_discussion_with_llm(
    cca_results: CCAResults,
    summary: ResultsSummary,
    literature_context: LiteratureContext,
    results_text: str = "",
    model_name: str = 'gemini-2.5-flash'
) -> Tuple[str, List[str], str]:
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

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

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

    prompt = f"""You are a senior scientific writer and neuroscientist specializing in neuroimaging genetics and developmental neuroscience.

Write an analytical Discussion section based on the CCA findings and related literature below.

STUDY CONTEXT:
This study used sparse Canonical Correlation Analysis (SCCA) to examine the multivariate relationship between
{summary.n_total_x} polygenic scores (PGS) for cognitive and psychiatric traits and {summary.n_total_y} brain network measures (BNMs)
derived from diffusion MRI tractography in a pediatric cohort.

{findings_text}

RELEVANT PRIOR LITERATURE:
{literature_text}

AVAILABLE CITATIONS (integrate these extensively throughout):
{chr(10).join([f'{format_citation(a)}' for a in literature_context.all_references[:20]])}

INSTRUCTIONS FOR NATURE-STYLE DISCUSSION SECTION:

1. STRUCTURE AND LENGTH:
   - Write 12-18 paragraphs with MINIMAL subsection headings (maximum 3-4 subsections total)
   - Target length: 3,000-4,000 words
   - Use flowing narrative structure rather than rigid subsections
   - Let ideas transition naturally between paragraphs
   - Focus on depth of analysis rather than organizational structure

2. SUBSECTION STRUCTURE (Use only 3-4 major subsections):
   - Opening (no heading): 2-3 paragraphs introducing main findings and significance
   - "Genetic architecture and brain network substrates": 5-7 paragraphs deeply integrating genetic and neural findings
   - "Implications and future directions": 3-4 paragraphs on clinical relevance, limitations, and future work
   - Closing (no heading): 1-2 paragraphs synthesizing key insights

3. OPENING PARAGRAPHS (2-3 paragraphs, no subsection heading):
   - Begin with a compelling statement of the principal finding
   - Emphasize the novelty: multivariate approach, developmental timing (preadolescent), large sample
   - Introduce the bipolar genetic architecture (cognitive vs psychiatric/metabolic traits)
   - Set up the key questions that the discussion will address
   - Establish why these findings matter for understanding brain development and psychiatric risk

4. MAIN DISCUSSION - "Genetic architecture and brain network substrates" (5-7 paragraphs):
   
   CRITICAL: This is the core of the discussion. Integrate the following themes fluidly across paragraphs:
   
   a) Genetic pleiotropy and shared pathways:
      - Deeply analyze what the bipolar PGS pattern reveals about genetic architecture
      - Connect to the 'p-factor' concept and transdiagnostic psychiatry
      - Discuss genetic correlations between cognitive and psychiatric traits
      - Extensively cite prior GWAS and genetic correlation studies
      - Explain biological mechanisms of pleiotropy (shared molecular pathways, developmental timing)
   
   b) Brain network substrates and their functional significance:
      - Integrate discussion of specific brain regions (putamen, hippocampus, insula, temporal pole)
      - Explain WHY these regions are biologically plausible given the genetic findings
      - Connect structural findings to functional networks and cognitive processes
      - Discuss the absence of global metrics and what this reveals about network organization
      - Cite neuroimaging studies that support or contextualize these findings
   
   c) Developmental neurobiology and mechanistic insights:
      - Discuss neurodevelopmental processes: myelination, synaptic pruning, circuit formation
      - Propose specific molecular mechanisms (neurotransmitter systems, synaptic proteins, inflammatory pathways)
      - Explain why these patterns are visible at ages 9-10 (critical developmental period)
      - Connect to gene expression atlases and developmental transcriptomics
      - Discuss how genetic variants influence white matter microstructure
      - Cite developmental neuroscience and molecular studies extensively
   
   d) Integration with prior neuroimaging-genetic literature:
      - Compare findings with previous imaging genetics studies (both univariate and multivariate)
      - Highlight convergence with prior work and explain discrepancies
      - Discuss advantages of CCA over traditional approaches
      - Compare pediatric vs adult findings
      - Synthesize across GWAS, imaging genetics, and developmental neuroscience literatures
      - Use 15-20+ citations naturally woven throughout this section
   
   WRITING APPROACH FOR THIS SECTION:
   - Each paragraph should integrate multiple themes (genetics + brain + mechanisms + literature)
   - Avoid separating topics into isolated paragraphs
   - Build arguments progressively, with each paragraph deepening the analysis
   - Use extensive citations (aim for 2-4 citations per paragraph)
   - Connect findings to broader theoretical frameworks
   - Propose testable hypotheses and mechanistic models

5. IMPLICATIONS AND FUTURE DIRECTIONS (3-4 paragraphs):
   
   a) Clinical and translational implications (1-2 paragraphs):
      - Discuss potential for early biomarkers and risk stratification
      - Address precision psychiatry and personalized intervention
      - Consider preventive strategies during sensitive developmental periods
      - Discuss ethical considerations of polygenic prediction in children
      - Cite relevant clinical and translational studies
   
   b) Limitations (1 paragraph):
      - Provide honest, substantive discussion of key limitations
      - Cross-sectional design, PGS from adult GWAS, population specificity
      - Measurement limitations, unmeasured confounders
      - Keep concise but thorough (3-5 sentences per limitation)
   
   c) Future research directions (1 paragraph):
      - Propose specific, concrete next steps
      - Longitudinal follow-up, multi-modal integration, mechanistic studies
      - Validation in diverse populations, gene-environment interactions
      - Clinical translation and intervention trials

6. CLOSING PARAGRAPHS (1-2 paragraphs, no subsection heading):
   - Synthesize the key insights from the discussion
   - Return to the broader significance for understanding brain development
   - End with a forward-looking statement about implications for psychiatry and neuroscience

7. CITATION STRATEGY (CRITICAL):
   - Aim for 20-30+ total citations throughout the discussion
   - Integrate citations naturally: "Previous work has shown... (Author et al., Year)"
   - Cluster related citations: "...consistent with multiple studies (Author1 et al., Year1; Author2 et al., Year2)"
   - Use citations to build arguments, not just support isolated facts
   - Cite both supporting and contrasting evidence
   - Only use papers from the provided literature list
   - Author-year format: (Author et al., Year) or "Author et al. (Year) demonstrated..."

8. SCIENTIFIC DEPTH AND INSIGHT:
   - Go beyond describing findings to explaining WHY and HOW
   - Propose mechanistic models and testable hypotheses
   - Connect molecular, cellular, circuit, and behavioral levels of analysis
   - Discuss evolutionary and developmental perspectives where relevant
   - Address methodological innovations and their implications
   - Consider alternative interpretations and address them
   - Relate findings to broader theoretical frameworks in neuroscience and psychiatry

9. WRITING STYLE:
   - Sophisticated, precise scientific language
   - Balance technical depth with accessibility
   - Use cautious interpretation: "suggests", "may indicate", "is consistent with", "appears to"
   - Active voice for impact when appropriate
   - Smooth transitions between ideas and paragraphs
   - Build a compelling narrative arc
   - Maintain objectivity while conveying significance
   - Vary sentence structure for readability

10. CONTEXT FROM RESULTS SECTION:
{results_text}

CONTEXT FROM INTRODUCTION:
    - Reference specific loading values and effect sizes
    - Discuss magnitude and precision of estimates
    - Compare effect sizes across PGS and brain regions
    - Relate statistical findings to biological and clinical significance
    - Use quantitative comparisons with prior literature where possible

11. CRITICAL REQUIREMENTS:
    - MINIMIZE subsection headings (only 3-4 total)
    - MAXIMIZE literature integration (20-30+ citations)
    - EMPHASIZE mechanistic insights and biological interpretation
    - CONNECT findings across multiple levels of analysis
    - PROPOSE testable hypotheses and future directions
    - MAINTAIN narrative flow rather than checklist structure

=== CRITICAL INSTRUCTIONS TO PREVENT HALLUCINATION ===

**ABSOLUTE PROHIBITIONS - DO NOT INCLUDE:**
1. DO NOT reference any figures (Fig. 1, Figure 2, etc.)
2. DO NOT reference any tables (Table 1, Supplementary Table, etc.)
3. DO NOT reference any supplementary materials (Supplementary Fig., Extended Data, etc.)
4. DO NOT invent or estimate canonical correlation coefficients (e.g., "r = 0.67", "Rc = 0.45")
5. DO NOT invent or estimate p-values (e.g., "p < 0.001", "p = 0.02")
6. DO NOT invent or estimate exact sample sizes (e.g., "N = 11,000", "n = 9,500")
7. DO NOT invent or estimate variance explained (e.g., "explained 15% of variance")
8. DO NOT invent effect sizes, odds ratios, or statistics not provided in the data
9. DO NOT mention specific bootstrap iteration counts unless explicitly provided
10. DO NOT cite papers that are not in the provided literature list

**USE ONLY THE DATA PROVIDED:**
- Only use loading values and statistics explicitly listed in the KEY FINDINGS section
- Only cite papers from the AVAILABLE CITATIONS list provided above
- If specific statistics are not provided, describe findings qualitatively instead

12. AVOID:
    - Excessive subsection headings (no more than 3-4 total)
    - Simply restating results without interpretation
    - Unsupported causal claims
    - Vague, hand-waving explanations
    - Ignoring contradictory evidence or alternative interpretations
    - Superficial treatment of complex topics
    - Isolated discussion of topics without integration
    - ANY references to figures or tables
    - Inventing statistics or numerical values not provided

Write a Discussion section that integrates prior literature to provide scientific depth and mechanistic understanding, using ONLY the data and citations provided:"""

    try:
        response = model.generate_content(prompt)
        discussion_text = response.text

        return discussion_text, ref_list, literature_text

    except Exception as e:
        raise RuntimeError(f"Error generating Discussion with LLM: {e}")


def generate_discussion_section(
    cca_results: Optional[CCAResults] = None,
    summary: Optional[ResultsSummary] = None,
    results_text: str = "",
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
        # Use our enhanced discussion generator
        # Pass literature context if we searched, otherwise it will be empty
        discussion, references, literature_text = generate_discussion_with_llm(
            cca_results, summary, literature_context, results_text=results_text, model_name='gemini-2.5-flash'
        )

    return discussion, references, literature_text, literature_context


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
        discussion, references, literature_text, literature_context = generate_discussion_section(
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
