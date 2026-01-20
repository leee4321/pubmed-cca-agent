"""
PubMed API Tool for searching and fetching scientific literature.
Uses NCBI E-utilities to search PubMed and retrieve abstracts.
Enhanced version with support for brain network and polygenic score research.
"""

import requests
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Define base URLs for E-utilities
BASE_URL_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BASE_URL_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Rate limiting
REQUEST_DELAY = 0.35  # seconds between requests


@dataclass
class PubMedArticle:
    """Represents a PubMed article with metadata."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: str
    doi: Optional[str] = None


def search_pubmed(query: str, max_results: int = 5) -> List[str]:
    """
    Search PubMed for a given query and return a list of PMIDs.
    """
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'retmode': 'json',
        'sort': 'relevance'
    }
    try:
        response = requests.get(BASE_URL_SEARCH, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        pmids = data.get('esearchresult', {}).get('idlist', [])
        time.sleep(REQUEST_DELAY)
        return pmids
    except Exception as e:
        print(f"Error searching PubMed for '{query}': {e}")
        return []


def fetch_details(pmid_list: List[str]) -> List[Dict]:
    """
    Fetch abstracts for a list of PMIDs (basic version).
    """
    if not pmid_list:
        return []

    ids = ','.join(pmid_list)
    params = {
        'db': 'pubmed',
        'id': ids,
        'retmode': 'xml'
    }

    try:
        response = requests.get(BASE_URL_FETCH, params=params, timeout=60)
        response.raise_for_status()

        # Parse XML to get abstracts
        root = ET.fromstring(response.content)
        articles = []

        for article in root.findall(".//PubmedArticle"):
            title = article.find(".//ArticleTitle")
            title_text = title.text if title is not None else "No Title"

            abstract_text = ""
            abstract = article.find(".//Abstract")
            if abstract is not None:
                for text_node in abstract.findall("AbstractText"):
                    if text_node.text:
                        abstract_text += text_node.text + " "

            articles.append({
                'title': title_text,
                'abstract': abstract_text.strip()
            })

        time.sleep(REQUEST_DELAY)
        return articles

    except Exception as e:
        print(f"Error fetching details: {e}")
        return []


def fetch_detailed_articles(pmid_list: List[str]) -> List[PubMedArticle]:
    """
    Fetch full article details including authors, journal, year, and DOI.
    """
    if not pmid_list:
        return []

    ids = ','.join(pmid_list)
    params = {
        'db': 'pubmed',
        'id': ids,
        'retmode': 'xml'
    }

    try:
        response = requests.get(BASE_URL_FETCH, params=params, timeout=60)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        articles = []

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                # Extract PMID
                pmid_elem = article_elem.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""

                # Extract title
                title_elem = article_elem.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else ""

                # Extract abstract
                abstract_parts = []
                for abstract_elem in article_elem.findall(".//AbstractText"):
                    label = abstract_elem.get("Label", "")
                    text = abstract_elem.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)

                # Extract authors
                authors = []
                for author_elem in article_elem.findall(".//Author"):
                    lastname = author_elem.find("LastName")
                    forename = author_elem.find("ForeName")
                    if lastname is not None:
                        name = lastname.text
                        if forename is not None:
                            name = f"{forename.text} {name}"
                        authors.append(name)

                # Extract journal
                journal_elem = article_elem.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""

                # Extract year
                year_elem = article_elem.find(".//PubDate/Year")
                if year_elem is None:
                    year_elem = article_elem.find(".//PubDate/MedlineDate")
                year = year_elem.text[:4] if year_elem is not None and year_elem.text else ""

                # Extract DOI
                doi = None
                for id_elem in article_elem.findall(".//ArticleId"):
                    if id_elem.get("IdType") == "doi":
                        doi = id_elem.text
                        break

                articles.append(PubMedArticle(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    journal=journal,
                    year=year,
                    doi=doi
                ))
            except Exception as e:
                print(f"Error parsing article: {e}")
                continue

        time.sleep(REQUEST_DELAY)
        return articles

    except Exception as e:
        print(f"Error fetching detailed articles: {e}")
        return []


def pubmed_search_and_get_abstracts(query: str, max_results: int = 3) -> str:
    """
    Combined function to search and get abstracts.
    This is the main tool function for the agent.
    """
    print(f"Searching PubMed for: {query}")
    pmids = search_pubmed(query, max_results)
    if not pmids:
        return "No results found."

    articles = fetch_details(pmids)
    results = []
    for i, art in enumerate(articles):
        results.append(f"Title: {art['title']}\nAbstract: {art['abstract']}\n")

    return "\n---\n".join(results)


def search_and_fetch_detailed(query: str, max_results: int = 5) -> List[PubMedArticle]:
    """
    Search PubMed and return detailed article information.
    """
    pmids = search_pubmed(query, max_results)
    return fetch_detailed_articles(pmids)


# ============================================================
# Specialized query builders for CCA research
# ============================================================

def build_brain_region_query(region_name: str, additional_terms: List[str] = None) -> str:
    """
    Build a PubMed query for a specific brain region.

    Args:
        region_name: Name of the brain region (e.g., 'superior frontal gyrus')
        additional_terms: Additional search terms to include
    """
    base_query = f'"{region_name}"[Title/Abstract]'
    context = '(brain[Title/Abstract] OR neuroimaging[Title/Abstract] OR MRI[Title/Abstract] OR connectivity[Title/Abstract])'

    parts = [base_query, context]

    if additional_terms:
        for term in additional_terms:
            parts.append(f'"{term}"[Title/Abstract]')

    return " AND ".join(parts)


def build_pgs_trait_query(trait_name: str, include_brain: bool = True) -> str:
    """
    Build a PubMed query for a polygenic score trait.

    Args:
        trait_name: Name of the trait (e.g., 'cognitive performance', 'schizophrenia')
        include_brain: Whether to include brain-related terms
    """
    base_query = f'"{trait_name}"[Title/Abstract]'
    genetic_context = '(genetic[Title/Abstract] OR polygenic[Title/Abstract] OR GWAS[Title/Abstract] OR heritability[Title/Abstract])'

    parts = [base_query, genetic_context]

    if include_brain:
        brain_context = '(brain[Title/Abstract] OR neural[Title/Abstract] OR cortical[Title/Abstract] OR white matter[Title/Abstract])'
        parts.append(brain_context)

    return " AND ".join(parts)


def build_network_metric_query(metric_name: str) -> str:
    """
    Build a PubMed query for a brain network metric.

    Args:
        metric_name: Name of the network metric (e.g., 'global efficiency', 'modularity')
    """
    base_query = f'"{metric_name}"[Title/Abstract]'
    context = '(brain network[Title/Abstract] OR connectome[Title/Abstract] OR graph theory[Title/Abstract] OR structural connectivity[Title/Abstract])'

    return f"{base_query} AND {context}"


def build_cca_related_query(pgs_trait: str, brain_measure: str) -> str:
    """
    Build a query to find research connecting a PGS trait with brain measures.

    Args:
        pgs_trait: The polygenic score trait name
        brain_measure: The brain network measure or region
    """
    return f'("{pgs_trait}"[Title/Abstract]) AND ("{brain_measure}"[Title/Abstract]) AND (brain[Title/Abstract] OR neural[Title/Abstract])'


def search_for_pgs_brain_association(trait: str, max_results: int = 5) -> List[PubMedArticle]:
    """
    Search for literature on the association between a PGS trait and brain structure/function.
    """
    query = build_pgs_trait_query(trait, include_brain=True)
    return search_and_fetch_detailed(query, max_results)


def search_for_brain_region_function(region: str, max_results: int = 5) -> List[PubMedArticle]:
    """
    Search for literature on the function and role of a specific brain region.
    """
    query = build_brain_region_query(region, ["function", "cognition"])
    return search_and_fetch_detailed(query, max_results)


def search_for_network_property(metric: str, max_results: int = 5) -> List[PubMedArticle]:
    """
    Search for literature on a specific brain network property.
    """
    query = build_network_metric_query(metric)
    return search_and_fetch_detailed(query, max_results)


# ============================================================
# Citation formatting utilities
# ============================================================

def format_citation(article: PubMedArticle) -> str:
    """Format a PubMedArticle as a short citation string (Author et al., Year)."""
    if article.authors:
        first_author = article.authors[0].split()[-1]  # Last name of first author
        if len(article.authors) > 1:
            return f"{first_author} et al., {article.year}"
        else:
            return f"{first_author}, {article.year}"
    return f"PMID:{article.pmid}, {article.year}"


def format_reference(article: PubMedArticle) -> str:
    """Format a PubMedArticle as a full reference."""
    if article.authors:
        if len(article.authors) > 6:
            author_str = ", ".join(article.authors[:6]) + ", et al."
        else:
            author_str = ", ".join(article.authors)
    else:
        author_str = "Unknown authors"

    ref = f"{author_str} ({article.year}). {article.title}. {article.journal}."
    if article.doi:
        ref += f" https://doi.org/{article.doi}"

    return ref


def format_articles_for_context(articles: List[PubMedArticle], max_abstract_length: int = 500) -> str:
    """
    Format a list of articles into a context string for LLM processing.
    """
    if not articles:
        return "No relevant articles found."

    context_parts = []
    for i, article in enumerate(articles, 1):
        abstract = article.abstract[:max_abstract_length] + "..." if len(article.abstract) > max_abstract_length else article.abstract
        context_parts.append(
            f"[{i}] {format_citation(article)}\n"
            f"Title: {article.title}\n"
            f"Abstract: {abstract}\n"
        )

    return "\n".join(context_parts)


# ============================================================
# PGS trait name mapping
# ============================================================

PGS_TRAIT_NAMES = {
    'ADHDeur6': 'attention deficit hyperactivity disorder',
    'CPeur2': 'cognitive performance',
    'EAeur1': 'educational attainment',
    'MDDeur6': 'major depressive disorder',
    'INSOMNIAeur6': 'insomnia',
    'SNORINGeur1': 'snoring',
    'IQeur2': 'intelligence',
    'PTSDmeta6': 'post-traumatic stress disorder',
    'DEPmulti': 'depression',
    'BMImulti': 'body mass index',
    'ALCDEP_METAauto': 'alcohol dependence',
    'ASDauto': 'autism spectrum disorder',
    'ASPauto': 'antisocial personality',
    'BIPauto': 'bipolar disorder',
    'CANNABISauto': 'cannabis use',
    'CROSSauto': 'cross-disorder psychiatric',
    'DRINKauto': 'alcohol consumption',
    'EDauto': 'eating disorder',
    'NEUROTICISMauto': 'neuroticism',
    'OCDauto': 'obsessive-compulsive disorder',
    'RISK4PCauto': 'risk-taking behavior',
    'SCZ_METAauto': 'schizophrenia',
    'SMOKERauto': 'smoking behavior',
    'WORRYauto': 'worry',
    'ANXIETYauto': 'anxiety',
    'RISKTOLauto': 'risk tolerance',
    'Happieur4': 'happiness',
    'GHappieur2': 'general happiness',
    'GHappiMeaneur1': 'mean happiness',
    'GHappiHealth6': 'health-related happiness',
}


def get_trait_full_name(trait_code: str) -> str:
    """Convert PGS trait code to full name for literature search."""
    return PGS_TRAIT_NAMES.get(trait_code, trait_code)


if __name__ == "__main__":
    # Test basic search
    print("Testing basic PubMed search...")
    print(pubmed_search_and_get_abstracts("brain network development adolescent", 2))

    print("\n" + "="*50 + "\n")

    # Test detailed search
    print("Testing detailed article fetch...")
    articles = search_for_pgs_brain_association("cognitive performance", max_results=2)
    for article in articles:
        print(f"Title: {article.title}")
        print(f"Citation: {format_citation(article)}")
        print(f"Journal: {article.journal}")
        print("-" * 30)
