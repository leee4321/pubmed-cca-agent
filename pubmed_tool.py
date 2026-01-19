
import requests
import time
import xml.etree.ElementTree as ET

# Define base URLs for E-utilities
BASE_URL_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BASE_URL_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def search_pubmed(query, max_results=5):
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
        response = requests.get(BASE_URL_SEARCH, params=params)
        response.raise_for_status()
        data = response.json()
        pmids = data.get('esearchresult', {}).get('idlist', [])
        return pmids
    except Exception as e:
        print(f"Error searching PubMed for '{query}': {e}")
        return []

def fetch_details(pmid_list):
    """
    Fetch abstracts for a list of PMIDs.
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
        response = requests.get(BASE_URL_FETCH, params=params)
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
            
        return articles
            
    except Exception as e:
        print(f"Error fetching details: {e}")
        return []

def pubmed_search_and_get_abstracts(query, max_results=3):
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

if __name__ == "__main__":
    # Test
    print(pubmed_search_and_get_abstracts("APOE Hippocampus", 1))
