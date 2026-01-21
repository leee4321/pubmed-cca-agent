import os
import re
from typing import List, Tuple, Optional
from google import genai
import torch
import pickle
import logging
from datetime import datetime

logging.basicConfig(
    level=os.environ.get("LOGGING_LEVEL", "INFO"),
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    # Fallback if nltk is not available
    def sent_tokenize(text: str) -> List[str]:
        """Simple sentence tokenizer fallback."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from discussion_generator import LiteratureContext, generate_discussion_section
from pubmed_tool import PubMedArticle, format_citation, format_reference

class FactChecker:
    """Fact-checking agent that verifies claims in discussion text against cited abstracts."""
    
    def __init__(self, model_name: str = 'gemini-flash-latest'):
        """
        Initialize the FactChecker.
        
        Args:
            model_name: Name of the Gemini model to use for entailment checking
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.nli_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')
        self.nli_model_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base')
        self.nli_model.eval()
        
    def _find_matching_reference(self, cite: str, ref_list: List[PubMedArticle]) -> List[Tuple[int, PubMedArticle]]:
        """
        Find PubMedArticle instances that match a citation string.
        
        Args:
            cite: Citation string like "Smith et al., 2020"
            ref_list: List of PubMedArticle instances
            
        Returns:
            List of tuples (index, PubMedArticle) that match the citation
        """
        
        # logger.debug("Inside _find_matching_reference")
        # logger.debug(f"cite: {cite}")
        # Parse citation: extract last name and year
        cite_parts = cite.split('et al')
        if len(cite_parts) > 0:
            last_name = cite_parts[0].strip()
        else:
            last_name = cite.split(',')[0].strip()
        # logger.debug(f"last name: {last_name}")
        
        # Extract year
        year_match = re.findall(r'(\d{4})', cite)
        if not year_match:
            return []
        year = int(year_match[-1])  # Take the last 4-digit number (likely the year)
        # logger.debug(f"year: {year}")
        
        
        matched_refs = []
        for i, article in enumerate(ref_list):
            # logger.debug(f"article.authors: {article.authors}")
            # Get first author's last name
            if not article.authors:
                continue
            first_author_last_name = article.authors[0].split()[-1]
            
            # Get publication year
            try:
                publication_year = int(article.year)
            except (ValueError, AttributeError):
                continue
            # logger.debug(f"{i} {first_author_last_name} {publication_year}")
            
            # Match if last name and year match
            if (last_name.lower() == first_author_last_name.lower()) and (year == publication_year):
                matched_refs.append(article)
        
        logger.debug(f"matched_refs: {matched_refs}")
        return matched_refs
    
    def _check_entailment_nli(self, claim: str, abstract: str) -> bool:
        
        logger.debug(f"---- Inside _check_entailment_nli function ----")
        logger.debug(f"abstract: {abstract}, claim: {claim}")
        
        inputs = self.nli_model_tokenizer([abstract], [claim], padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            scores = self.nli_model(**inputs).logits
            label_mapping = ['contradiction', 'entailment', 'neutral']
            score_max = scores.argmax(dim=1)[0]
            label = label_mapping[score_max]
            
            logger.debug(f"nli label: {label}")
            if label == 'entailment':
                return True
            else:
                return False

        
        
    
    def _check_entailment_llm(self, claim: str, abstract: str) -> bool:
        """
        Check if a claim is entailed by an abstract using LLM.
        
        Args:
            claim: The claim to verify
            abstract: The abstract text to check against
            
        Returns:
            True if claim is entailed, False otherwise
        """
        PROMPT = f"""Verify if the claim is stated in the abstract. The claim must be relevant to the content of the abstract and entailed by the abstract.
Answer with yes or no. Enclose your answer with \\box{{}}.
- claim: {claim}
- abstract: {abstract}
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=PROMPT
            )
            text = response.text

        except Exception as e:
            raise RuntimeError(f"Error checking entailment with LLM: {e}")
        
        # Extract answer from boxed response
        # Look for \box{yes} or \box{no} pattern
        answer_match = re.search(r'\\box\{(\w+)\}', text, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).lower().strip()
            if answer == "yes":
                return True
            elif answer == "no":
                return False
        
        # Fallback: look for yes/no in the response
        text_lower = text.lower()
        if 'yes' in text_lower and 'no' not in text_lower:
            return True
        elif 'no' in text_lower and 'yes' not in text_lower:
            return False
        
        # If unclear, default to False (conservative approach)
        return False
        
    def extract_and_verify_citations(self, discussion: str, ref_list: List[str], refs: LiteratureContext) -> List[dict]:
        """
        Extract citations from discussion text and verify claims against abstracts.
        
        Args:
            discussion: The discussion text containing citations
            ref_list: List of formatted reference strings
            refs: LiteratureContext containing PubMedArticle instances
            
        Returns:
            List of dictionaries with citation verification results
        """
        logger.debug("Inside extract_and_verify_citations function.")
        # Extract citations using pattern matching
        # Pattern: "Author et al., Year" or "Author et al., Year; Author2 et al., Year2"
        # This pattern captures the entire sequence of citations separated by semicolons
        # Individual citation pattern (non-capturing): either "Author et al., Year" or "Author, Year"
        single_citation = r'(?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+et\s+al\.?,\s+\d{4}|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*,\s+\d{4})'
        # Match one citation, then zero or more semicolon-separated citations
        citation_pattern = r'(\s*' + single_citation + r'(?:\s*;\s*' + single_citation + r')*)'
        citation_matches = re.findall(citation_pattern, discussion)
        
        # citation_matches is now a list of strings (single capturing group)
        citations = [match.strip() for match in citation_matches]
        # logger.info(f"citation_matches: {citation_matches}")
        # logger.info(f"citations: {citations}")
        
        verification_results = []
        all_articles = refs.all_references
        
        for cite in citations:
            logger.debug(f"cite: {cite}")
            
            
            # Find position of citation in discussion
            cite_start_index = discussion.find(cite)
            if cite_start_index == -1:
                continue
            cite_start_index -= 1  # to account for (
            
            # Extract the sentence containing the claim (sentence before citation)
            sentences = sent_tokenize(discussion[:cite_start_index])
            if sentences:
                claim = sentences[-1].strip()
            else:
                # Fallback: take text before citation
                claim = discussion[max(0, cite_start_index-200):cite_start_index].strip()
            logger.debug(f"claim: {claim}")
                
                
            
            # Handle multiple citations separated by semicolons
            if ';' in cite:
                cite_parts = [c.strip() for c in cite.split(';')]
            else:
                cite_parts = [cite]
            
            for cite_part in cite_parts:
                logger.debug(f"cite_part: {cite_part}")
                
                # Find matching references
                matched_refs = self._find_matching_reference(cite_part, all_articles)
                
                if not matched_refs:
                    verification_results.append({
                        'citation': cite_part,
                        'claim': claim,
                        'verified': False,
                        'reason': 'No matching reference found'
                    })
                    continue
                
                # Check entailment for each matched reference
                entailed = False
                matched_article = None
                for article in matched_refs:
                    if article.abstract:
                        entailed = self._check_entailment_nli(claim, article.abstract)
                        if entailed:
                            matched_article = article
                            break
                if (matched_article is None) and (len(matched_refs) == 1):
                    matched_article = matched_refs[0]
                
                verification_results.append({
                    'citation': cite_part,
                    'claim': claim,
                    'verified': entailed,
                    'article': matched_article,
                    'matched_refs_count': len(matched_refs)
                })
        
        return verification_results
    
    def verify_discussion(self, discussion: str, refs: LiteratureContext) -> dict:
        """
        Verify all citations in a discussion text.
        
        Args:
            discussion: The discussion text to verify
            refs: LiteratureContext containing PubMedArticle instances
            
        Returns:
            Dictionary with verification summary
        """
        # Build reference list strings for citation matching
        ref_list = [format_reference(article) for article in refs.all_references]
        
        # Extract and verify citations
        verification_results = self.extract_and_verify_citations(discussion, ref_list, refs)
        
        # Calculate statistics
        total_citations = len(verification_results)
        verified_count = sum(1 for r in verification_results if r.get('verified', False))
        unverified_count = total_citations - verified_count
        
        return {
            'total_citations': total_citations,
            'verified': verified_count,
            'unverified': unverified_count,
            'verification_rate': verified_count / total_citations if total_citations > 0 else 0.0,
            'details': verification_results
        }
        
if __name__ == "__main__":
    
    test_file_dir = "test_files"
    output_dir = "results"
    
    if os.path.exists(os.path.join(test_file_dir, 'discussion.txt')) and os.path.exists(os.path.join(test_file_dir, 'literature_context.pickle')):
        with open(os.path.join(test_file_dir, 'discussion.txt'), 'r') as f:
            discussion = f.read()
        
        with open(os.path.join(test_file_dir, 'literature_context.pickle'), 'rb') as f:
            literature_context = pickle.load(f)
        
    else:
        discussion, refs, literature_text, literature_context = generate_discussion_section(
                gather_literature=True,
                verbose=True,
                base_dir="./input"
            )
        
        with open(os.path.join(test_file_dir, 'discussion.txt'), 'w') as f:
            f.write(discussion)
            
        with open(os.path.join(test_file_dir, 'literature_context.pickle'), 'wb') as f:
            pickle.dump(literature_context, f)
        
    
    factchecker = FactChecker()
    verification_result = factchecker.verify_discussion(discussion, literature_context)
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(output_dir, f'verification_result_{timestamp}.txt'), 'w') as f:
        f.write('='*50)
        f.write('\n')
        f.write(f"Total citations: {verification_result['total_citations']}\n")
        f.write(f"Verified count: {verification_result['verified']}\n")
        f.write(f"Verification rate: {verification_result['verification_rate']}\n")
        
        f.write('Raw verification results\n')
        for result in verification_result['details']:
            f.write(f"Citation: {result['citation']}\n")
            f.write(f"Claim: {result['claim']}\n")
            f.write(f"Verified: {result['verified']}\n")
            if result.get('article', None) is not None:
                try:
                    f.write(f"Reference: {result['article'].authors[0]} ({result['article'].year}) {result['article'].title}\n")
                    f.write(f"Abstract of the Reference: {result['article'].abstract}\n")
                except:
                    logger.debug(result['article'])
            f.write("-" * 30 + "\n")
        
        f.write('='*50)
    
    
    