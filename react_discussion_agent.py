import os
import re
import time
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai
from dotenv import load_dotenv

from data_loader import CCAResults
from results_generator import ResultsSummary, get_trait_full_name
from pubmed_tool import (
    search_and_fetch_detailed,
    format_citation,
    format_articles_for_context,
    PubMedArticle
)

load_dotenv()

class ReActDiscussionAgent:
    """
    Agent that generates a paper discussion section using the ReAct (Thought-Action-Observation) framework.
    """
    
    def __init__(self, cca_results: CCAResults, summary: ResultsSummary, results_text: str = "", model_name: str = "gemini-1.5-flash"):
        self.cca_results = cca_results
        self.summary = summary
        self.results_text = results_text
        self.model_name = model_name
        self.written_sentences = []
        self.history = []
        self.all_found_articles = []
        self.max_steps = 15
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def _execute_search(self, query: str) -> str:
        """Action: search_pubmed(query)"""
        print(f"  [Action] Searching PubMed: {query}")
        try:
            articles = search_and_fetch_detailed(query, max_results=3)
            self.all_found_articles.extend(articles)
            return format_articles_for_context(articles)
        except Exception as e:
            return f"Error searching PubMed: {str(e)}"
            
    def _execute_write(self, text: str) -> str:
        """Action: write(text)"""
        print(f"  [Action] Writing: {text[:50]}...")
        # Check if the text actually ends with a sentence terminator
        text = text.strip()
        if text:
            if not any(text.endswith(p) for p in ['.', '!', '?']):
                text += "."
            self.written_sentences.append(text)
        return f"Sentence(s) recorded successfully. Current draft has {len(self.written_sentences)} sentence(s)."

    def _get_system_prompt(self) -> str:
        top_neg_pgs = [(get_trait_full_name(n), e) for n, e, _, _ in self.summary.significant_negative_pgs[:3]]
        top_pos_pgs = [(get_trait_full_name(n), e) for n, e, _, _ in self.summary.significant_positive_pgs[:3]]
        
        findings_text = f"""
KEY FINDINGS FROM CCA:
- Overall Analysis: {self.summary.analysis_description}
- Highly Negative PGS Variables: {', '.join([f'{n} (β={e:.3f})' for n, e in top_neg_pgs])}
- Highly Positive PGS Variables: {', '.join([f'{n} (β={e:.3f})' for n, e in top_pos_pgs])}

GENERATED RESULTS SECTION (FOR REFERENCE):
{self.results_text}
"""

        return f"""You are a world-class neuroimaging scientist. Your task is to write the Discussion section of a scientific paper based on CCA results.
You must use the ReAct (Thought -> Action -> Observation) framework to build the discussion sentence-by-sentence.

{findings_text}

Available Actions:
1. `search_pubmed(query)`: Search for literature to support your discussion.
2. `write(text)`: Write one or more sentences for the final paper. Include citations like [1], [2] based on your search results.
3. `finish()`: Call this when you have completed a comprehensive discussion.

Format:
Thought: Your reasoning about what to do next.
Action: `action_name(arguments)`
Observation: The result of your action.

Important Rules:
- Only write text meant for the final paper using the `write()` action.
- Ensure every `write()` action that makes a claim includes a citation based on your `search_pubmed` results.
- Number your citations [1], [2], etc., in the order they appear in the final text.
- Be specific about brain regions and traits.
- Maximum 15 steps total. Focus on high-impact findings.

Begin!
"""

    def generate(self) -> Tuple[str, List[PubMedArticle]]:
        prompt = self._get_system_prompt()
        current_history = []
        
        for step in range(self.max_steps):
            # Formulate the prompt with history
            input_text = prompt + "\n" + "\n".join(current_history) + "\nThought:"
            
            try:
                response = self.model.generate_content(input_text)
                response_text = response.text.strip()
            except Exception as e:
                print(f"Error calling Gemini: {e}")
                break
            
            print(f"\n[Step {step+1}]")
            
            # Parsing "Thought: ... Action: ... "
            # We look for Thought: and Action: in the response
            thought_part = ""
            action_part = ""
            
            if "Thought:" in response_text:
                parts = response_text.split("Thought:", 1)[1].split("Action:", 1)
                thought_part = parts[0].strip()
                if len(parts) > 1:
                    action_part = parts[1].strip()
            elif "Action:" in response_text:
                action_part = response_text.split("Action:", 1)[1].strip()
            
            print(f"  Thought: {thought_part}")
            
            # Identify action and observation
            observation = ""
            action_name = ""
            action_args = ""
            
            if not action_part:
                # Check for finish() in thought or just raw
                if "finish()" in response_text.lower():
                    print("  Action: finish()")
                    break
                observation = "Error: No action found. Please provide an Action: `action_name(args)`."
            else:
                match = re.search(r"(\w+)\((.*?)\)", action_part)
                if match:
                    action_name = match.group(1)
                    action_args = match.group(2).strip("'\"")
                    
                    if action_name == "search_pubmed":
                        observation = self._execute_search(action_args)
                    elif action_name == "write":
                        observation = self._execute_write(action_args)
                    elif action_name == "finish":
                        print("  Action: finish()")
                        break
                    else:
                        observation = f"Error: Unknown action '{action_name}'"
                else:
                    observation = "Error: Could not parse action format. Use `action_name(args)`."
            
            print(f"  Observation: {observation[:100]}...")
            
            current_history.append(f"Thought: {thought_part}")
            current_history.append(f"Action: `{action_name}({action_args})`")
            current_history.append(f"Observation: {observation}")
            
            # Context window management
            if len(current_history) > 10:
                current_history = current_history[-8:]

        raw_text = " ".join(self.written_sentences)
        final_text = self._apply_smoothing(raw_text)
        
        return final_text, self.all_found_articles

    def _apply_smoothing(self, text: str) -> str:
        if not text:
            return "No discussion generated."
            
        print("\n[Smoothing Agent] Refining text flow...")
        prompt = f"""You are a scientific editor. Refine the following paper discussion for flow, tone, and logical structure. 
Maintain all citations and scientific findings. Use a single-paragraph or multi-paragraph format as appropriate for a high-impact journal.

Draft Text:
{text}

Refined Discussion:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return text

def generate_discussion_react(cca_results: CCAResults, summary: ResultsSummary, results_text: str = "") -> Tuple[str, List[str], str]:
    """Function to be called by agent.py"""
    agent = ReActDiscussionAgent(cca_results, summary, results_text=results_text)
    discussion, articles = agent.generate()
    
    # Format references
    seen_pmids = set()
    unique_articles = []
    for a in articles:
        if a.pmid not in seen_pmids:
            unique_articles.append(a)
            seen_pmids.add(a.pmid)
            
    formatted_refs = [f"[{i+1}] {format_citation(a)}: {a.title}" for i, a in enumerate(unique_articles)]
    literature_text = format_articles_for_context(unique_articles)
    
    return discussion, formatted_refs, literature_text
