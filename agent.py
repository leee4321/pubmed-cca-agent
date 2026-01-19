
import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from pubmed_tool import pubmed_search_and_get_abstracts

load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # prompt user if missing? For now assume it will be there or user provided.
    # We can try to get it from input if missing.
    pass

genai.configure(api_key=api_key)

# Initialize Model
model = genai.GenerativeModel('gemini-pro')

def analyze_cca_results(csv_file):
    """
    Analyzes CCA results using Gemini and PubMed.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        return "Error: CSV file not found."
    
    # 2. Prepare Context for the Agent
    # Get top contributors from the CSV
    top_genes = df[df['Type'] == 'Gene'].head(3)
    top_regions = df[df['Type'] == 'Brain_Region'].head(3)
    
    context_str = "Top CCA Genes:\n" + top_genes.to_string() + "\n\n"
    context_str += "Top CCA Brain Regions:\n" + top_regions.to_string() + "\n"
    
    print("Loaded CCA Context:")
    print(context_str)
    
    # 3. ReAct Loop (Simplified)
    # We will do a few turns of thought/action
    
    prompt = f"""
    You are a biomedical AI researcher. You have analyzed a dataset using canonical correlation analysis (CCA) and found a strong relationship between a set of genes and a set of brain regions.
    
    The top contributing variables are:
    {context_str}
    
    Your goal is to interpret the biological meaning of this relationship. 
    Why might these genes be related to these brain regions?
    
    You have access to a tool: `pubmed_search`. 
    To use it, output: ACTION: pubmed_search(query)
    
    If you have enough information to form a hypothesis, output: FINAL ANSWER: [your interpretation]
    
    Begin.
    """
    
    history = []
    
    # Turn 1
    print("\n--- Agent Step 1 ---")
    response = model.generate_content(prompt)
    print(f"Agent Thought: {response.text}")
    history.append(response.text)
    
    # Check for action
    if "ACTION: pubmed_search" in response.text:
        # Extract query
        # very naive parsing
        import re
        match = re.search(r"pubmed_search\((.*?)\)", response.text)
        if match:
            query = match.group(1).strip('"').strip("'")
            tool_output = pubmed_search_and_get_abstracts(query)
            print(f"Tool Output: {tool_output[:500]}...") # truncated for display
            
            # Feed back to model
            prompt_turn_2 = prompt + f"\n\nAgent: {response.text}\n\nTool Output: {tool_output}\n\nContinue reasoning."
            
            print("\n--- Agent Step 2 ---")
            response_2 = model.generate_content(prompt_turn_2)
            print(f"Agent Thought: {response_2.text}")
            return response_2.text
            
    return response.text

if __name__ == "__main__":
    analyze_cca_results('cca_results.csv')
