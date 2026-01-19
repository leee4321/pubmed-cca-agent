# PubMed CCA Agent

This project implements an AI agent that interprets Canonical Correlation Analysis (CCA) results using Google Gemini and the PubMed API.

## Overview

The agent takes multivariable-to-multivariable relationship results (provided in CSV format) and performs the following:
1.  **Top Contributor Identification**: Identifies the genes and brain regions (or other variables) that contribute most to the canonical variates.
2.  **Autonomous Research**: Uses a ReAct loop to search PubMed for biological and medical literature related to these top variables.
3.  **Synthesis & Interpretation**: Synthesizes the retrieved abstracts to provide a high-level biological hypothesis or explanation for the observed correlations.

## Project Structure

- `agent.py`: The main ReAct agent loop using Google Gemini.
- `pubmed_tool.py`: A wrapper for NCBI E-utilities to search and fetch abstracts.
- `generate_cca_data.py`: A script to generate synthetic CCA results for testing.
- `cca_results.csv`: Example input file (generated or provided).

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/leee4321/pubmed-cca-agent.git
    cd pubmed-cca-agent
    ```

2.  **Environment Setup**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configuration**:
    Create a `.env` file and add your Google AI Studio API Key:
    ```
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

## Usage

Run the agent to analyze the current results:
```bash
python agent.py
```

## Credits
Inspired by the [Google Gemini Cookbook](https://github.com/google-gemini/cookbook).
