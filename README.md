# PubMed CCA Agent

An AI-powered agent that analyzes sparse Canonical Correlation Analysis (CCA) results and automatically generates scientific paper sections (Results and Discussion) using Google Gemini and PubMed literature search.

## Overview

This tool is designed for neuroimaging genetics research, specifically for analyzing the relationship between **Polygenic Scores (PGS)** and **Brain Network Measures (BNMs)**. The agent:

1. **Loads CCA Bootstrap Results**: Reads bootstrap-validated CCA loadings with confidence intervals
2. **Generates Results Section**: Creates a formal Results section with statistical summaries
3. **Searches Prior Literature**: Automatically queries PubMed for relevant research on identified traits and brain regions
4. **Generates Discussion Section**: Synthesizes findings with prior literature to create a comprehensive Discussion

## Project Structure

```
pubmed_cca_agent/
├── agent.py                 # Main entry point and CLI interface
├── data_loader.py           # Data loading utilities for CCA results and labels
├── results_generator.py     # Results section generation (rule-based + LLM)
├── discussion_generator.py  # Discussion section generation with PubMed integration
├── pubmed_tool.py           # PubMed API wrapper with specialized query builders
├── requirements.txt         # Python dependencies
├── README.md
├── input/                   # Input data files (default location)
│   ├── bootstrap_result_summary_x_loading_comp1.csv
│   ├── bootstrap_result_summary_y_loading_comp1.csv
│   ├── FreeSurfer_label.csv
│   ├── analysis_results_description.txt
│   ├── Introduction.txt
│   └── Methods.txt
└── output/                  # Generated output files
```

## Input Files Required

Place the following files in the `input/` directory:

| File | Description |
|------|-------------|
| `bootstrap_result_summary_x_loading_comp1.csv` | X loadings (PGS) with bootstrap CIs |
| `bootstrap_result_summary_y_loading_comp1.csv` | Y loadings (BNM) with bootstrap CIs |
| `FreeSurfer_label.csv` | Brain region abbreviation to full name mapping |
| `analysis_results_description.txt` | Description of the CCA analysis |
| `Introduction.txt` | Paper introduction text (for context) |
| `Methods.txt` | Paper methods text (for context) |

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/leee4321/pubmed-cca-agent.git
    cd pubmed-cca-agent
    ```

2. **Create virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Configure API Key**:
    Create a `.env` file with your Google AI Studio API key:
    ```
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

## Usage

### Generate Both Results and Discussion
```bash
python agent.py --mode generate
```

### Generate Only Results Section
```bash
python agent.py --mode results
```

### Generate Only Discussion Section
```bash
python agent.py --mode discussion
```

### Interactive Mode
```bash
python agent.py --mode interactive
```

### Command Line Options
```
Options:
  --mode, -m        Operation mode: generate, interactive, results, discussion
  --input-dir, -i   Directory containing input data files (default: input)
  --output-dir, -o  Directory to save output files (default: output)
  --no-llm          Disable LLM, use rule-based generation only
  --no-literature   Skip PubMed literature search
  --quiet, -q       Suppress progress output
```

### Examples

```bash
# Full generation with custom directories
python agent.py -m generate -i ./data -o ./results

# Quick results without PubMed search
python agent.py -m generate --no-literature

# Rule-based generation (no API key needed)
python agent.py -m generate --no-llm
```

## Output Files

Generated files are saved to the output directory with timestamps:
- `results_YYYYMMDD_HHMMSS.txt` - Results section
- `discussion_YYYYMMDD_HHMMSS.txt` - Discussion section
- `references_YYYYMMDD_HHMMSS.txt` - Reference list from PubMed
- `paper_sections_YYYYMMDD_HHMMSS.txt` - Combined document

## Module Details

### `pubmed_tool.py`
- PubMed E-utilities API wrapper
- Specialized query builders for:
  - PGS trait-brain associations
  - Brain region functions
  - Network metrics (efficiency, modularity, etc.)
- Citation formatting utilities
- PGS trait code to full name mapping

### `data_loader.py`
- Loads bootstrap CSV files with CCA loadings
- Parses FreeSurfer brain region labels
- Identifies significant loadings based on confidence intervals
- Categorizes Y loadings by metric type (degree, strength, etc.)

### `results_generator.py`
- Extracts summary statistics from CCA results
- Rule-based Results text generation
- LLM-enhanced Results section with Gemini

### `discussion_generator.py`
- Automated PubMed literature gathering
- Context-aware Discussion generation
- Reference list compilation

## Dependencies

- `google-generativeai>=0.3.0` - Google Gemini API
- `requests>=2.28.0` - HTTP requests for PubMed API
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical operations
- `python-dotenv>=1.0.0` - Environment variable management

## Credits

- Inspired by the [Google Gemini Cookbook](https://github.com/google-gemini/cookbook)
- Uses [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) for PubMed access
