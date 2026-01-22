# PubMed CCA Agent

An AI-powered agent that analyzes sparse Canonical Correlation Analysis (CCA) results and automatically generates scientific paper sections (Results and Discussion) using Google Gemini and PubMed literature search. **Now with automated fact-checking and figure generation!**

## Overview

This tool is designed for neuroimaging genetics research, specifically for analyzing the relationship between **Polygenic Scores (PGS)** and **Brain Network Measures (BNMs)**. The agent:

1. **Loads CCA Bootstrap Results**: Reads bootstrap-validated CCA loadings with confidence intervals
2. **Generates Results Section**: Creates a formal Results section with statistical summaries
3. **Searches Prior Literature**: Automatically queries PubMed for relevant research on identified traits and brain regions
4. **Generates Discussion Section**: Synthesizes findings with prior literature using Nature-style prompts
5. **🆕 Verifies Citations**: Uses NLI-based fact-checking to verify claims against cited abstracts
6. **🆕 Creates Visualizations**: Automatically generates publication-ready figures

## 🆕 What's New

### Citation Verification (Fact-Checking Agent)
- Automatically verifies that citations in the Discussion support the claims made
- Uses cross-encoder NLI (Natural Language Inference) model
- Generates verification reports with detailed analysis

### Automated Figure Generation
- Creates publication-ready visualizations of CCA loadings
- Generates separate figures for PGS and brain network loadings
- Saves figures in both PNG and PDF formats

### Enhanced Literature Context
- Saves complete literature text for debugging and verification
- No abstract length limits for comprehensive analysis
- Improved citation integration with Nature-style prompts

## Project Structure

```
pubmed_cca_agent/
├── agent.py                 # Main entry point and CLI interface
├── data_loader.py           # Data loading utilities for CCA results and labels
├── results_generator.py     # Results section generation (rule-based + LLM)
├── discussion_generator.py  # Discussion section generation with PubMed integration
├── pubmed_tool.py           # PubMed API wrapper with specialized query builders
├── factchecking_agent.py    # 🆕 Citation verification and fact-checking
├── figure_generator.py      # 🆕 Automated figure generation
├── requirements.txt         # Python dependencies
├── README.md
├── MERGE_SUMMARY.md        # Details on recent code merge
├── input/                   # Input data files (default location)
│   ├── bootstrap_result_summary_x_loading_comp1.csv
│   ├── bootstrap_result_summary_y_loading_comp1.csv
│   ├── FreeSurfer_label.csv
│   ├── analysis_results_description.txt
│   ├── Introduction.txt
│   └── Methods.txt
└── results/                 # Generated output files
```

## Input Files Required

### Option 1: Default Directory Structure (Recommended)

Place the following files in the `input/` directory:

| File | Required | Description |
|------|----------|-------------|
| `bootstrap_result_summary_x_loading_comp1.csv` | ✅ Yes | X loadings (PGS) with bootstrap CIs |
| `bootstrap_result_summary_y_loading_comp1.csv` | ✅ Yes | Y loadings (BNM) with bootstrap CIs |
| `FreeSurfer_label.csv` | ✅ Yes | Brain region abbreviation to full name mapping |
| `analysis_results_description.txt` | ✅ Yes | Description of the CCA analysis |
| `Introduction.txt` | ⭕ Optional | Paper introduction text (for context) |
| `Methods.txt` | ⭕ Optional | Paper methods text (for context) |

### Option 2: Individual File Specification

You can specify individual files using command-line arguments instead of using the default directory structure. This is useful for custom datasets or different file naming conventions.

**Required arguments** (all 4 must be provided together):
- `--x-loading`: Path to X loading CSV file
- `--y-loading`: Path to Y loading CSV file
- `--freesurfer-labels`: Path to FreeSurfer labels CSV file
- `--analysis-desc`: Path to analysis description text file

**Optional arguments**:
- `--introduction`: Path to Introduction text file
- `--methods`: Path to Methods text file

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

3. **Install NLTK data** (required for fact-checking):
    ```bash
    python -c "import nltk; nltk.download('punkt')"
    ```

4. **Configure API Key**:
    Create a `.env` file with your Google AI Studio API key:
    ```
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

## Usage

### Basic Usage (Default Directory)

Generate both Results and Discussion using files from `input/` directory:
```bash
python agent.py --mode generate
```

### 🆕 Generate with Figures

Create visualizations along with text sections:
```bash
python agent.py --mode generate --generate-figures
```

### Individual File Specification

Specify custom CCA result files:
```bash
# Required files only
python agent.py --mode generate \
  --x-loading input/bootstrap_result_summary_x_loading_comp1.csv \
  --y-loading input/bootstrap_result_summary_y_loading_comp1.csv \
  --freesurfer-labels input/FreeSurfer_label.csv \
  --analysis-desc input/analysis_results_description.txt

# With optional introduction and methods
python agent.py --mode generate \
  --x-loading input/bootstrap_result_summary_x_loading_comp1.csv \
  --y-loading input/bootstrap_result_summary_y_loading_comp1.csv \
  --freesurfer-labels input/FreeSurfer_label.csv \
  --analysis-desc input/analysis_results_description.txt \
  --introduction input/Introduction.txt \
  --methods input/Methods.txt

# Custom dataset with different file names and figures
python agent.py --mode generate \
  --x-loading data/my_x_loadings.csv \
  --y-loading data/my_y_loadings.csv \
  --freesurfer-labels data/brain_labels.csv \
  --analysis-desc data/description.txt \
  --generate-figures \
  --output-dir custom_results
```

### 🆕 Citation Verification (Fact-Checking)

Verify citations in a generated discussion:
```bash
python factchecking_agent.py
```

This will:
1. Generate a Discussion section (or load existing one from `test_files/`)
2. Extract all citations from the text
3. Verify each claim against the cited abstract using NLI
4. Save a detailed verification report to `results/verification_result_*.txt`

### Other Modes

Generate only Results section:
```bash
python agent.py --mode results
```

Generate only Discussion section:
```bash
python agent.py --mode discussion
```

Interactive mode:
```bash
python agent.py --mode interactive
```

### Command Line Options

```
General Options:
  --mode, -m              Operation mode: generate, interactive, results, discussion
  --output-dir, -o        Directory to save output files (default: results)
  --no-llm                Disable LLM, use rule-based generation only
  --no-literature         Skip PubMed literature search
  --quiet, -q             Suppress progress output
  --generate-figures      🆕 Generate visualization figures for results

Individual File Specification:
  --x-loading             Path to X loading CSV (required with other CCA files)
  --y-loading             Path to Y loading CSV (required with other CCA files)
  --freesurfer-labels     Path to FreeSurfer labels CSV (required with other CCA files)
  --analysis-desc         Path to analysis description (required with other CCA files)
  --introduction          Path to Introduction text (optional)
  --methods               Path to Methods text (optional)
```

### Additional Examples

```bash
# Full generation with figures and custom directories
python agent.py -m generate --generate-figures -o ./results

# Quick results without PubMed search (faster)
python agent.py -m generate --no-literature

# Rule-based generation (no API key needed)
python agent.py -m generate --no-llm

# Custom files with figures but without literature search
python agent.py -m generate \
  --x-loading data/x.csv \
  --y-loading data/y.csv \
  --freesurfer-labels data/labels.csv \
  --analysis-desc data/desc.txt \
  --no-literature \
  --generate-figures \
  --output-dir results
```

## Output Files

Generated files are saved to the output directory with timestamps:

### Standard Outputs
- `results_YYYYMMDD_HHMMSS.txt` - Results section
- `discussion_YYYYMMDD_HHMMSS.txt` - Discussion section
- `references_YYYYMMDD_HHMMSS.txt` - Reference list from PubMed
- `paper_sections_YYYYMMDD_HHMMSS.txt` - Combined document

### 🆕 New Outputs
- `literature_text_YYYYMMDD_HHMMSS.txt` - Complete literature context used by LLM (for debugging)
- `x_loading_figure_*.png/pdf` - PGS loading visualization
- `y_loading_figure_*.png/pdf` - Brain network loading visualization
- `verification_result_YYYYMMDD_HHMMSS.txt` - Citation verification report

## Module Details

### `agent.py`
- Main CLI interface with flexible file specification
- Coordinates all components (data loading, generation, output)
- Supports multiple operation modes

### `pubmed_tool.py`
- PubMed E-utilities API wrapper
- Specialized query builders for:
  - PGS trait-brain associations
  - Brain region functions
  - Network metrics (efficiency, modularity, etc.)
- Citation formatting utilities
- PGS trait code to full name mapping
- **🆕 No abstract length limits for comprehensive context**

### `data_loader.py`
- Loads bootstrap CSV files with CCA loadings
- Parses FreeSurfer brain region labels
- Identifies significant loadings based on confidence intervals
- Categorizes Y loadings by metric type (degree, strength, etc.)

### `results_generator.py`
- Extracts summary statistics from CCA results
- Rule-based Results text generation
- **Enhanced LLM prompts for Nature-style formatting**
- Uses `gemini-2.5-flash` for high-quality output

### `discussion_generator.py`
- Automated PubMed literature gathering
- Context-aware Discussion generation with Nature-style prompts
- Reference list compilation
- **🆕 Returns literature_text for debugging and verification**
- Uses `gemini-2.5-flash` model

### 🆕 `factchecking_agent.py`
- **Citation extraction** from Discussion text using regex
- **Claim extraction** using sentence tokenization
- **NLI-based verification** using cross-encoder/nli-deberta-v3-base
- Generates detailed verification reports with:
  - Total citations found
  - Verification rate
  - Individual claim analysis
  - Matched abstracts

### 🆕 `figure_generator.py`
- **Automated visualization** of CCA loadings
- Creates separate figures for X (PGS) and Y (BNM) loadings
- **Publication-ready formatting**:
  - Shows loading estimates with 95% confidence intervals
  - Highlights significant loadings
  - Color-coded by sign (positive/negative)
- Saves in both PNG and PDF formats

## Dependencies

### Core Dependencies
- `google-generativeai>=0.3.0` - Google Gemini API
- `requests>=2.28.0` - HTTP requests for PubMed API
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical operations
- `python-dotenv>=1.0.0` - Environment variable management

### 🆕 Visualization Dependencies
- `matplotlib>=3.5.0` - Figure generation
- `seaborn>=0.11.0` - Enhanced visualizations
- `nilearn>=0.9.0` - Neuroimaging visualizations

### 🆕 Fact-Checking Dependencies
- `nltk>=3.8` - Natural language processing and sentence tokenization
- `transformers` - For NLI model (cross-encoder/nli-deberta-v3-base)
- `torch` - PyTorch for running NLI model
- `biopython>=1.79` - Biological data processing

## Advanced Features

### Fact-Checking Workflow

```python
from factchecking_agent import FactChecker
from discussion_generator import generate_discussion_section

# Generate discussion with literature
discussion, refs, literature_text, literature_context = generate_discussion_section(
    gather_literature=True,
    verbose=True,
    base_dir="./input"
)

# Initialize fact checker
factchecker = FactChecker()

# Verify citations
verification_result = factchecker.verify_discussion(discussion, literature_context)

print(f"Total citations: {verification_result['total_citations']}")
print(f"Verified: {verification_result['verified']}")
print(f"Verification rate: {verification_result['verification_rate']:.2%}")

# Access detailed results
for result in verification_result['details']:
    print(f"Citation: {result['citation']}")
    print(f"Claim: {result['claim']}")
    print(f"Verified: {result['verified']}")
```

### Programmatic API Usage

```python
from agent import generate_paper_sections

# Generate all sections programmatically
results, discussion, references, literature_text = generate_paper_sections(
    base_dir="input",
    output_dir="results",
    use_llm=True,
    search_literature=True,
    generate_figures=True,
    verbose=True
)

# Access individual components
print(f"Results length: {len(results)} characters")
print(f"Discussion length: {len(discussion)} characters")
print(f"Number of references: {len(references)}")
```

## Model Configuration

The agent uses Google's Gemini models:
- **Default model**: `gemini-2.5-flash` (fast, high-quality)
- **Alternative**: `gemini-2.5-pro` (even higher quality, slower)

To change models, see `MODEL_CONFIG.md` for details.

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `.env` file contains a valid `GOOGLE_API_KEY`
2. **NLTK Error**: Run `python -c "import nltk; nltk.download('punkt')"`
3. **Import Error for transformers**: Install with `pip install transformers torch`
4. **Memory Error with NLI model**: The fact-checking agent loads a transformer model. Ensure sufficient RAM (4GB+ recommended)

### Performance Tips

- Use `--no-literature` flag to skip PubMed search for faster generation
- Use `--no-llm` for rule-based generation without API calls
- Fact-checking is computationally intensive; consider running it separately

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{pubmed_cca_agent,
  title={PubMed CCA Agent: AI-Powered Scientific Writing for Neuroimaging Genetics},
  author={Lee, Eunji},
  year={2026},
  url={https://github.com/leee4321/pubmed-cca-agent}
}
```

## Credits

- Inspired by the [Google Gemini Cookbook](https://github.com/google-gemini/cookbook)
- Uses [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) for PubMed access
- Fact-checking powered by [cross-encoder NLI models](https://huggingface.co/cross-encoder/nli-deberta-v3-base)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### v2.0.0 (2026-01-22)
- ✨ Added fact-checking agent for citation verification
- ✨ Added automated figure generation
- ✨ Enhanced prompts with Nature-style formatting
- ✨ Added literature_text debugging output
- ✨ Removed abstract length limits for comprehensive analysis
- 🔧 Updated to gemini-2.5-flash model
- 🔧 Improved CLI with individual file specification
- 📝 Added comprehensive merge documentation

### v1.0.0 (2026-01-19)
- 🎉 Initial release with basic functionality
- Results and Discussion generation
- PubMed literature integration
- Bootstrap CCA analysis support
