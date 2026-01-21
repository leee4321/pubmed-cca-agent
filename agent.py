"""
PubMed CCA Agent - Main Entry Point

This agent analyzes CCA results and generates scientific paper sections
(Results and Discussion) using PubMed literature and LLM synthesis.
"""

import os
import argparse
from datetime import datetime
from typing import Optional, Tuple

from google import genai
from dotenv import load_dotenv

from data_loader import load_default_cca_results, CCAResults
from results_generator import generate_results_section, extract_results_summary, ResultsSummary
from discussion_generator import (
    generate_discussion_section,
    gather_literature_for_discussion,
    save_discussion_output
)
from pubmed_tool import pubmed_search_and_get_abstracts

load_dotenv()


def check_genai_key():
    """Check if Google API key is set."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found in environment.")
        print("Please set it in your .env file or environment variables.")
        return False
    return True


def generate_paper_sections(
    base_dir: Optional[str] = None,
    x_loading_path: Optional[str] = None,
    y_loading_path: Optional[str] = None,
    freesurfer_label_path: Optional[str] = None,
    analysis_desc_path: Optional[str] = None,
    introduction_path: Optional[str] = None,
    methods_path: Optional[str] = None,
    output_dir: str = "results",
    use_llm: bool = True,
    search_literature: bool = True,
    verbose: bool = True
) -> Tuple[str, str, list]:
    """
    Generate Results and Discussion sections for a scientific paper.

    Args:
        base_dir: Directory containing input data files (default: input)
        x_loading_path: Path to X loading CSV (overrides base_dir)
        y_loading_path: Path to Y loading CSV (overrides base_dir)
        freesurfer_label_path: Path to FreeSurfer labels CSV (overrides base_dir)
        analysis_desc_path: Path to analysis description (overrides base_dir)
        introduction_path: Optional path to Introduction text file
        methods_path: Optional path to Methods text file
        output_dir: Directory to save output files (default: results)
        use_llm: Whether to use LLM for generation
        search_literature: Whether to search PubMed for prior literature
        verbose: Whether to print progress

    Returns:
        Tuple of (results_text, discussion_text, references_list)
    """
    # Ensure API key is present
    if use_llm and not check_genai_key():
        print("Falling back to rule-based generation (no LLM).")
        use_llm = False

    # Load CCA results
    if verbose:
        print("\n" + "=" * 60)
        print("Loading CCA Results...")
        print("=" * 60)

    # Check if individual files are specified
    if all([x_loading_path, y_loading_path, freesurfer_label_path, analysis_desc_path]):
        # Load from individual files
        if verbose:
            print("  Loading from specified files:")
            print(f"    X loading: {x_loading_path}")
            print(f"    Y loading: {y_loading_path}")
            print(f"    FreeSurfer labels: {freesurfer_label_path}")
            print(f"    Analysis description: {analysis_desc_path}")
            if introduction_path:
                print(f"    Introduction: {introduction_path}")
            if methods_path:
                print(f"    Methods: {methods_path}")
        
        cca_results = load_cca_results(
            x_loading_path=x_loading_path,
            y_loading_path=y_loading_path,
            freesurfer_label_path=freesurfer_label_path,
            analysis_description_path=analysis_desc_path,
            introduction_path=introduction_path,
            methods_path=methods_path
        )
    else:
        # Load from default directory
        if base_dir is None:
            base_dir = "input"
        
        if verbose:
            print(f"  Loading from directory: {base_dir}")
        
        cca_results = load_default_cca_results(base_dir)

    if verbose:
        print(f"  - Loaded {len(cca_results.x_loadings)} X loadings (PGS)")
        print(f"  - Loaded {len(cca_results.y_loadings)} Y loadings (BNM)")
        print(f"  - Loaded {len(cca_results.freesurfer_labels)} FreeSurfer labels")

    # Extract summary
    summary = extract_results_summary(cca_results)

    if verbose:
        print(f"\n  Significant X loadings (95% CI): {summary.n_significant_x}")
        print(f"  Significant Y loadings (95% CI): {summary.n_significant_y}")

    # Generate Results section
    if verbose:
        print("\n" + "=" * 60)
        print("Generating Results Section...")
        print("=" * 60)

    results_text = generate_results_section(
        cca_results=cca_results,
        use_llm=use_llm,
        base_dir=base_dir if base_dir else "."
    )

    # Generate Discussion section
    if verbose:
        print("\n" + "=" * 60)
        print("Generating Discussion Section...")
        print("=" * 60)

    discussion_text, references = generate_discussion_section(
        cca_results=cca_results,
        summary=summary,
        gather_literature=search_literature,
        verbose=verbose,
        base_dir=base_dir if base_dir else "."
    )

    return results_text, discussion_text, references


def save_output(
    results_text: str,
    discussion_text: str,
    references: list,
    output_dir: str = "results"
):
    """Save generated sections to files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Results
    results_path = os.path.join(output_dir, f"results_{timestamp}.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(results_text)
    print(f"Results saved to: {results_path}")

    # Save Discussion
    discussion_path = os.path.join(output_dir, f"discussion_{timestamp}.txt")
    with open(discussion_path, 'w', encoding='utf-8') as f:
        f.write("DISCUSSION\n")
        f.write("=" * 60 + "\n\n")
        f.write(discussion_text)
    print(f"Discussion saved to: {discussion_path}")

    # Save References
    references_path = os.path.join(output_dir, f"references_{timestamp}.txt")
    with open(references_path, 'w', encoding='utf-8') as f:
        f.write("REFERENCES\n")
        f.write("=" * 60 + "\n\n")
        for ref in references:
            f.write(ref + "\n\n")
    print(f"References saved to: {references_path}")

    # Save combined document
    combined_path = os.path.join(output_dir, f"paper_sections_{timestamp}.txt")
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("GENERATED PAPER SECTIONS\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("RESULTS\n")
        f.write("-" * 60 + "\n\n")
        f.write(results_text)
        f.write("\n\n")

        f.write("DISCUSSION\n")
        f.write("-" * 60 + "\n\n")
        f.write(discussion_text)
        f.write("\n\n")

        f.write("REFERENCES\n")
        f.write("-" * 60 + "\n\n")
        for ref in references:
            f.write(ref + "\n\n")

    print(f"Combined document saved to: {combined_path}")

    return combined_path


def interactive_mode():
    """Run the agent in interactive mode for custom queries."""
    print("\n" + "=" * 60)
    print("PubMed CCA Agent - Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  'generate' - Generate Results and Discussion sections")
    print("  'search <query>' - Search PubMed for a specific query")
    print("  'results' - Generate only Results section")
    print("  'discussion' - Generate only Discussion section")
    print("  'quit' or 'exit' - Exit the program")
    print()

    while True:
        try:
            user_input = input("\nAgent> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            elif user_input.lower() == 'generate':
                results, discussion, refs = generate_paper_sections(verbose=True)
                save_output(results, discussion, refs)

            elif user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                if query:
                    print(f"\nSearching PubMed for: {query}")
                    result = pubmed_search_and_get_abstracts(query, max_results=5)
                    print(result)
                else:
                    print("Please provide a search query.")

            elif user_input.lower() == 'results':
                print("\nGenerating Results section...")
                results = generate_results_section(use_llm=True)
                print("\n" + results)

            elif user_input.lower() == 'discussion':
                print("\nGenerating Discussion section...")
                discussion, refs = generate_discussion_section(verbose=True)
                print("\n" + discussion)
                print("\nReferences:")
                for ref in refs[:5]:
                    print(f"  {ref}")

            else:
                print(f"Unknown command: {user_input}")
                print("Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='PubMed CCA Agent - Generate paper sections from CCA results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default input directory
  python agent.py --mode generate

  # Specify individual CCA files (required)
  python agent.py --mode generate \\
    --x-loading input/bootstrap_result_summary_x_loading_comp1.csv \\
    --y-loading input/bootstrap_result_summary_y_loading_comp1.csv \\
    --freesurfer-labels input/FreeSurfer_label.csv \\
    --analysis-desc input/analysis_results_description.txt

  # With optional introduction and methods
  python agent.py --mode generate \\
    --x-loading input/x_loading.csv \\
    --y-loading input/y_loading.csv \\
    --freesurfer-labels input/labels.csv \\
    --analysis-desc input/description.txt \\
    --introduction input/intro.txt \\
    --methods input/methods.txt
        """
    )
    
    # Input file specification (alternative to --input-dir)
    input_group = parser.add_argument_group('Individual file specification (overrides --input-dir)')
    input_group.add_argument(
        '--x-loading',
        type=str,
        help='Path to X loading bootstrap results CSV file (required if not using --input-dir)'
    )
    input_group.add_argument(
        '--y-loading',
        type=str,
        help='Path to Y loading bootstrap results CSV file (required if not using --input-dir)'
    )
    input_group.add_argument(
        '--freesurfer-labels',
        type=str,
        help='Path to FreeSurfer labels CSV file (required if not using --input-dir)'
    )
    input_group.add_argument(
        '--analysis-desc',
        type=str,
        help='Path to analysis description text file (required if not using --input-dir)'
    )
    input_group.add_argument(
        '--introduction',
        type=str,
        help='Path to Introduction text file (optional)'
    )
    input_group.add_argument(
        '--methods',
        type=str,
        help='Path to Methods text file (optional)'
    )
    
    # General options
    parser.add_argument(
        '--mode', '-m',
        choices=['generate', 'interactive', 'results', 'discussion'],
        default='generate',
        help='Operation mode (default: generate)'
    )
    parser.add_argument(
        '--input-dir', '-i',
        default='input',
        help='Directory containing input data files (default: input). Ignored if individual files are specified.'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='results',
        help='Directory to save output files (default: results)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM generation (use rule-based approach)'
    )
    parser.add_argument(
        '--no-literature',
        action='store_true',
        help='Skip PubMed literature search'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()
    
    # Validate individual file arguments
    individual_files_specified = any([
        args.x_loading, args.y_loading, 
        args.freesurfer_labels, args.analysis_desc
    ])
    
    if individual_files_specified:
        # If any individual file is specified, all required files must be specified
        required_files = {
            '--x-loading': args.x_loading,
            '--y-loading': args.y_loading,
            '--freesurfer-labels': args.freesurfer_labels,
            '--analysis-desc': args.analysis_desc
        }
        
        missing_files = [name for name, path in required_files.items() if not path]
        
        if missing_files:
            parser.error(f"When specifying individual files, all required files must be provided. "
                        f"Missing: {', '.join(missing_files)}")

    if args.mode == 'interactive':
        interactive_mode()

    elif args.mode == 'generate':
        results, discussion, refs = generate_paper_sections(
            base_dir=args.input_dir if not individual_files_specified else None,
            x_loading_path=args.x_loading,
            y_loading_path=args.y_loading,
            freesurfer_label_path=args.freesurfer_labels,
            analysis_desc_path=args.analysis_desc,
            introduction_path=args.introduction,
            methods_path=args.methods,
            output_dir=args.output_dir,
            use_llm=not args.no_llm,
            search_literature=not args.no_literature,
            verbose=not args.quiet
        )
        save_output(results, discussion, refs, args.output_dir)

        print("\n" + "=" * 60)
        print("Generation complete!")
        print("=" * 60)

    elif args.mode == 'results':
        cca_results = load_default_cca_results(args.input_dir)
        results = generate_results_section(
            cca_results=cca_results,
            use_llm=not args.no_llm
        )
        print(results)

    elif args.mode == 'discussion':
        discussion, refs = generate_discussion_section(
            gather_literature=not args.no_literature,
            verbose=not args.quiet,
            base_dir=args.input_dir
        )
        print(discussion)
        print("\n\nReferences:")
        for ref in refs:
            print(ref)


# Legacy function for backward compatibility
def analyze_cca_results(csv_file=None):
    """
    Legacy function for backward compatibility.
    Now redirects to the new generate_paper_sections function.
    """
    print("Note: analyze_cca_results is deprecated.")
    print("Using new generation pipeline...")

    results, discussion, refs = generate_paper_sections(verbose=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(results)

    print("\n" + "=" * 60)
    print("DISCUSSION")
    print("=" * 60)
    print(discussion)

    return discussion


if __name__ == "__main__":
    main()
