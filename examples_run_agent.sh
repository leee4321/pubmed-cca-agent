#!/bin/bash
# Example script showing how to run the agent with individual file specification

# Example 1: Use default input directory
echo "Example 1: Using default input directory"
python agent.py --mode generate

# Example 2: Specify individual files (all required files)
echo -e "\nExample 2: Using individual files (required only)"
python agent.py --mode generate \
  --x-loading input/bootstrap_result_summary_x_loading_comp1.csv \
  --y-loading input/bootstrap_result_summary_y_loading_comp1.csv \
  --freesurfer-labels input/FreeSurfer_label.csv \
  --analysis-desc input/analysis_results_description.txt

# Example 3: Specify individual files with optional introduction and methods
echo -e "\nExample 3: Using individual files (with optional files)"
python agent.py --mode generate \
  --x-loading input/bootstrap_result_summary_x_loading_comp1.csv \
  --y-loading input/bootstrap_result_summary_y_loading_comp1.csv \
  --freesurfer-labels input/FreeSurfer_label.csv \
  --analysis-desc input/analysis_results_description.txt \
  --introduction input/Introduction.txt \
  --methods input/Methods.txt \
  --output-dir results

# Example 4: Custom dataset with different file names
echo -e "\nExample 4: Custom dataset"
# python agent.py --mode generate \
#   --x-loading data/my_x_loadings.csv \
#   --y-loading data/my_y_loadings.csv \
#   --freesurfer-labels data/brain_labels.csv \
#   --analysis-desc data/description.txt \
#   --output-dir custom_results

# Example 5: Without LLM (rule-based generation only)
echo -e "\nExample 5: Without LLM"
# python agent.py --mode generate \
#   --x-loading input/bootstrap_result_summary_x_loading_comp1.csv \
#   --y-loading input/bootstrap_result_summary_y_loading_comp1.csv \
#   --freesurfer-labels input/FreeSurfer_label.csv \
#   --analysis-desc input/analysis_results_description.txt \
#   --no-llm

# Example 6: Without literature search (faster)
echo -e "\nExample 6: Without literature search"
# python agent.py --mode generate \
#   --x-loading input/bootstrap_result_summary_x_loading_comp1.csv \
#   --y-loading input/bootstrap_result_summary_y_loading_comp1.csv \
#   --freesurfer-labels input/FreeSurfer_label.csv \
#   --analysis-desc input/analysis_results_description.txt \
#   --no-literature
