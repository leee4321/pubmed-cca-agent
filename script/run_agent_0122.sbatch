#!/bin/bash
#SBATCH --job-name pubmed_cca_v2
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=node3
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/connectome/leee4321/pubmed_cca_agent/script/slurm_logs/R-%x.o%j
#SBATCH -e /scratch/connectome/leee4321/pubmed_cca_agent/script/slurm_logs/R-%x.e%j
set +x

# 1. Move to project directory
cd /scratch/connectome/leee4321/pubmed_cca_agent

# 2. Activate Conda environment (Python 3.10)
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /home/connectome/leee4321/.conda/envs/pubmed_cca_310

# Critical: Ignore local packages (~/.local) to prevent version conflicts
export PYTHONNOUSERSITE=1

# Debug: check environment
echo "=== Environment Info ==="
which python
python --version
echo "========================"

# 3. Install/Update dependencies
# Added nltk, seaborn, matplotlib, etc. from the recent merge
echo "Installing/Updating requirements..."
pip install -r requirements.txt

# 4. Download NLTK data (Required for Fact Checking Agent)
# This downloads 'punkt' tokenizer data if not already present
echo "Checking NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# 5. Run the Agent (Generate Mode)
# - Generates Results and Discussion sections
# - Uses Gemini 2.5 Flash model
# - Generates Figures (--generate-figures)
# - Contextualized with Introduction and Methods
# Run in full generation mode with ReAct enabled for Discussion
# This ensures Discussion is tightly coupled with the generated Results
python agent.py --mode generate --react \
  --x-loading input/bootstrap_result_summary_x_loading_comp1.csv \
  --y-loading input/bootstrap_result_summary_y_loading_comp1.csv \
  --freesurfer-labels input/FreeSurfer_label.csv \
  --analysis-desc input/analysis_results_description.txt \
  --introduction input/Introduction.txt \
  --methods input/Methods.txt \
  --generate-figures \
  --output-dir results

# 6. Run Fact Checking (Optional but recommended)
# Verifies citations in the generated discussion
# Results will be saved to results/verification_result_*.txt
echo "Starting Fact Checking Agent..."
# We pass the same output dir context if needed, but factchecking_agent.py 
# currently looks for the latest generated discussion automatically or runs generation again.
# To allow it to verify the JUST generated discussion, we run it directly.
python factchecking_agent.py

echo "All tasks completed. Check 'results' directory for outputs."
