#!/bin/bash
#SBATCH --job-name pubmed_cca_gen
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
which python
python --version

# 3. Install dependencies (force install to be sure)
pip install -r requirements.txt

# 4. Run the Agent to generate results and discussion
# 
# OPTION 1: Use default input directory (recommended for standard setup)
# This will use all files from the input/ folder
python agent.py --mode generate

# OPTION 2: Specify individual CCA files (for custom datasets)
# Uncomment and modify the following to use specific files:
#
# python agent.py --mode generate \
#   --x-loading input/bootstrap_result_summary_x_loading_comp1.csv \
#   --y-loading input/bootstrap_result_summary_y_loading_comp1.csv \
#   --freesurfer-labels input/FreeSurfer_label.csv \
#   --analysis-desc input/analysis_results_description.txt \
#   --introduction input/Introduction.txt \
#   --methods input/Methods.txt \
#   --output-dir results
#
# Note: When using individual files, all 4 required files must be specified:
#   --x-loading, --y-loading, --freesurfer-labels, --analysis-desc
# Optional files: --introduction, --methods
