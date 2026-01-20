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
# This will use the input/ folder and save to results/ folder
python agent.py --mode generate
