#!/bin/bash

#SBATCH --job-name=rpo-rl-training
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=32G
#SBATCH --output=logs/training-job.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tagi9332@colorado.edu

# 1. Environment Setup
module purge
module load git
module load uv/0.8.15

# 2. Navigate to the Project Root
# Since you submit from 'RPOD_RL/slurm', we move up one level to 'RPOD_RL'
cd ..

# 3. Create/Activate Python 3.12 Environment
# This solves your 'contourpy' and 'Python version' errors
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.12..."
    uv venv --python 3.12
fi
source .venv/bin/activate

# 4. Install Dependencies
echo "Installing requirements..."
uv pip install -r requirements.txt

# 5. Execute Training
echo "Starting training..."
# Path is relative to project root
python src/train/docking_sim_multi_process.py