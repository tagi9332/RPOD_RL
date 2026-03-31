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

# Clear out any loaded modules
module purge

# Load Git
module load git

# Load the EXACT version of UV the cluster told you it has
module load uv/0.8.15

# Load the standard Python (Alpine usually uses 'python' or 'anaconda')
module load python

# --- Git Repository Setup ---
REPO_URL="https://github.com/tagi9332/RPOD_RL.git"
REPO_DIR="RPOD_RL"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository for the first time..."
    git clone $REPO_URL
else
    echo "Repository already exists. Pulling the latest changes..."
    cd $REPO_DIR
    git pull
    cd ..
fi

cd $REPO_DIR

# --- uv Environment Setup ---
# Create a virtual environment (.venv) if one doesn't already exist
if [ ! -d ".venv" ]; then
    echo "Creating new uv virtual environment..."
    uv venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install or update dependencies (adjust this if you use pyproject.toml instead)
echo "Syncing dependencies..."
uv pip install -r requirements.txt
# Note: If you use a pyproject.toml, you can replace the above line with `uv sync`

# --- Execute Training ---
echo "Starting training..."
python src/train/docking_sim_multi_process.py