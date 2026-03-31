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

# 3. Destroy any local/broken .venv to force uv to use Scratch
rm -rf .venv

# 4. Set up the Scratch Environment
# Using alpine1 explicitly to match your curc-quota 10TB allocation
mkdir -p "/scratch/alpine1/$USER"
export VENV_DIR="/scratch/alpine/$USER/rpod_env"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment on SCRATCH at $VENV_DIR..."
    uv venv "$VENV_DIR" --python 3.12
fi

# Activate the scratch environment
source "$VENV_DIR/bin/activate"

# Verify the environment in the logs
echo "VIRTUAL_ENV is set to: $VIRTUAL_ENV"
echo "Using Python from: $(which python)"

# 5. Install Dependencies
echo "Installing requirements..."
uv pip install -r requirements_linux.txt

# 6. Execute Training
echo "Starting training..."

# Add the current directory (RPOD_RL) to the Python Search Path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Use -u to see logs in real-time (no more waiting!)
python -u src/train/docking_sim_multi_process.py