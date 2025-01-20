#!/bin/sh


#SBATCH --job-name=standard(gpu)
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --mem=8000


nvidia-smi

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

source ltpEnvironment/bin/activate

PYTHON_SCRIPT="./main.py"

python3 "$PYTHON_SCRIPT"