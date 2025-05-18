#!/bin/bash
#SBATCH --job-name=nlp_gpu
#SBATCH --output=nlp_output_%j.log
#SBATCH --error=nlp_error_%j.log
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu    # Use your system's GPU partition name
#SBATCH --gres=gpu:1       # Request 1 GPU

# Go to project directory
cd /d/hpc/home/ks4681/ul-fri-nlp-course-project-2024-2025-piskotki

# Check if container exists, build if needed
if [ ! -f nlp.sif ]; then
    echo "Building Singularity container..."
    singularity build nlp.sif nlp.def
fi

# Load any required modules (uncomment if needed on your system)
# module load cuda nvidia-container-runtime

# Run the script with GPU support
echo "Running main.py with GPU support..."
singularity exec --nv nlp.sif python src/main.py "output your query, please"

echo "Job completed!"