#!/bin/bash
#SBATCH --job-name=nlp_project
#SBATCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# Default question if none provided
QUERY="${1:-what kind of model are you?}"

singularity run --nv -B /d/hpc/home/ks4681/ul-fri-nlp-course-project-2024-2025-piskotki:/workspace nlp.sif python /workspace/src/main.py "$QUERY"