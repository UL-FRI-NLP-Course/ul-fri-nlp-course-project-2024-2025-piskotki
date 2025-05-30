#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclude=wn208,wn210,wn220,wn221,wn224
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/piskotki-project-%J.out
#SBATCH --error=logs/piskotki-project-%J.err
#SBATCH --job-name="Piškotki NLP project"

# srun singularity exec --nv ./containers/container-rag-piskotki.sif python \
#   ul-fri-nlp-course-project-2024-2025-piskotki/src/main.py ul-fri-nlp-course-project-2024-2025-piskotki/data/testing_questions.txt ul-fri-nlp-course-project-2024-2025-piskotki/data/model_answers.txt

srun singularity exec --nv ./containers/container-rag-piskotki.sif python \
  ul-fri-nlp-course-project-2024-2025-piskotki/src/main.py "Which game engine was Clair Obscur: Expedition 33 developed with?"