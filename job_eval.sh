#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --output=logs/piskotki-project-%J.out
#SBATCH --error=logs/piskotki-project-%J.err
#SBATCH --job-name="Piškotki NLP project"

srun singularity exec --nv piskotki.sif python \
  ./src/rag_evaluator.py --questions ./data/testing_questions.txt --chatgpt ./data/chatgpt_answers.txt --model ./data/model_answers.txt