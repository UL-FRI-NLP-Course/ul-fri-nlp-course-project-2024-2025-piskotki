#!/usr/bin/env python3
# filepath: answer_comparator.py

import argparse
import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sentence_transformers import SentenceTransformer
import scipy.spatial.distance as distance
import pandas as pd
import matplotlib.pyplot as plt
import time

def read_file_lines(file_path):
    """Read all lines from a file, stripping whitespace."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def calculate_length_penalty(model_text, reference_text, base_penalty=0.01):
    """Calculate length penalty for verbose answers compared to reference.
    Rewards answers that are more concise than reference, penalizes longer ones."""
    model_words = len(model_text.split())
    ref_words = len(reference_text.split())
    
    # If model answer is shorter than reference, give bonus
    if model_words <= ref_words:
        return 1.0 + (base_penalty * (ref_words - model_words) / ref_words)
    # If model answer is longer, give penalty
    else:
        return 1.0 - (base_penalty * (model_words - ref_words) / ref_words)

def calculate_bleu(reference, candidate):
    """Calculate BLEU score with smoothing to handle zero n-gram matches."""
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    
    # Create a smoothing function instance
    smoothie = SmoothingFunction().method1  # You can try different methods (1-7)
    
    try:
        # Use a lower n-gram order and apply smoothing
        return sentence_bleu(
            reference_tokens, 
            candidate_tokens,
            weights=(0.5, 0.5),  # Just use 1-grams and 2-grams
            smoothing_function=smoothie
        )
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return 0.0

def calculate_rouge(reference, candidate):
    """Calculate ROUGE scores between reference and candidate texts."""
    rouge = Rouge()
    try:
        scores = rouge.get_scores(candidate, reference)[0]
        return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']
    except ValueError:
        # Handle very short texts
        return 0, 0, 0

def calculate_semantic_similarity(model, reference, candidate):
    """Calculate semantic similarity using sentence embeddings."""
    ref_embedding = model.encode(reference)
    cand_embedding = model.encode(candidate)
    return 1 - distance.cosine(ref_embedding, cand_embedding)

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def normalized_levenshtein(s1, s2):
    """
    Calculate the normalized Levenshtein distance between two strings.
    Returns a value between 0 (completely different) and 1 (identical).
    """
    if not s1 and not s2:
        return 1.0  # Both strings are empty
    
    # Calculate raw Levenshtein distance
    lev_dist = levenshtein_distance(s1, s2)
    
    # Normalize by the length of the longer string
    max_len = max(len(s1), len(s2))
    
    # Return similarity score (1 - normalized distance)
    return 1.0 - (lev_dist / max_len)

def main():
    timestamp_ms = int(time.time() * 1000)
    parser = argparse.ArgumentParser(description='Compare model answers to ChatGPT answers with preference for conciseness')
    parser.add_argument('--questions', required=True, help='Path to questions file')
    parser.add_argument('--chatgpt', required=True, help='Path to ChatGPT answers file (ground truth)')
    parser.add_argument('--model', required=True, help='Path to your model answers file')
    parser.add_argument('--output', default='comparison_results.csv', help='Output file for comparison results')
    args = parser.parse_args()

     # Extract base name and extension from output path
    base_name, extension = os.path.splitext(args.output)
    timestamped_output = f"{base_name}_{timestamp_ms}{extension}"
    args.output = timestamped_output
    
    # Check if files exist
    for file_path in [args.questions, args.chatgpt, args.model]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return
    
    # Read files
    questions = read_file_lines(args.questions)
    chatgpt_answers = read_file_lines(args.chatgpt)
    model_answers = read_file_lines(args.model)
    
    # Ensure all files have the same number of lines
    if not (len(questions) == len(chatgpt_answers) == len(model_answers)):
        print("Error: All files must have the same number of lines")
        print(f"Questions: {len(questions)}, ChatGPT: {len(chatgpt_answers)}, Model: {len(model_answers)}")
        return
    
    # Load semantic similarity model
    print("Loading sentence transformer model...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare results
    results = []
    
    print(f"Comparing {len(questions)} answer pairs...")
    for i, (question, chatgpt_answer, model_answer) in enumerate(zip(questions, chatgpt_answers, model_answers)):
        # Calculate metrics (using ChatGPT as reference)
        bleu_score = calculate_bleu(chatgpt_answer, model_answer)
        rouge1, rouge2, rougeL = calculate_rouge(chatgpt_answer, model_answer)
        semantic_similarity = calculate_semantic_similarity(semantic_model, chatgpt_answer, model_answer)
        levenshtein_sim = normalized_levenshtein(chatgpt_answer, model_answer)

        # Calculate length penalty
        length_penalty = calculate_length_penalty(model_answer, chatgpt_answer)
        
        # Calculate overall score (with length penalty applied)
        overall_score = (0.25 * semantic_similarity + 0.25 * rougeL + 
                        0.15 * bleu_score + 0.15 * rouge1 + 
                        0.2 * levenshtein_sim) * length_penalty        
        # Store results
        results.append({
            'Question': question,
            'ChatGPT Answer': chatgpt_answer,
            'Model Answer': model_answer,
            'ChatGPT Length': len(chatgpt_answer.split()),
            'Model Length': len(model_answer.split()),
            'BLEU Score': bleu_score,
            'ROUGE-1': rouge1,
            'ROUGE-2': rouge2,
            'ROUGE-L': rougeL,
            'Semantic Similarity': semantic_similarity,
            'Levenshtein Similarity': levenshtein_sim,
            'Length Penalty': length_penalty,
            'Overall Score': overall_score
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(questions)} questions")
    
    # Convert to DataFrame and save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)


    
    # Calculate summary statistics
    avg_bleu = np.mean([r['BLEU Score'] for r in results])
    avg_rouge1 = np.mean([r['ROUGE-1'] for r in results])
    avg_rouge2 = np.mean([r['ROUGE-2'] for r in results])
    avg_rougeL = np.mean([r['ROUGE-L'] for r in results])
    avg_semantic = np.mean([r['Semantic Similarity'] for r in results])
    avg_levenshtein = np.mean([r['Levenshtein Similarity'] for r in results])
    avg_length_penalty = np.mean([r['Length Penalty'] for r in results])
    avg_overall = np.mean([r['Overall Score'] for r in results])
    
    print("\nSummary Statistics:")
    print(f"Total questions: {len(questions)}")
    print(f"Average BLEU score: {avg_bleu:.4f}")
    print(f"Average ROUGE-1 score: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2 score: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L score: {avg_rougeL:.4f}")
    print(f"Average Semantic Similarity: {avg_semantic:.4f}")
    print(f"Average Levenshtein Similarity: {avg_levenshtein:.4f}")
    print(f"Average Length Penalty: {avg_length_penalty:.4f}")
    print(f"Average Overall Score: {avg_overall:.4f}")
    # Create and save summary plots
    plt.figure(figsize=(10, 6))
    
    metrics = ['BLEU', 'ROUGE-1', 'ROUGE-L', 'Semantic', 'Levenshtein', 'Length Bonus', 'Overall']
    values = [avg_bleu, avg_rouge1, avg_rougeL, avg_semantic, avg_levenshtein, avg_length_penalty, avg_overall]    
    plt.bar(metrics, values, color='skyblue')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Relative to ChatGPT Answers')
    plt.ylim(0, 1.1)  # Set y-axis limit
    
    plt.tight_layout()
    plt.savefig(f'metrics_summary{timestamp_ms}.png')
    
    # Generate histogram of overall scores
    plt.figure(figsize=(10, 6))
    plt.hist([r['Overall Score'] for r in results], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Overall Score')
    plt.ylabel('Number of Answers')
    plt.title('Distribution of Overall Answer Scores')
    plt.savefig(f'score_distribution_{timestamp_ms}.png')
    
    print(f"Results saved to {args.output}, metrics_summary.png, and score_distribution.png")

if __name__ == "__main__":
    main()