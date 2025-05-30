import argparse
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import scipy.spatial.distance as distance

def read_file_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def calculate_semantic_similarity(model, reference, candidate):
    ref_embedding = model.encode(reference)
    cand_embedding = model.encode(candidate)
    return 1 - distance.cosine(ref_embedding, cand_embedding)

def main():
    timestamp_ms = int(time.time() * 1000)
    parser = argparse.ArgumentParser(description='Compare model answers to ChatGPT answers using semantic similarity')
    parser.add_argument('--questions', required=True, help='Path to questions file')
    parser.add_argument('--chatgpt', required=True, help='Path to ChatGPT answers file (ground truth)')
    parser.add_argument('--model', required=True, help='Path to your model answers file')
    parser.add_argument('--output', default='semantic_results.csv', help='Output file for comparison results')
    args = parser.parse_args()

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

    if not (len(questions) == len(chatgpt_answers) == len(model_answers)):
        print("Error: All files must have the same number of lines")
        return

    print("Loading sentence transformer model...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    results = []

    print(f"Comparing {len(questions)} answer pairs...")
    for i, (question, chatgpt_answer, model_answer) in enumerate(zip(questions, chatgpt_answers, model_answers)):
        semantic_similarity = calculate_semantic_similarity(semantic_model, chatgpt_answer, model_answer)

        results.append({
            'Question': question,
            'ChatGPT Answer': chatgpt_answer,
            'Model Answer': model_answer,
            'Semantic Similarity': semantic_similarity
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(questions)} questions")

    df = pd.DataFrame(results)
    df.to_csv(timestamped_output, index=False)

    avg_semantic = df['Semantic Similarity'].mean()
    print("\nSummary:")
    print(f"Total questions: {len(questions)}")
    print(f"Average Semantic Similarity: {avg_semantic:.4f}")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['Semantic Similarity'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Semantic Similarity')
    plt.ylabel('Number of Answers')
    plt.title('Distribution of Semantic Similarity Scores')
    plt.savefig(f'semantic_score_distribution_{timestamp_ms}.png')

    print(f"Results saved to {args.output} and semantic_score_distribution_{timestamp_ms}.png")

if __name__ == "__main__":
    main()
