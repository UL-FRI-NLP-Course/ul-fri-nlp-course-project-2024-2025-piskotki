import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math

class RAGEvaluator:
    def __init__(self, csv_path, length_penalty_factor=0.002):
        """Initialize the evaluator with the path to the CSV file.
        
        Args:
            csv_path: Path to the CSV file with questions and answers
            length_penalty_factor: Factor to penalize long answers (default: 0.002)
                                   Higher values = stronger penalty for length
        """
        self.csv_path = csv_path
        self.data = None
        self.metrics = None
        self.model = None
        self.length_penalty_factor = length_penalty_factor
        
        # Download required NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Initialize sentence transformer model for semantic similarity
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate_length_penalty(self, text):
        """Calculate length penalty for a given text.
        
        The penalty increases as text gets longer.
        Formula: 1 - min(0.5, length_penalty_factor * word_count)
        This ensures the penalty is at most 0.5 (50%) even for very long texts.
        """
        word_count = len(text.split())
        penalty = min(0.5, self.length_penalty_factor * word_count)
        return 1.0 - penalty
        
    def load_data(self):
        """Load data from the CSV file."""
        print(f"Loading data from {self.csv_path}...")
        self.data = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.data)} questions with answers.")
        return self
        
    def calculate_metrics(self):
        """Calculate evaluation metrics for each row."""
        print("Calculating metrics...")
        
        # Initialize columns for metrics
        metrics = {
            'bleu_rag_human': [],
            'bleu_chatgpt_human': [],
            'rouge1_rag_human': [],
            'rouge1_chatgpt_human': [],
            'semantic_sim_rag_human': [],
            'semantic_sim_chatgpt_human': [],
            'rag_length_penalty': [],
            'chatgpt_length_penalty': [],
            'rag_length_adjusted_score': [],
            'chatgpt_length_adjusted_score': [],
            'rag_word_count': [],
            'chatgpt_word_count': [],
            'rag_winner': []  # 1 if RAG wins, 0 if ChatGPT wins, 0.5 if tie
        }
        
        smoothie = SmoothingFunction().method1
        
        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            rag_answer = row['RAG_MODEL_ANSWER']
            chatgpt_answer = row['CHATGPT_ANSWER']
            human_answer = row['HUMAN_ANSWER']
            
            # Calculate word counts
            rag_word_count = len(rag_answer.split())
            chatgpt_word_count = len(chatgpt_answer.split())
            metrics['rag_word_count'].append(rag_word_count)
            metrics['chatgpt_word_count'].append(chatgpt_word_count)
            
            # Calculate length penalties
            rag_length_penalty = self.calculate_length_penalty(rag_answer)
            chatgpt_length_penalty = self.calculate_length_penalty(chatgpt_answer)
            metrics['rag_length_penalty'].append(rag_length_penalty)
            metrics['chatgpt_length_penalty'].append(chatgpt_length_penalty)
            
            # Calculate BLEU scores
            try:
                rag_bleu = sentence_bleu([human_answer.split()], rag_answer.split(), 
                                         smoothing_function=smoothie)
            except Exception:
                rag_bleu = 0
                
            try:
                chatgpt_bleu = sentence_bleu([human_answer.split()], chatgpt_answer.split(), 
                                            smoothing_function=smoothie)
            except Exception:
                chatgpt_bleu = 0
            
            metrics['bleu_rag_human'].append(rag_bleu)
            metrics['bleu_chatgpt_human'].append(chatgpt_bleu)
            
            # Calculate ROUGE scores
            rag_rouge = self.rouge_scorer.score(human_answer, rag_answer)
            chatgpt_rouge = self.rouge_scorer.score(human_answer, chatgpt_answer)
            
            metrics['rouge1_rag_human'].append(rag_rouge['rouge1'].fmeasure)
            metrics['rouge1_chatgpt_human'].append(chatgpt_rouge['rouge1'].fmeasure)
            
            # Calculate semantic similarity using sentence embeddings
            rag_embedding = self.model.encode([rag_answer])
            chatgpt_embedding = self.model.encode([chatgpt_answer])
            human_embedding = self.model.encode([human_answer])
            
            rag_sim = cosine_similarity(rag_embedding, human_embedding)[0][0]
            chatgpt_sim = cosine_similarity(chatgpt_embedding, human_embedding)[0][0]
            
            metrics['semantic_sim_rag_human'].append(rag_sim)
            metrics['semantic_sim_chatgpt_human'].append(chatgpt_sim)
            
            # Calculate length-adjusted scores (semantic similarity adjusted for length)
            rag_adjusted = rag_sim * rag_length_penalty
            chatgpt_adjusted = chatgpt_sim * chatgpt_length_penalty
            
            metrics['rag_length_adjusted_score'].append(rag_adjusted)
            metrics['chatgpt_length_adjusted_score'].append(chatgpt_adjusted)
            
            # Determine winner based on length-adjusted semantic similarity
            if rag_adjusted > chatgpt_adjusted:
                metrics['rag_winner'].append(1)
            elif chatgpt_adjusted > rag_adjusted:
                metrics['rag_winner'].append(0)
            else:
                metrics['rag_winner'].append(0.5)
        
        # Convert metrics to DataFrame
        self.metrics = pd.DataFrame(metrics)
        
        # Add metrics to original data
        self.data = pd.concat([self.data, self.metrics], axis=1)
        return self
    
    def generate_statistics(self):
        """Generate summary statistics for the evaluation."""
        if self.metrics is None:
            raise ValueError("Metrics have not been calculated yet. Call calculate_metrics() first.")
            
        stats = {
            'mean_bleu_rag': self.metrics['bleu_rag_human'].mean(),
            'mean_bleu_chatgpt': self.metrics['bleu_chatgpt_human'].mean(),
            'mean_rouge1_rag': self.metrics['rouge1_rag_human'].mean(),
            'mean_rouge1_chatgpt': self.metrics['rouge1_chatgpt_human'].mean(),
            'mean_semantic_sim_rag': self.metrics['semantic_sim_rag_human'].mean(),
            'mean_semantic_sim_chatgpt': self.metrics['semantic_sim_chatgpt_human'].mean(),
            'mean_rag_length_adjusted': self.metrics['rag_length_adjusted_score'].mean(),
            'mean_chatgpt_length_adjusted': self.metrics['chatgpt_length_adjusted_score'].mean(),
            'mean_rag_word_count': self.metrics['rag_word_count'].mean(),
            'mean_chatgpt_word_count': self.metrics['chatgpt_word_count'].mean(),
            'rag_win_percentage': self.metrics['rag_winner'].mean() * 100
        }
        
        self.stats = stats
        return self
    
    def save_results(self, output_path=None):
        """Save the results to a CSV file."""
        if output_path is None:
            output_path = self.csv_path.replace('.csv', '_evaluated.csv')
        
        print(f"Saving results to {output_path}...")
        self.data.to_csv(output_path, index=False)
        return self
    
    def plot_results(self, output_dir=None):
        """Generate and save visualization plots."""
        if output_dir is None:
            output_dir = '.'
            
        print("Generating plots...")
        
        # Plot 1: RAG vs ChatGPT semantic similarity
        plt.figure(figsize=(10, 6))
        data = pd.DataFrame({
            'RAG Model': self.metrics['semantic_sim_rag_human'],
            'ChatGPT': self.metrics['semantic_sim_chatgpt_human']
        })
        sns.boxplot(data=data)
        plt.title('Semantic Similarity to Human Answers')
        plt.ylabel('Cosine Similarity')
        plt.savefig(f"{output_dir}/semantic_similarity_comparison.png")
        
        # Plot 2: Length-adjusted similarity
        plt.figure(figsize=(10, 6))
        adjusted_data = pd.DataFrame({
            'RAG Model': self.metrics['rag_length_adjusted_score'],
            'ChatGPT': self.metrics['chatgpt_length_adjusted_score']
        })
        sns.boxplot(data=adjusted_data)
        plt.title('Length-Adjusted Semantic Similarity')
        plt.ylabel('Score')
        plt.savefig(f"{output_dir}/length_adjusted_comparison.png")
        
        # Plot 3: Win percentage
        plt.figure(figsize=(8, 8))
        win_counts = self.metrics['rag_winner'].value_counts()
        labels = ['ChatGPT wins', 'Tie', 'RAG wins']
        sizes = [
            (win_counts[0] if 0 in win_counts else 0),
            (win_counts[0.5] if 0.5 in win_counts else 0),
            (win_counts[1] if 1 in win_counts else 0)
        ]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Win Distribution: RAG vs ChatGPT')
        plt.savefig(f"{output_dir}/win_distribution.png")
        
        # Plot 4: Word count comparison
        plt.figure(figsize=(10, 6))
        word_count_data = pd.DataFrame({
            'RAG Model': self.metrics['rag_word_count'],
            'ChatGPT': self.metrics['chatgpt_word_count']
        })
        sns.boxplot(data=word_count_data)
        plt.title('Answer Length Comparison')
        plt.ylabel('Word Count')
        plt.savefig(f"{output_dir}/word_count_comparison.png")
        
        # Plot 5: Metric comparison
        plt.figure(figsize=(14, 7))
        metrics_data = pd.DataFrame({
            'BLEU': [self.stats['mean_bleu_rag'], self.stats['mean_bleu_chatgpt']],
            'ROUGE-1': [self.stats['mean_rouge1_rag'], self.stats['mean_rouge1_chatgpt']],
            'Semantic Similarity': [self.stats['mean_semantic_sim_rag'], self.stats['mean_semantic_sim_chatgpt']],
            'Length-Adjusted Score': [self.stats['mean_rag_length_adjusted'], self.stats['mean_chatgpt_length_adjusted']]
        }, index=['RAG Model', 'ChatGPT'])
        metrics_data.plot(kind='bar', figsize=(14, 7))
        plt.title('Average Metrics Comparison')
        plt.ylabel('Score')
        plt.savefig(f"{output_dir}/metrics_comparison.png")
        
        # Plot 6: Length vs. Similarity scatter plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(self.metrics['rag_word_count'], self.metrics['semantic_sim_rag_human'], alpha=0.7)
        plt.title('RAG: Length vs. Similarity')
        plt.xlabel('Word Count')
        plt.ylabel('Semantic Similarity')
        
        plt.subplot(1, 2, 2)
        plt.scatter(self.metrics['chatgpt_word_count'], self.metrics['semantic_sim_chatgpt_human'], alpha=0.7)
        plt.title('ChatGPT: Length vs. Similarity')
        plt.xlabel('Word Count')
        plt.ylabel('Semantic Similarity')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/length_vs_similarity.png")
        
        print(f"Plots saved to {output_dir}")
        return self
    
    def print_report(self):
        """Print a summary report of the evaluation."""
        if self.stats is None:
            raise ValueError("Statistics have not been generated yet. Call generate_statistics() first.")
            
        print("\n" + "="*60)
        print("RAG EVALUATION REPORT (WITH LENGTH PENALTY)")
        print("="*60)
        
        print("\nMetric Averages:")
        print(f"BLEU Score:                RAG: {self.stats['mean_bleu_rag']:.4f}  |  ChatGPT: {self.stats['mean_bleu_chatgpt']:.4f}")
        print(f"ROUGE-1 F1:                RAG: {self.stats['mean_rouge1_rag']:.4f}  |  ChatGPT: {self.stats['mean_rouge1_chatgpt']:.4f}")
        print(f"Semantic Similarity:       RAG: {self.stats['mean_semantic_sim_rag']:.4f}  |  ChatGPT: {self.stats['mean_semantic_sim_chatgpt']:.4f}")
        print(f"Length-Adjusted Score:     RAG: {self.stats['mean_rag_length_adjusted']:.4f}  |  ChatGPT: {self.stats['mean_chatgpt_length_adjusted']:.4f}")
        
        print("\nAnswer Length:")
        print(f"Average Word Count:        RAG: {self.stats['mean_rag_word_count']:.1f}  |  ChatGPT: {self.stats['mean_chatgpt_word_count']:.1f}")
        
        print("\nWin Analysis (based on length-adjusted scores):")
        rag_wins = (self.metrics['rag_winner'] == 1).sum()
        chatgpt_wins = (self.metrics['rag_winner'] == 0).sum()
        ties = (self.metrics['rag_winner'] == 0.5).sum()
        total = len(self.metrics)
        
        print(f"RAG wins:      {rag_wins} ({rag_wins/total*100:.1f}%)")
        print(f"ChatGPT wins:  {chatgpt_wins} ({chatgpt_wins/total*100:.1f}%)")
        print(f"Ties:          {ties} ({ties/total*100:.1f}%)")
        
        print("\nConclusion:")
        if self.stats['mean_rag_length_adjusted'] > self.stats['mean_chatgpt_length_adjusted']:
            print("The RAG model generally produces more concise and relevant answers compared to ChatGPT.")
        elif self.stats['mean_rag_length_adjusted'] < self.stats['mean_chatgpt_length_adjusted']:
            print("ChatGPT generally produces more concise and relevant answers compared to the RAG model.")
        else:
            print("Both models perform similarly in terms of providing concise and relevant answers.")
        
        if self.stats['mean_rag_word_count'] < self.stats['mean_chatgpt_word_count']:
            print("The RAG model produces shorter answers on average compared to ChatGPT.")
        elif self.stats['mean_rag_word_count'] > self.stats['mean_chatgpt_word_count']:
            print("ChatGPT produces shorter answers on average compared to the RAG model.")
        
        print("="*60 + "\n")
        return self
    
    def evaluate(self, output_path=None, output_dir=None):
        """Run the full evaluation pipeline."""
        return (self
                .load_data()
                .calculate_metrics()
                .generate_statistics()
                .save_results(output_path)
                .plot_results(output_dir)
                .print_report())

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RAG model performance')
    parser.add_argument('csv_file', help='Path to the CSV file with answers')
    parser.add_argument('--output', help='Path to save the evaluated CSV file')
    parser.add_argument('--plots-dir', help='Directory to save plots')
    parser.add_argument('--length-penalty', type=float, default=0.002, 
                        help='Factor to penalize long answers (default: 0.002)')
    args = parser.parse_args()
    print("Starting evaluation...")
    
    evaluator = RAGEvaluator(args.csv_file, length_penalty_factor=args.length_penalty)
    evaluator.evaluate(args.output, args.plots_dir)