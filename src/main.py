import sys
import os
import time
from pathlib import Path
from rag_model import RAGSystem

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("1. Single query: python src/main.py \"your question\"")
        print("2. Batch from file: python src/main.py input.txt")
        print("3. Batch with output: python src/main.py input.txt output.txt")
        sys.exit(1)

    # Initialize the Retrieval-Augmented Generation (RAG) system
    rag = RAGSystem()
    rag.initialize_models()

    # -------- Mode 1 or 2: Single query or input file --------
    if len(sys.argv) == 2:
        input_arg = sys.argv[1]
        if os.path.exists(input_arg):
            # Mode 2: Input file (one query per line), print results to console
            with open(input_arg, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            for query in queries:
                result = rag.generate_response_with_retriever(query, top_k=5)
                print(f"Q: {query}\nA: {result}\n{'=' * 50}\n")
        else:
            # Mode 1: Single query passed directly as argument
            query = input_arg
            result = rag.generate_response_with_retriever(query, top_k=5)
            print(result)

    # -------- Mode 3: Batch input with output file --------
    elif len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        timestamp = int(time.time() * 1000)
        model_answers_filename = f"model_answers_{timestamp}.txt"

        # Read input queries and prepare output files
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out, \
             open(model_answers_filename, 'w', encoding='utf-8') as f_model:

            queries = [line.strip() for line in f_in if line.strip()]
            for query in queries:
                result = rag.generate_response_with_retriever(query, top_k=5)

                # Output full Q&A for user readability
                f_out.write(f"Q: {query}\nA: {result}\n{'=' * 50}\n\n")
                
                # Output model answers only (for metrics or evaluation)
                f_model.write(f"{result}\n")

if __name__ == "__main__":
    main()
