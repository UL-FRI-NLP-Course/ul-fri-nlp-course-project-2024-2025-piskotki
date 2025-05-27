# src/main.py
#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from rag_model import RAGSystem
from query import search  # Your existing search function

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("1. Single query: python src/main.py \"your question\"")
        print("2. Batch from file: python src/main.py input.txt")
        print("3. Batch with output: python src/main.py input.txt output.txt")
        sys.exit(1)

    # Initialize RAG system
    rag = RAGSystem()
    rag.initialize_models()
    
    # Handle different input modes
    if len(sys.argv) == 2:
        if os.path.exists(sys.argv[1]):
            # Mode 2: Input file to console
            with open(sys.argv[1], 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            for query in queries:
                # most_relevant = search(query, 1)[0]
                # rag.process_document(most_relevant["text"])
                # result = rag.generate_response(query)
                # print(f"Q: {query}\nA: {result}\n{'='*50}")
                result = rag.generate_response_with_retriever(query, top_k=5)
        else:
            # Mode 1: Single query
            query = sys.argv[1] if not os.path.exists(sys.argv[1]) else "sample query"
            # most_relevant = search(query, 1)[0]
            # rag.process_document(most_relevant["text"])
            # result = rag.generate_response(query)
            result = rag.generate_response_with_retriever(query, top_k=5)
            print(result)
    elif len(sys.argv) == 3:
        # Mode 3: Input file to output file
        with open(sys.argv[1], 'r', encoding='utf-8') as f_in, open(sys.argv[2], 'w', encoding='utf-8') as f_out:
            queries = [line.strip() for line in f_in if line.strip()]
            for query in queries:
                # most_relevant = search(query, 1)[0]
                # rag.process_document(most_relevant["text"])
                # result = rag.generate_response(query)
                result = rag.generate_response_with_retriever(query, top_k=5)
                f_out.write(f"Q: {query}\nA: {result}\n{'='*50}\n\n")

if __name__ == "__main__":
    main()