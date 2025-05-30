import numpy as np
import faiss
import torch
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from retriever import Retriever

class RAGSystem:
    def __init__(self):
        self.tokenizer = None
        self.generator = None
        self.emb_model = None
        self.index = None
        self.chunked_passages = []

    def initialize_models(self):
        model_name = "deepseek-ai/deepseek-llm-7b-chat"

        # Load tokenizer and model with device optimization
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Initialize text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            return_full_text=False
        )

        # Initialize sentence embedding model
        self.emb_model = SentenceTransformer("all-mpnet-base-v2")
        if torch.cuda.is_available():
            self.emb_model = self.emb_model.to('cuda')

    def generate_response_with_retriever(self, query, top_k=5):
        retriever = Retriever()
        
        # Step 1: Retrieve relevant Wikipedia chunks
        hits = retriever.retrieve_and_process(query, num_results=top_k, chunk_size=1024)
        
        # Step 2: Build prompt from retrieved content
        prompt = retriever.build_prompt(hits, query)

        # Step 3: Generate a response using the language model
        output = self.generator(prompt, max_new_tokens=256, do_sample=True)[0]
        
        return output["generated_text"].replace('\n', '')
