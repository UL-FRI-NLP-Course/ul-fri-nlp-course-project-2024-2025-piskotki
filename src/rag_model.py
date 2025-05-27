import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch
from retriever import Retriever

class RAGSystem:
    def __init__(self):
        self.tokenizer = None
        self.generator = None
        self.emb_model = None
        self.index = None
        self.chunked_passages = []
        
    def initialize_models(self):
        """Initialize all models and components"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="float16",
        )

        model_name = "deepseek-ai/deepseek-llm-7b-chat"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype="auto",
        )

        self.generator = pipeline("text-generation", model=model, 
                                tokenizer=self.tokenizer, 
                                return_full_text=False)
        
        self.emb_model = SentenceTransformer("all-mpnet-base-v2")
        if torch.cuda.is_available():
            self.emb_model = self.emb_model.to('cuda')

    def process_document(self, raw_text):
        """Process document into chunks and build FAISS index"""
        self.chunked_passages = []
        dimension = self.emb_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dimension)
        
        for chunk in self._chunk_text(raw_text):
            self.chunked_passages.append(chunk)
            embedding = self.emb_model.encode(chunk).astype('float32')
            self.index.add(np.expand_dims(embedding, axis=0))

    def _chunk_text(self, text, max_tokens=500):
        """Split text into chunks"""
        toks = self.tokenizer(text)["input_ids"]
        return [self.tokenizer.decode(toks[i:i+max_tokens]) 
                for i in range(0, len(toks), max_tokens)]

    def generate_response(self, query, top_k=5):
        """Run RAG pipeline for a single query"""
        q_emb = self.emb_model.encode(query).astype('float32')
        q_emb = np.expand_dims(q_emb, axis=0)

        D, I = self.index.search(q_emb, len(self.chunked_passages))
        chosen = I[0]

        partials = []
        for idx in chosen:
            chunk = self.chunked_passages[idx]
            prompt = (
                f"Context:\n{chunk}\n\n"
                f"Question: {query}\nAnswer briefly:"
            )
            out = self.generator(prompt, max_new_tokens=128, do_sample=True)[0]
            ans = out["generated_text"]
            partials.append(ans)

        combined = "\n".join(f"- {a}" for a in partials)
        fuse_prompt = (
            f"I have these partial answers:\n{combined}\n\n" 
            f"Please synthesize into one concise, coherent answer.\n\n Answer: "
        )
        fused = self.generator(fuse_prompt, max_new_tokens=256, do_sample=True)[0]
        return fused["generated_text"]
    
    def generate_response_with_retriever(self, query, top_k=5):
        """Use Retriever to get relevant chunks and generate response"""
        retriever = Retriever()
        hits = retriever.retrieve_and_process(query)
        
        prompt = retriever.build_prompt(hits, query)
        out = self.generator(prompt, max_new_tokens=256, do_sample=True)[0]
        return out["generated_text"]
