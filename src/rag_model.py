import numpy as np
import faiss
from query import search
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch

### --- CONFIGURATION --- ###
# Constants for text processing and RAG
TOKEN_CHUNK_SIZE = 500     # Maximum tokens per text chunk
RAG_TOP_K = 1              # Number of relevant chunks to retrieve for RAG

# Example query for demonstration
prompt = "When is GTA VI coming out?"

# Retrieve the most relevant document for the query
most_relevant = search(prompt, 1)[0]
print(f"Retrieved document: {most_relevant['title']}")
raw_text = most_relevant["text"]

# Configuration for 4-bit quantization to optimize memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16",
)

### --- MODEL INITIALIZATION --- ###
# Load the DeepSeek-LLM 7B model with quantization
model_name = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype="auto",
)

# Initialize text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# Load embedding model (all-mpnet-base-v2) and move to GPU if available
emb_model = SentenceTransformer("all-mpnet-base-v2")
emb_model = emb_model.to('cuda') if torch.cuda.is_available() else emb_model

### --- TEXT PROCESSING FUNCTIONS --- ###
def chunk_text(text, max_tokens=TOKEN_CHUNK_SIZE):
    """Split input text into chunks of specified maximum token length.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    toks = tokenizer(text)["input_ids"]
    return [tokenizer.decode(toks[i:i+max_tokens]) 
            for i in range(0, len(toks), max_tokens)]

# Initialize FAISS index for vector similarity search
dimension = 768  # Embedding dimension of the model
index = faiss.IndexFlatL2(dimension)
chunked_passages = []

# Process each chunk: store text and add embedding to FAISS index
for chunk in chunk_text(raw_text):
    chunked_passages.append(chunk)
    embedding = emb_model.encode(chunk).astype('float32')
    index.add(np.expand_dims(embedding, axis=0))

### --- RAG IMPLEMENTATION --- ###
def iterative_rag(query: str, top_k=RAG_TOP_K):
    """Generate an answer using Retrieval-Augmented Generation.
    
    1. Retrieves relevant text chunks
    2. Generates partial answers for each chunk
    3. Combines partial answers into final response
    
    Args:
        query: User question
        top_k: Number of chunks to retrieve
        
    Returns:
        Generated answer string
    """
    # Embed the query and prepare for FAISS search
    q_emb = emb_model.encode(query).astype('float32')
    q_emb = np.expand_dims(q_emb, axis=0)  # Shape: (1, dimension)

    # Retrieve top_k most relevant chunks
    distances, indices = index.search(q_emb, top_k)
    selected_chunk_indices = indices[0]

    # Generate partial answers for each relevant chunk
    partial_answers = []
    for idx in selected_chunk_indices:
        context = chunked_passages[idx]
        prompt_template = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer briefly:"
        )
        response = generator(prompt_template, max_new_tokens=128, do_sample=False)[0]
        answer = response["generated_text"].split("Answer")[-1].strip(" :\n")
        partial_answers.append(answer)

    # Combine partial answers into coherent final response
    combined_answers = "\n".join(f"- {ans}" for ans in partial_answers)
    fusion_prompt = (
        f"I have these partial answers:\n{combined_answers}\n\n"
        "Please synthesize into one concise, coherent answer."
    )
    final_response = generator(fusion_prompt, max_new_tokens=256, do_sample=False)[0]
    return final_response["generated_text"]

# Execute RAG pipeline and print results
print(iterative_rag(prompt))
