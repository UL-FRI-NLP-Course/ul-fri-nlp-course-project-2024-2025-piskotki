import json
import numpy as np
import faiss
from query import search
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

### 0. — CONFIGURATION — ###

TOKEN_CHUNK_SIZE = 500     # max tokens per chunk
RAG_TOP_K = 5             # how many chunks to retrieve

prompt = "When is GTA VI coming out?"

# Extract just the long text fields
most_relevant = search(prompt, 1)[0]
print(most_relevant["title"])
raw_text = most_relevant["text"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16",
)

# 5. Load DeepSeek-LLM 7B
model_name = "deepseek-ai/deepseek-llm-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype="auto",
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=True, do_sample=True)

# Load model for embedding
# alternatives: msmarco-MiniLM-L6-cos-v5, multi-qa-MiniLM-L6-cos-v1
emb_model = SentenceTransformer("all-mpnet-base-v2")
emb_model = emb_model.to('cuda') # if you have a GPU, otherwise remove this line

### 3. — CHUNK EACH DOCUMENT INTO ≤500-TOKEN PIECES — ###
def chunk_text(text, max_tokens=TOKEN_CHUNK_SIZE):
    toks = tokenizer(text)["input_ids"]
    chunks = []
    for i in range(0, len(toks), max_tokens):
        chunk_ids = toks[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids))
    return chunks

chunked_passages = []
dimension = emb_model.get_sentence_embedding_dimension()  # MiniLM vector size
index = faiss.IndexFlatL2(dimension)
for chunk in chunk_text(raw_text):
    chunked_passages.append(chunk)
    embedding = emb_model.encode(chunk).astype('float32')
    index.add(np.expand_dims(embedding, axis=0))


### 5. — ITERATIVE-RAG FUNCTION — ###
def iterative_rag(query: str, top_k=RAG_TOP_K):
    # 5.1 embed the query (and make it 2D for FAISS)
    q_emb = emb_model.encode(query).astype('float32')
    q_emb = np.expand_dims(q_emb, axis=0)  # now shape is (1, dimension)

    # 5.2 retrieve top_k chunk IDs
    D, I = index.search(q_emb, len(chunked_passages))
    chosen = I[0]

    # 5.3 generate a partial answer for each chunk
    partials = []
    for idx in chosen:
        chunk = chunked_passages[idx]
        prompt = (
            f"Context:\n{chunk}\n\n"
            f"Question: {query}\nAnswer briefly:"
        )
        out = generator(prompt, max_new_tokens=128, do_sample=True)[0]
        ans = out["generated_text"]
        partials.append(ans)

    # 5.4 fuse partials into final
    combined = "\n".join(f"- {a}" for a in partials)
    fuse_prompt = (
        f"I have these partial answers:\n{combined}\n\n" 
        f"Please synthesize into one concise, coherent answer."
    )
    fused = generator(fuse_prompt, max_new_tokens=256, do_sample=True, return_full_text=True)[0]
    return fused["generated_text"]

print(iterative_rag(prompt))
