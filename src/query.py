import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

FAISS_INDEX_FILE = '../data/vector.index'
MAPPING_FILE = '../data/video_game_index_mapping.json'

# Load everything
index = faiss.read_index(FAISS_INDEX_FILE)
with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
    mapping = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

def search(query, top_k=5):
    query_vec = model.encode([query]).astype('float32')

    D, I = index.search(query_vec, top_k)  # D = distances, I = indices

    results = []
    for dist, idx in zip(D[0], I[0]):
        if str(idx) in mapping:
            results.append({
                'title': mapping[str(idx)]['title'],
                'url': mapping[str(idx)]['url'],
                'text': mapping[str(idx)]['text'],
                'distance': dist
            })
    return results

# Example
query = "who composed the score for the last of us part 2?"
results = search(query)

for r in results:
    print(f"Title: {r['title']}\nURL: {r['url']}\nDistance: {r['distance']:.4f}\n")
