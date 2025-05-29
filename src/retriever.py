from langchain.text_splitter import RecursiveCharacterTextSplitter
from wiki_scraper import WikiScraper
import faiss
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import requests

class Retriever:
    def __init__(self):
        self.wiki_scraper = WikiScraper()
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        if torch.cuda.is_available():
            self.embedder = self.embedder.to('cuda')

    def retrieve_wikipedia_links(self, query, num_results=5):
        base_url = 'http://207.154.241.192:8080/search'
        params = {
            'q': query + " site:wikipedia.org",
            'format': 'json',
            'pageno': 1,
            'categories': 'general',
            'language': 'en',
            'safesearch': 0,
            'engines': 'google,bing,duckduckgo,brave',
            'max_results': num_results
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'http://207.154.241.192:8080/',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
            return []
        
        results = response.json().get('results', [])
        
        wiki_links = []
        for result in results:
            url = result.get('url', '')
            if 'wikipedia.org' in url:
                wiki_links.append(url)
            if len(wiki_links) >= num_results:
                break
        
        return wiki_links

    def scrape_wikipedia_pages(self, wiki_links):
        results = []
        for link in wiki_links:
            try:
                title, text = self.wiki_scraper.scrape_page(link)
                results.append((title, text))
            except Exception as e:
                print(f"Error scraping {link}: {e}")
        return results
    
    def split_text_into_chunks(self, text, chunk_size=500, chunk_overlap=200):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = splitter.create_documents([text])
        return [chunk.page_content.lstrip(" .,!?\n") for chunk in chunks]
    
    def retrieve_and_process(self, query, num_results=2, chunk_size=500, chunk_overlap=200):
        wiki_links = self.retrieve_wikipedia_links(query, num_results)
        if not wiki_links:
            print("No Wikipedia links found.")
            return []

        scraped_pages = self.scrape_wikipedia_pages(wiki_links)
        if not scraped_pages:
            print("No pages successfully scraped.")
            return []

        processed_chunks = []
        for title, text in scraped_pages:
            chunks = self.split_text_into_chunks(text, chunk_size, chunk_overlap)
            for chunk in chunks:
                processed_chunks.append((title, chunk))

        if not processed_chunks:
            print("No text chunks generated.")
            return []

        # Embed chunks
        embeddings, metadata = self.embed_chunks(processed_chunks)
        if embeddings.shape[0] == 0:
            print("Embedding failed or returned empty array.")
            return []

        # Build FAISS index
        index = self.build_faiss_index(embeddings)

        # Search for the query
        hits = self.search_chunk(index, metadata, query)

        return hits
    
    def embed_chunks(self, chunks):
        texts = [chunk_text for _, chunk_text in chunks]
        embs = self.embedder.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        embs = np.array(embs, dtype='float32')
        return embs, chunks
    

    def build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def search_chunk(self, index, metadata, query, top_k=5):
        q_emb = self.embedder.encode([query], convert_to_tensor=False)
        q_emb = np.array(q_emb, dtype='float32')

        distances, indices = index.search(q_emb, top_k)

        hits = []
        for dist, idx in zip(distances[0], indices[0]):
            title, chunk_text = metadata[idx]
            hits.append((title, chunk_text, float(dist)))
        return hits
    
    def build_prompt(self, hits, query):
        prompt = [
            "You are an expert assistant. Use only the information provided below to answer the user’s question. Do not make up any facts; if the answer is not contained in the context, respond with “I don’t know.”",
            "",
            "Context:"
        ]
        for i, (title, chunk, _) in enumerate(hits, start=1):
            prompt.append(f"[Source {i}: {title}]")
            prompt.append(chunk)
            prompt.append("")  # blank line
        prompt.append("Question:")
        prompt.append(query)
        prompt.append("")
        prompt.append("Answer:")
        return "\n".join(prompt)


# query = "What will the new Mafia game be about?"
# retriever = Retriever()
# results = retriever.retrieve_and_process(query)
# for title, chunk_text, distance in results:
#     print(f"Title: {title}\nChunk: {chunk_text}\nDistance: {distance}\n{'='*50}")