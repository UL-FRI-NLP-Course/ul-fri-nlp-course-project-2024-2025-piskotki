from langchain.text_splitter import RecursiveCharacterTextSplitter
from wiki_scraper import WikiScraper
import faiss
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import requests

class Retriever:
    def __init__(self):
        # Initialize the Wikipedia scraper and sentence embedding model
        self.wiki_scraper = WikiScraper()
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        
        # Move embedder to GPU if available
        if torch.cuda.is_available():
            self.embedder = self.embedder.to('cuda')

    def retrieve_wikipedia_links(self, query, num_results=5):
        """
        Retrieves Wikipedia links related to the given query using a searxNG search engine.
        """
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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'http://207.154.241.192:8080/',
            'X-Requested-With': 'XMLHttpRequest'
        }

        # Make the request and handle errors
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
            return []

        # Extract Wikipedia links
        results = response.json().get('results', [])
        wiki_links = [res['url'] for res in results if 'wikipedia.org' in res.get('url', '')]
        
        return wiki_links[:num_results]

    def scrape_wikipedia_pages(self, wiki_links):
        """
        Scrapes Wikipedia pages and returns a list of (title, text) tuples.
        """
        results = []
        for link in wiki_links:
            try:
                title, text = self.wiki_scraper.scrape_page(link)
                results.append((title, text))
            except Exception as e:
                print(f"Error scraping {link}: {e}")
        return results

    def split_text_into_chunks(self, text, chunk_size=500, chunk_overlap=200):
        """
        Splits long text into manageable overlapping chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = splitter.create_documents([text])
        return [chunk.page_content.lstrip(" .,!?\n") for chunk in chunks]

    def retrieve_and_process(self, query, num_results=2, chunk_size=500, chunk_overlap=200):
        """
        Full retrieval and processing pipeline: fetch links, scrape text, chunk, embed, search.
        """
        # Step 1: Retrieve Wikipedia links
        wiki_links = self.retrieve_wikipedia_links(query, num_results)
        if not wiki_links:
            print("No Wikipedia links found.")
            return []

        # Step 2: Scrape pages
        scraped_pages = self.scrape_wikipedia_pages(wiki_links)
        if not scraped_pages:
            print("No pages successfully scraped.")
            return []

        # Step 3: Split scraped text into chunks
        processed_chunks = []
        for title, text in scraped_pages:
            chunks = self.split_text_into_chunks(text, chunk_size, chunk_overlap)
            for chunk in chunks:
                processed_chunks.append((title, chunk))

        if not processed_chunks:
            print("No text chunks generated.")
            return []

        # Step 4: Embed text chunks
        embeddings, metadata = self.embed_chunks(processed_chunks)
        if embeddings.shape[0] == 0:
            print("Embedding failed or returned empty array.")
            return []

        # Step 5: Create FAISS index and search
        index = self.build_faiss_index(embeddings)
        hits = self.search_chunk(index, metadata, query)

        return hits

    def embed_chunks(self, chunks):
        """
        Converts list of (title, chunk_text) into sentence embeddings.
        """
        texts = [chunk_text for _, chunk_text in chunks]
        embs = self.embedder.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        return np.array(embs, dtype='float32'), chunks

    def build_faiss_index(self, embeddings):
        """
        Builds a FAISS index from the embeddings for similarity search.
        """
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def search_chunk(self, index, metadata, query, top_k=5):
        """
        Searches for top_k most similar chunks to the query using FAISS index.
        """
        q_emb = self.embedder.encode([query], convert_to_tensor=False)
        q_emb = np.array(q_emb, dtype='float32')
        distances, indices = index.search(q_emb, top_k)

        hits = []
        for dist, idx in zip(distances[0], indices[0]):
            title, chunk_text = metadata[idx]
            hits.append((title, chunk_text, float(dist)))
        return hits

    def build_prompt(self, hits, query):
        """
        Builds a final prompt for the language model from search hits and a user query.
        """
        prompt = [
            "You are an expert assistant. Use only the information provided below to answer the user’s question. Do not make up any facts; if the answer is not contained in the context, respond with “I don’t know.”",
            "",
            "Context:"
        ]
        for i, (title, chunk, _) in enumerate(hits, start=1):
            prompt.append(f"[Source {i}: {title}]")
            prompt.append(chunk)
            prompt.append("")  # Blank line

        prompt.append("Question:")
        prompt.append(query)
        prompt.append("")
        prompt.append("Answer:")
        
        return "\n".join(prompt)
