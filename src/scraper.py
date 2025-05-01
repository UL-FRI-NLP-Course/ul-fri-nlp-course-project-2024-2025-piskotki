import requests
import json
import os
import signal
import re
from bs4 import BeautifulSoup, Tag
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Hardcoded file names
INPUT_FILE = 'data/upcoming_video_games.txt'
FAISS_INDEX_FILE = 'data/vector.index'
OUTPUT_FILE = 'data/video_game_index_mapping.json'

# Prepare session with desktop User-Agent
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/90.0.4430.212 Safari/537.36'
})

# Gracefully handle Ctrl+C
stop_signal = False

def handle_sigint(signum, frame):
    global stop_signal
    stop_signal = True

signal.signal(signal.SIGINT, handle_sigint)

# Load model for embedding
# alternatives: msmarco-MiniLM-L6-cos-v5, multi-qa-MiniLM-L6-cos-v1
model = SentenceTransformer("all-MiniLM-L6-v2")
model = model.to('cuda') # if you have a GPU, otherwise remove this line

# Setup FAISS index (initialize)
dimension = 384  # MiniLM vector size
if os.path.exists(FAISS_INDEX_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
else:
    index = faiss.IndexFlatL2(dimension)
    
# Load or initialize mapping
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
else:
    mapping = {}

def scrape_page(url):
    resp = session.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, 'html.parser')

    # Locate main content div
    content_div = soup.find('div', class_='mw-content-ltr mw-parser-output')
    description_texts = []
    if content_div:
        # Remove reference markers (superscripted numbers)
        for sup in content_div.find_all('sup', class_='reference'):
            sup.decompose()

        # find all <p> tags without class and id attributes
        for p in content_div.find_all('p', class_=False, id=False):
            # ensure spaces between text segments
            text = p.get_text(separator=' ', strip=True)
            # strip any leftover bracketed numbers like [1]
            text = re.sub(r"\[\d+\]", "", text)
            # remove unwanted spaces inside parentheses
            text = re.sub(r"\(\s+", "(", text)
            text = re.sub(r"\s+\)", ")", text)
            # collapse multiple spaces into one
            text = re.sub(r"\s{2,}", " ", text)
            # remove space before punctuation .,;:!?
            text = re.sub(r"\s+([\.,;:!\?])", r"\1", text)
            # collapse spaces around apostrophes
            text = re.sub(r"\s+'", "'", text)
            text = re.sub(r"'\s+", "'", text)
            if text:
                description_texts.append(text)
    
    full_text = " ".join(description_texts)
    title = soup.find('h1', id='firstHeading').get_text(strip=True)
    return title, full_text


def main():
    # Load URLs
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    completed_titles = set(mapping.values())

    for url in urls:
        if stop_signal:
            print('Interrupted; stopping early.')
            break
        try:
            title, text = scrape_page(url)
            if title in completed_titles:
                continue
            
            embedding = model.encode(text).astype('float32')

            # Add to FAISS
            index.add(np.expand_dims(embedding, axis=0))
            # Record mapping (store title, URL and text)
            mapping[str(index.ntotal - 1)] = {'title': title, 'url': url, 'text': text}

            # Save index and mapping
            faiss.write_index(index, FAISS_INDEX_FILE)
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
                json.dump(mapping, out, ensure_ascii=False, indent=4)

            print(f"Saved {title}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

if __name__ == '__main__':
    main()
