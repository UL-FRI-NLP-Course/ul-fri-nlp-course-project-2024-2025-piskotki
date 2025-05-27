import requests
import json
import os
import signal
import re
from bs4 import BeautifulSoup, Tag
from sentence_transformers import SentenceTransformer
import numpy as np

class WikiScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/90.0.4430.212 Safari/537.36'
        })

    def scrape_page(self, url):
        resp = self.session.get(url)
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