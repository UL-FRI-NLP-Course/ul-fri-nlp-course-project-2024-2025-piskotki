import requests
import re
from bs4 import BeautifulSoup

class WikiScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/90.0.4430.212 Safari/537.36'
        })

    def scrape_page(self, url):
        # Fetch the page
        resp = self.session.get(url)
        resp.raise_for_status()

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(resp.content, 'html.parser')

        # Locate the main content div
        content_div = soup.find('div', class_='mw-content-ltr mw-parser-output')
        description_texts = []

        if content_div:
            # Remove reference footnotes like [1], [2], etc.
            for sup in content_div.find_all('sup', class_='reference'):
                sup.decompose()

            # Extract all paragraph tags without extra classes or IDs
            for p in content_div.find_all('p', class_=False, id=False):
                text = p.get_text(separator=' ', strip=True)

                # Remove square bracket references (e.g. [1])
                text = re.sub(r"\[\d+\]", "", text)

                # Clean up spacing and punctuation formatting
                text = re.sub(r"\(\s+", "(", text)              # No space after '('
                text = re.sub(r"\s+\)", ")", text)              # No space before ')'
                text = re.sub(r"\s{2,}", " ", text)             # Collapse multiple spaces
                text = re.sub(r"\s+([\.,;:!\?])", r"\1", text)  # No space before punctuation
                text = re.sub(r"\s+'", "'", text)               # No space before apostrophe
                text = re.sub(r"'\s+", "'", text)               # No space after apostrophe

                if text:
                    description_texts.append(text)

        # Combine all cleaned paragraph texts into a single string
        full_text = " ".join(description_texts)

        # Get the page title
        title_tag = soup.find('h1', id='firstHeading')
        title = title_tag.get_text(strip=True) if title_tag else "Untitled"

        return title, full_text
