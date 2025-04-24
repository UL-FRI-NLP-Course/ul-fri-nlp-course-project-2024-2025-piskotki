import requests
import json
import os
import signal
import re
from bs4 import BeautifulSoup, Tag

# Hardcoded file names
INPUT_FILE = 'video_game_wikipedia_pages.txt'
OUTPUT_FILE = 'video_game_descriptions.json'

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

    data = {
        'url': url,
        'title': soup.find('h1', id='firstHeading').get_text(strip=True),
        'description': description_texts
    }
    return data


def main():
    # Load URLs
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    # Load existing output if it exists
    output_data = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            try:
                output_data = json.load(f)
            except json.JSONDecodeError:
                pass

    completed = set(output_data.keys())

    for url in urls:
        if stop_signal:
            print('Interrupted; stopping early.')
            break
        try:
            page_data = scrape_page(url)
            title = page_data['title']
            if title in completed:
                continue

            # Insert into output structure
            output_data[title] = {
                'description': page_data['description']
            }

            # Save progress after each page
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
                json.dump(output_data, out, ensure_ascii=False, indent=4)
            print(f"Saved {title}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

if __name__ == '__main__':
    main()
