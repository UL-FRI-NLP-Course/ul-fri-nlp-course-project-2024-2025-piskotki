import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
import threading
import atexit

# Base settings
BASE_URL = "https://en.wikipedia.org"
START_CATEGORY = "https://en.wikipedia.org/wiki/Category:Video_games"

# Delay between requests (in seconds)
# Wikipedia guideline: 1s crawl-delay. We use 0.5s at your own risk.
SLEEP_TIME = 0

visited_categories = set()
game_pages = set()
lock = threading.Lock()

# Persistent HTTP session with custom User-Agent
session = requests.Session()
session.headers.update({
    'User-Agent': 'WikiGameScraper/1.0 (https://yourdomain.example/)'
})


def save_progress():
    """
    Write current progress to disk.
    """
    with lock:
        with open('data/video_game_wikipedia_pages.txt', 'w', encoding='utf-8') as f:
            for page in sorted(game_pages):
                f.write(page + '\n')
        with open('data/visited_categories.txt', 'w', encoding='utf-8') as f:
            for cat in sorted(visited_categories):
                f.write(cat + '\n')
    print(f"[Checkpoint] {len(game_pages)} pages, {len(visited_categories)} categories saved.")

# ensure progress is saved on exit
atexit.register(save_progress)


def get_soup(url):
    """
    Fetch URL and return BeautifulSoup, sleeping between requests.
    """
    time.sleep(SLEEP_TIME)
    resp = session.get(url)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def scrape_category(url):
    """
    Recursively scrape the given category page:
    - Collect article links into game_pages
    - Track visited_categories
    - Follow pagination and subcategories
    """
    if url in visited_categories:
        return
    with lock:
        visited_categories.add(url)
    print(f"Scraping category: {url}")

    try:
        # 1) Handle paginated pages of this category
        next_page = url
        while next_page:
            soup = get_soup(next_page)

            # Collect game article links
            for link in soup.select('div#mw-pages a[href]'):
                href = link.get('href')
                if href and href.startswith('/wiki/') and ':' not in href:
                    full_url = urllib.parse.urljoin(BASE_URL, href)
                    with lock:
                        game_pages.add(full_url)

            # Find "next page" link if present
            next_link = soup.find('a', string='next page')
            next_page = urllib.parse.urljoin(BASE_URL, next_link['href']) if next_link and next_link.get('href') else None

        # 2) Recurse into subcategories (from first page)
        soup0 = get_soup(url)
        for subcat in soup0.select('div#mw-subcategories ul li a'):
            href = subcat.get('href')
            if href:
                subcat_url = urllib.parse.urljoin(BASE_URL, href)
                scrape_category(subcat_url)

    except Exception as e:
        # Save progress on any error and re-raise
        print(f"Error at category {url}: {e}")
        save_progress()
        raise


if __name__ == '__main__':
    try:
        scrape_category(START_CATEGORY)
    except KeyboardInterrupt:
        print("\nScrape interrupted by user. Saving progress...")
    finally:
        save_progress()
        print(f"Finished with {len(game_pages)} game pages across {len(visited_categories)} categories.")
