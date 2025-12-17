from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import os
import time

os.makedirs("data", exist_ok=True)

BASE_URL = "https://www.shl.com/products/product-catalog/"
PAGE_SIZE = 12
MAX_PAGES = 100  # safety cap
TYPES = [1, 2, 3, 4]  # ALL individual test categories

all_links = set()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    for t in TYPES:
        print(f"\nScraping type={t}")

        for page_num in range(MAX_PAGES):
            start = page_num * PAGE_SIZE
            url = f"{BASE_URL}?start={start}&type={t}"

            print(f"  start={start}")

            page.goto(url, timeout=60000)
            page.wait_for_load_state("networkidle")
            time.sleep(1)

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            rows = soup.select("td.custom__table-heading__title a[href]")
            if not rows:
                break  # no more pages for this type

            for a in rows:
                href = a["href"]
                if href.startswith("/products/product-catalog/view/"):
                    all_links.add("https://www.shl.com" + href)

    browser.close()

# Save links
with open("data/assessment_links.txt", "w", encoding="utf-8") as f:
    for link in sorted(all_links):
        f.write(link + "\n")

print(f"\nTotal Individual Test Solution links collected: {len(all_links)}")
