from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os

# ---------------- CONFIG ----------------
LINKS_FILE = "data/assessment_links.txt"
OUTPUT_CSV = "data/shl_assessments.csv"
FAILED_URLS_FILE = "data/failed_urls.txt"

MAX_RETRIES = 2
PAGE_TIMEOUT = 30000  # 30 seconds
SLEEP_BETWEEN_PAGES = 1  # polite delay

# ----------------------------------------

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Load URLs
with open(LINKS_FILE, "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]

print(f"Total URLs to scrape: {len(urls)}")

records = []
failed_urls = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    for idx, url in enumerate(urls, start=1):
        print(f"\n[{idx}/{len(urls)}] Scraping: {url}")

        success = False

        # ---------- RETRY LOGIC ----------
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                page.goto(url, timeout=PAGE_TIMEOUT)
                page.wait_for_load_state("networkidle")
                time.sleep(SLEEP_BETWEEN_PAGES)
                success = True
                break
            except Exception as e:
                print(f"  Attempt {attempt} failed: {e}")

        if not success:
            print("  Skipping URL after retries.")
            failed_urls.append(url)
            continue

        soup = BeautifulSoup(page.content(), "html.parser")
        page_text = soup.get_text(separator="\n")

        # ---------- NAME ----------
        name = "Unknown"
        h1 = soup.find("h1")
        if h1:
            name = h1.get_text(strip=True)

        # ---------- DESCRIPTION ----------
        description = ""
        desc_heading = soup.find(string=re.compile(r"Description", re.I))
        if desc_heading:
            parent = desc_heading.parent
            paragraphs = parent.find_all_next("p", limit=6)
            description = " ".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

        # ---------- DURATION ----------
        duration = None
        duration_match = re.search(
            r"Approximate Completion Time.*?(\d+)", page_text, re.I
        )
        if duration_match:
            duration = int(duration_match.group(1))

        # ---------- TEST TYPE ----------
        test_type = ["Unknown"]
        test_type_match = re.search(r"Test Type\s*:\s*(.+)", page_text)
        if test_type_match:
            test_type = [test_type_match.group(1).strip()]

        # ---------- REMOTE SUPPORT ----------
        remote_support = "Unknown"
        remote_match = re.search(r"Remote Testing\s*:\s*(Yes|No)", page_text, re.I)
        if remote_match:
            remote_support = remote_match.group(1).capitalize()

        # ---------- ADAPTIVE SUPPORT ----------
        adaptive_support = "Unknown"
        adaptive_match = re.search(r"Adaptive\s*:\s*(Yes|No)", page_text, re.I)
        if adaptive_match:
            adaptive_support = adaptive_match.group(1).capitalize()

        # ---------- SAVE RECORD ----------
        records.append({
            "name": name,
            "url": url,
            "description": description,
            "duration": duration,
            "test_type": test_type,
            "adaptive_support": adaptive_support,
            "remote_support": remote_support
        })

    browser.close()

# ---------- SAVE CSV ----------
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

# ---------- SAVE FAILED URLS ----------
with open(FAILED_URLS_FILE, "w", encoding="utf-8") as f:
    for u in failed_urls:
        f.write(u + "\n")

print("\n================ SUMMARY ================")
print(f"Total URLs input      : {len(urls)}")
print(f"Successfully scraped  : {len(df)}")
print(f"Failed URLs           : {len(failed_urls)}")
print(f"CSV saved to           : {OUTPUT_CSV}")
print(f"Failed URLs saved to   : {FAILED_URLS_FILE}")
print("========================================")
