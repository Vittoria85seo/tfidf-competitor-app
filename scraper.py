import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

MIN_WORDS = 50  # if fewer words scraped, try JS fallback


def _parse_soup(soup):
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    meta_title = ""
    title_tag = soup.find("title")
    if title_tag:
        meta_title = title_tag.get_text(strip=True)

    meta_desc = ""
    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag:
        meta_desc = desc_tag.get("content", "")

    headings = []
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        t = h.get_text(strip=True)
        if t:
            headings.append(t)

    body = soup.get_text(separator=" ", strip=True)
    combined = " ".join([meta_title, meta_desc, " ".join(headings), body])

    return {
        "meta_title": meta_title,
        "meta_description": meta_desc,
        "headings": " ".join(headings),
        "body": body,
        "combined": combined,
    }


def _scrape_static(url):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_soup(soup)


def _scrape_playwright(url):
    from playwright.sync_api import sync_playwright  # optional dependency
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            extra_http_headers={"User-Agent": HEADERS["User-Agent"]}
        )
        page.goto(url, timeout=30000, wait_until="networkidle")
        content = page.content()
        browser.close()
    soup = BeautifulSoup(content, "html.parser")
    return _parse_soup(soup)


def scrape_url(url):
    """Scrape a URL and return a content dict. Never raises — errors go in result['error']."""
    empty = {
        "meta_title": "",
        "meta_description": "",
        "headings": "",
        "body": "",
        "combined": "",
    }

    try:
        result = _scrape_static(url)
        if len(result["body"].split()) < MIN_WORDS:
            try:
                result = _scrape_playwright(url)
            except Exception:
                pass  # keep static result if playwright unavailable
        return {**result, "url": url, "error": None}

    except Exception as static_err:
        try:
            result = _scrape_playwright(url)
            return {**result, "url": url, "error": None}
        except Exception as js_err:
            return {**empty, "url": url, "error": f"Static: {static_err} | JS: {js_err}"}
