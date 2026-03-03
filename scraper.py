import requests
from bs4 import BeautifulSoup

# Full browser-like headers to avoid blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,it;q=0.8,de;q=0.7,fr;q=0.6",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}

MIN_WORDS = 100  # if fewer words scraped, try JS fallback


def _parse_soup(soup):
    for tag in soup(["script", "style", "noscript", "svg", "path"]):
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
        "word_count": len(body.split()),
    }


def _scrape_static(url):
    session = requests.Session()
    session.headers.update(HEADERS)
    resp = session.get(url, timeout=20, allow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_soup(soup)


def _scrape_playwright(url):
    from playwright.sync_api import sync_playwright  # optional dependency
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            locale="en-US",
            extra_http_headers={
                "Accept-Language": HEADERS["Accept-Language"],
            },
        )
        page = ctx.new_page()
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
        "word_count": 0,
    }

    try:
        result = _scrape_static(url)
        if result["word_count"] < MIN_WORDS:
            try:
                js_result = _scrape_playwright(url)
                if js_result["word_count"] > result["word_count"]:
                    result = js_result
            except Exception:
                pass  # keep static result if playwright unavailable
        return {**result, "url": url, "error": None}

    except Exception as static_err:
        try:
            result = _scrape_playwright(url)
            return {**result, "url": url, "error": None}
        except Exception as js_err:
            return {**empty, "url": url, "error": f"Static: {static_err} | JS: {js_err}"}
