"""
Lead Enrichment Pipeline — AI-powered company research tool.
Scrapes a company website, uses LLM to extract structured business intelligence.
Features: caching, batch CSV processing, rate limiting, better error handling.
"""
import os
import re
import json
import time
import httpx
import asyncio
import sqlite3
from collections import defaultdict, deque
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

app = FastAPI(title="Lead Enrichment Pipeline")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
MODEL = os.getenv("MODEL", "deepseek-v4-flash")

# In-memory search history (most recent first, max 20)
search_history: list[dict] = []

# ---------------------------------------------------------------------------
# SQLite cache — persists enrichment results across restarts
# ---------------------------------------------------------------------------
CACHE_DB = Path(__file__).parent / "cache.db"

def init_cache():
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS enrichment_cache (
            url TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def get_cached(url: str) -> dict | None:
    conn = sqlite3.connect(str(CACHE_DB))
    row = conn.execute(
        "SELECT result_json FROM enrichment_cache WHERE url = ?",
        (url,)
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def set_cached(url: str, result: dict):
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute(
        "INSERT OR REPLACE INTO enrichment_cache (url, result_json, created_at) VALUES (?, ?, ?)",
        (url, json.dumps(result), time.time())
    )
    conn.commit()
    conn.close()

init_cache()

# ---------------------------------------------------------------------------
# Rate limiting — max 10 requests per minute per IP
# ---------------------------------------------------------------------------
RATE_LIMIT = 10  # requests per minute
RATE_WINDOW = 60  # seconds
rate_tracker: dict[str, deque] = defaultdict(lambda: deque())

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    tracker = rate_tracker[ip]
    # Remove old entries
    while tracker and now - tracker[0] > RATE_WINDOW:
        tracker.popleft()
    if len(tracker) >= RATE_LIMIT:
        return False
    tracker.append(now)
    return True

# ---------------------------------------------------------------------------
# Web scraping
# ---------------------------------------------------------------------------

BLOCKED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".pdf", ".zip", ".mp4", ".mp3", ".woff", ".woff2", ".ttf",
    ".css", ".js", ".map",
}

SOCIAL_DOMAINS = {
    "linkedin.com": "LinkedIn",
    "twitter.com": "Twitter",
    "x.com": "Twitter",
    "facebook.com": "Facebook",
    "instagram.com": "Instagram",
    "youtube.com": "YouTube",
    "github.com": "GitHub",
    "crunchbase.com": "Crunchbase",
}


async def scrape_website(url: str, max_chars: int = 8000) -> dict:
    """
    Fetch a company website and extract clean text + metadata.
    Returns {text, title, meta_description, social_links, emails}.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        )
    }

    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            verify=False,
        ) as client:
            resp = await client.get(url, headers=headers)

        # Check for Cloudflare/bot protection
        if resp.status_code == 403:
            cf_ray = resp.headers.get("cf-ray", "")
            if cf_ray:
                raise httpx.HTTPStatusError(
                    "CLOUDFLARE_BLOCKED",
                    request=resp.request,
                    response=resp,
                )
            raise httpx.HTTPStatusError(
                "ACCESS_DENIED",
                request=resp.request,
                response=resp,
            )

        resp.raise_for_status()
        html = resp.text

    except httpx.ConnectTimeout:
        return {
            "text": "",
            "title": "",
            "meta_description": "",
            "social_links": {},
            "emails": [],
            "_error": "Connection timed out — the website took too long to respond.",
        }
    except httpx.ConnectError:
        return {
            "text": "",
            "title": "",
            "meta_description": "",
            "social_links": {},
            "emails": [],
            "_error": f"Could not connect to {url}. Check if the URL is correct.",
        }

    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, footer, header noise
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()

    # Title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Meta description
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()
    if not meta_desc:
        og_tag = soup.find("meta", attrs={"property": "og:description"})
        if og_tag and og_tag.get("content"):
            meta_desc = og_tag["content"].strip()

    # Main text content
    text_parts = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li", "span", "div"]):
        t = el.get_text(strip=True)
        if t and len(t) > 10:
            text_parts.append(t)

    full_text = " ".join(text_parts)
    # Collapse whitespace
    full_text = re.sub(r"\s+", " ", full_text).strip()
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]

    # Social links
    social_links = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        for domain, name in SOCIAL_DOMAINS.items():
            if domain in href and name not in social_links:
                social_links[name] = href

    # Email addresses
    emails = list(set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", full_text)))
    # Filter out obvious non-contact emails
    emails = [e for e in emails if not any(x in e.lower() for x in ["example.com", "sentry", "wixpress", "your-domain"])]

    return {
        "text": full_text,
        "title": title,
        "meta_description": meta_desc,
        "social_links": social_links,
        "emails": emails[:10],
    }


# ---------------------------------------------------------------------------
# AI extraction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a B2B lead enrichment analyst. You scrape company websites and extract structured business intelligence for sales and marketing teams.

Given website text, extract:
- company_description: 2-3 sentence summary of what the company does
- industry: primary industry/sector
- company_size: estimated size range (e.g. "1-10", "11-50", "51-200", "201-500", "500+")
- location: headquarters location if findable, otherwise "Not found"
- key_contacts: list of {name, title, email, linkedin} found on the page (empty list if none)
- tech_stack: technologies mentioned on the site (frameworks, platforms, tools)
- value_proposition: 1-sentence summary of their main offering
- target_market: who they sell to (B2B, B2C, enterprise, SMB, etc.)

Rules:
- Only include information you can find in the text. Do NOT make things up.
- If something is not found, use "Not found" or an empty list.
- Emails must be actual email addresses found in the text.
- Names must be actual names found in the text (e.g. from team/about pages).
- Return ONLY valid JSON, no markdown, no explanations.

Return this JSON structure:
{
  "company_description": "...",
  "industry": "...",
  "company_size": "...",
  "location": "...",
  "key_contacts": [{"name": "...", "title": "...", "email": "...", "linkedin": "..."}],
  "tech_stack": ["...", "..."],
  "value_proposition": "...",
  "target_market": "..."
}"""

USER_PROMPT_TEMPLATE = """Company website URL: {url}
Page title: {title}
Meta description: {meta_desc}

Website text (first {chars} chars):
---
{text}
---

Extract the business intelligence as JSON."""


async def extract_company_info(
    url: str,
    scraped: dict,
) -> dict:
    """Send scraped text to the LLM and get structured company info."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        url=url,
        title=scraped["title"],
        meta_desc=scraped["meta_description"],
        chars=len(scraped["text"]),
        text=scraped["text"],
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2000,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]

    # Some models put the response in a "reasoning" field
    if not content and "reasoning" in data["choices"][0]["message"]:
        content = data["choices"][0]["message"]["reasoning"]

    # Strip markdown wrappers if present
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()

    # Remove any leading/trailing non-JSON text
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1:
        content = content[first_brace:last_brace + 1]

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                parsed = {"error": "Failed to parse LLM output", "raw": content[:500]}
        else:
            parsed = {"error": "No JSON found in LLM output", "raw": content[:500]}

    return parsed


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "features": ["caching", "batch-csv", "rate-limiting"],
        "rate_limit": f"{RATE_LIMIT} requests per {RATE_WINDOW}s",
        "max_batch_size": 20,
    }


@app.get("/api/history")
async def history():
    return {"history": search_history[:20]}


@app.post("/api/enrich")
async def enrich_company(request: Request):
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        return JSONResponse(
            {"error": "Rate limit exceeded. Maximum 10 requests per minute. Please wait and try again."},
            status_code=429,
        )

    body = await request.json()
    company_name = body.get("company_name", "").strip()
    url = body.get("url", "").strip()

    if not url:
        return JSONResponse(
            {"error": "Website URL is required"},
            status_code=400,
        )

    # Normalize URL
    if not url.startswith("http"):
        url = "https://" + url

    # Check cache first
    cached = get_cached(url)
    if cached:
        cached["cached"] = True
        return JSONResponse(cached)

    start_time = time.time()

    try:
        # Step 1: Scrape website
        scraped = await scrape_website(url)

        # Check for scrape errors
        if scraped.get("_error"):
            return JSONResponse(
                {"error": scraped["_error"]},
                status_code=422,
            )

        if len(scraped["text"]) < 50:
            return JSONResponse(
                {"error": "Could not extract enough text from the website. The site may be JavaScript-rendered, behind a login wall, or blocking automated access."},
                status_code=422,
            )

        # Step 2: AI extraction
        ai_result = await extract_company_info(url, scraped)

        elapsed = round(time.time() - start_time, 1)

        # Merge scraped data (emails, social links) with AI result
        if isinstance(ai_result, dict) and "error" not in ai_result:
            # Add scraped emails if AI didn't find them
            if not ai_result.get("key_contacts"):
                ai_result["key_contacts"] = []
            if scraped["emails"] and not any(c.get("email") for c in ai_result["key_contacts"]):
                for email in scraped["emails"][:3]:
                    ai_result["key_contacts"].append({
                        "name": "Not found",
                        "title": "Not found",
                        "email": email,
                        "linkedin": "",
                    })

            # Ensure social links are included
            ai_result["social_links"] = scraped.get("social_links", {})
            ai_result["scraped_emails"] = scraped.get("emails", [])
        else:
            ai_result["social_links"] = scraped.get("social_links", {})
            ai_result["scraped_emails"] = scraped.get("emails", [])

        result = {
            "company_name": company_name or scraped.get("title", url),
            "url": url,
            "data": ai_result,
            "elapsed_seconds": elapsed,
            "scraped_text_length": len(scraped["text"]),
            "cached": False,
        }

        # Cache the result
        set_cached(url, result)

        # Add to history (keep most recent 20)
        search_history.insert(0, {
            "company_name": result["company_name"],
            "url": url,
            "industry": ai_result.get("industry", "Unknown"),
            "elapsed_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        if len(search_history) > 20:
            search_history.pop()

        return JSONResponse(result)

    except httpx.HTTPStatusError as e:
        error_msg = str(e)
        if "CLOUDFLARE_BLOCKED" in error_msg:
            return JSONResponse(
                {"error": f"This website is protected by Cloudflare and blocks automated scraping. Try a different URL or check the company manually."},
                status_code=403,
            )
        return JSONResponse(
            {"error": f"Website returned HTTP {e.response.status_code}"},
            status_code=502,
        )
    except httpx.RequestError as e:
        return JSONResponse(
            {"error": f"Could not reach the website: {str(e)[:200]}"},
            status_code=502,
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"An error occurred: {str(e)[:200]}"},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Batch CSV processing
# ---------------------------------------------------------------------------

@app.post("/api/batch")
async def batch_enrich(request: Request):
    """Process multiple company URLs from a JSON list."""
    # Rate limiting (batch counts as 1 request for rate limit, but max 20 URLs)
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        return JSONResponse(
            {"error": "Rate limit exceeded. Maximum 10 requests per minute."},
            status_code=429,
        )

    body = await request.json()
    urls = body.get("urls", [])

    if not urls or not isinstance(urls, list):
        return JSONResponse(
            {"error": "Please provide a list of URLs in the 'urls' field"},
            status_code=400,
        )

    if len(urls) > 20:
        return JSONResponse(
            {"error": "Maximum 20 URLs per batch request"},
            status_code=400,
        )

    results = []
    errors = []

    for i, url in enumerate(urls):
        url = url.strip()
        if not url:
            continue
        if not url.startswith("http"):
            url = "https://" + url

        # Check cache
        cached = get_cached(url)
        if cached:
            cached["cached"] = True
            results.append(cached)
            continue

        try:
            scraped = await scrape_website(url)

            if scraped.get("_error"):
                errors.append({"url": url, "error": scraped["_error"]})
                continue

            if len(scraped["text"]) < 50:
                errors.append({"url": url, "error": "Not enough text extracted from website"})
                continue

            ai_result = await extract_company_info(url, scraped)
            elapsed = 0  # batch doesn't track individual time

            if isinstance(ai_result, dict) and "error" not in ai_result:
                if not ai_result.get("key_contacts"):
                    ai_result["key_contacts"] = []
                ai_result["social_links"] = scraped.get("social_links", {})
                ai_result["scraped_emails"] = scraped.get("emails", [])

            result = {
                "company_name": scraped.get("title", url),
                "url": url,
                "data": ai_result,
                "elapsed_seconds": elapsed,
                "scraped_text_length": len(scraped["text"]),
                "cached": False,
            }

            set_cached(url, result)
            results.append(result)

        except httpx.HTTPStatusError as e:
            if "CLOUDFLARE_BLOCKED" in str(e):
                errors.append({"url": url, "error": "Website protected by Cloudflare — blocks automated scraping"})
            else:
                errors.append({"url": url, "error": f"Website returned HTTP {e.response.status_code}"})
        except Exception as e:
            errors.append({"url": url, "error": str(e)[:200]})

    return JSONResponse({
        "total": len(urls),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    })


@app.post("/api/batch-csv")
async def batch_csv_enrich(file: UploadFile = File(...)):
    """Upload a CSV file with company URLs, process them all."""
    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    # Parse CSV — expect a column with URLs (first column or a "url"/"website" column)
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return JSONResponse(
            {"error": "CSV must have a header row and at least one data row"},
            status_code=400,
        )

    # Find the URL column
    header = lines[0].lower().strip()
    url_col_idx = 0
    for idx, col in enumerate(header.split(",")):
        if "url" in col or "website" in col or "site" in col:
            url_col_idx = idx
            break

    urls = []
    for line in lines[1:]:
        cols = line.split(",")
        if len(cols) > url_col_idx:
            url = cols[url_col_idx].strip().strip('"').strip("'")
            if url:
                urls.append(url)

    if not urls:
        return JSONResponse(
            {"error": "No URLs found in the CSV"},
            status_code=400,
        )

    if len(urls) > 20:
        urls = urls[:20]

    # Process each URL
    results = []
    errors = []

    for url in urls:
        if not url.startswith("http"):
            url = "https://" + url

        cached = get_cached(url)
        if cached:
            cached["cached"] = True
            results.append(cached)
            continue

        try:
            scraped = await scrape_website(url)

            if scraped.get("_error"):
                errors.append({"url": url, "error": scraped["_error"]})
                continue

            if len(scraped["text"]) < 50:
                errors.append({"url": url, "error": "Not enough text extracted"})
                continue

            ai_result = await extract_company_info(url, scraped)

            if isinstance(ai_result, dict) and "error" not in ai_result:
                if not ai_result.get("key_contacts"):
                    ai_result["key_contacts"] = []
                ai_result["social_links"] = scraped.get("social_links", {})
                ai_result["scraped_emails"] = scraped.get("emails", [])

            result = {
                "company_name": scraped.get("title", url),
                "url": url,
                "data": ai_result,
                "elapsed_seconds": 0,
                "scraped_text_length": len(scraped["text"]),
                "cached": False,
            }

            set_cached(url, result)
            results.append(result)

        except httpx.HTTPStatusError as e:
            if "CLOUDFLARE_BLOCKED" in str(e):
                errors.append({"url": url, "error": "Cloudflare blocked"})
            else:
                errors.append({"url": url, "error": f"HTTP {e.response.status_code}"})
        except Exception as e:
            errors.append({"url": url, "error": str(e)[:200]})

    return JSONResponse({
        "total": len(urls),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    })


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text())