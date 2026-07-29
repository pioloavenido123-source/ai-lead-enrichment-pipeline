"""
Lead Enrichment Pipeline — AI-powered company research tool.
Scrapes a company website, uses LLM to extract structured business intelligence.
"""
import os
import re
import json
import time
import httpx
import asyncio
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="Lead Enrichment Pipeline")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
MODEL = os.getenv("MODEL", "deepseek-v4-flash")

# In-memory search history (most recent first, max 20)
search_history: list[dict] = []

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

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        verify=False,
    ) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        html = resp.text

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
        "max_tokens": 1000,
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

    # Strip markdown wrappers if present
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()

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
    return {"status": "ok", "model": MODEL}


@app.get("/api/history")
async def history():
    return {"history": search_history[:20]}


@app.post("/api/enrich")
async def enrich_company(request: Request):
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

    start_time = time.time()

    try:
        # Step 1: Scrape website
        scraped = await scrape_website(url)

        if len(scraped["text"]) < 50:
            return JSONResponse(
                {"error": "Could not extract enough text from the website. The site may be JavaScript-rendered or blocked."},
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
        }

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
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text())