# Lead Enrichment Pipeline — Case Study

## The Problem

Sales and marketing teams spend hours manually researching prospects — visiting company websites, hunting for contact info, identifying decision-makers, and figuring out tech stack and target market. This is tedious, inconsistent, and doesn't scale. One rep might find the CTO's email; another might miss it entirely. The quality of lead research depends entirely on who does it and how much time they have.

Existing tools (ZoomInfo, Clearbit) are expensive and opaque — you can't see how they got the data or customize what's extracted. For small teams and freelancers, there's no affordable option that provides transparent, AI-powered company intelligence on demand.

## My Role

Solo developer. I built this as a personal project to demonstrate a practical AI automation pipeline — combining web scraping, LLM extraction, and structured data output. This directly reinforces the automation experience from my work at Relaytask, where I built n8n and GHL workflow automations for marketing operations.

## The Solution

A web app where you enter a company website URL, and the AI automatically:
1. Scrapes the website content
2. Extracts structured business intelligence using an LLM
3. Presents it as clean, exportable data cards

### Architecture

```
User input (company URL)
    ↓
FastAPI backend receives request
    ↓
Web scraper (httpx + BeautifulSoup)
  - Fetches website HTML
  - Removes script/style/nav/footer noise
  - Extracts: text content, page title, meta description
  - Finds: email addresses (regex), social media links
    ↓
LLM extraction (deepseek-v4-flash via Ollama Cloud)
  - Receives system prompt (B2B analyst role)
  - Gets cleaned website text (max 8000 chars)
  - Returns structured JSON with 8 fields
    ↓
Backend merges scraped data (emails, social links) with AI output
    ↓
Frontend renders as visual data cards
    ↓
User can export as CSV or JSON
```

### What Gets Extracted

| Field | Description |
|-------|-------------|
| Company Description | 2-3 sentence summary of what the company does |
| Industry | Primary industry/sector |
| Company Size | Estimated range (1-10, 11-50, 51-200, etc.) |
| Location | Headquarters location |
| Key Contacts | Names, titles, emails, LinkedIn profiles |
| Tech Stack | Technologies mentioned on the site |
| Value Proposition | 1-sentence summary of their main offering |
| Target Market | B2B, B2C, enterprise, SMB, etc. |

### Key Technical Decisions

1. **Server-side scraping (not headless browser)** — Used httpx + BeautifulSoup instead of Playwright/Selenium. Faster, lighter, and sufficient for most company websites. JavaScript-heavy sites that don't render server-side will get less data — a known tradeoff for speed.

2. **Text truncation at 8000 chars** — LLM context windows are finite. 8000 chars gives enough content for accurate extraction while keeping the LLM call fast (5-6 seconds typical).

3. **Structured JSON output from LLM** — The system prompt explicitly defines the JSON schema. If the LLM wraps output in markdown, the backend strips it and parses. If parsing fails, a regex extracts the JSON block as a fallback.

4. **Hybrid data collection** — Emails and social links are extracted via regex/DOM parsing (reliable, deterministic) AND via the LLM (may catch things the regex misses). Results are merged so nothing is lost.

5. **In-memory search history** — Last 20 searches stored in memory. No database needed for a demo tool. History persists as long as the container runs.

6. **CSV + JSON export** — Sales teams need CSV for spreadsheets and CRM import. Developers need JSON for API integration. Both are client-side generated (Blob + download) so no server round-trip.

### Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Backend | Python FastAPI | Async, fast, clean API structure |
| Web Scraping | httpx + BeautifulSoup | Async HTTP + robust HTML parsing |
| AI | deepseek-v4-flash (Ollama Cloud) | Fast, affordable, good at structured extraction |
| Frontend | Vanilla HTML/CSS/JS | No build step, fast load, easy to maintain |
| Deployment | Docker, Caddy, Cloudflare | Consistent with server infrastructure |

### Challenges & Solutions

- **JavaScript-rendered sites** — Some sites (React/Next.js SPAs) return minimal HTML. The scraper gets the static HTML, which may not include dynamic content. Trade-off: speed over completeness. A Playwright fallback could be added for JS-heavy sites.

- **LLM output format inconsistency** — The LLM sometimes wraps JSON in markdown code blocks or adds explanatory text. Solution: strip markdown wrappers, then regex-extract JSON as a fallback parser.

- **Email extraction quality** — Regex catches emails in the text, but also catches non-contact emails (e.g. support@sentry.io from a script tag). Solution: filter out known non-contact patterns (example.com, sentry, wixpress, etc.).

- **Rate of scraping** — No rate limiting on the scraper. For a demo tool this is fine, but for production use, a queue with rate limiting would be needed to avoid being blocked by target sites.

### What I Learned

- Web scraping at scale requires careful handling of edge cases (redirects, JS-rendered content, bot detection, broken HTML)
- LLMs are excellent at structured extraction from unstructured text, but need strong prompt engineering to consistently return valid JSON
- Combining deterministic methods (regex, DOM parsing) with LLM extraction gives better results than either approach alone
- Async Python (httpx + FastAPI) is significantly faster than synchronous alternatives for I/O-bound workloads like web scraping

## Live Demo

**URL:** https://leads.betamaxgroup.tech

Try it with any company website — enter the URL, click Enrich, and get structured business intelligence in about 5 seconds. Export the results as CSV or JSON.