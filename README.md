# Lead Enrichment Pipeline

AI-powered company research tool. Enter a company website URL, get structured business intelligence in seconds.

## Features

- Web scraping (httpx + BeautifulSoup) extracts clean text from any company website
- AI extraction (deepseek-v4-flash) identifies 8 key data points from the scraped content
- CSV and JSON export for CRM import or API integration
- Search history (in-memory, last 20 searches)
- Clean dark-themed UI with animated loading steps

## What It Extracts

- Company description (2-3 sentences)
- Industry/sector
- Estimated company size
- Headquarters location
- Key contacts (name, title, email, LinkedIn)
- Tech stack
- Value proposition
- Target market (B2B, B2C, enterprise, SMB)

## Tech Stack

- **Backend:** Python FastAPI (async)
- **Scraping:** httpx + BeautifulSoup
- **AI:** deepseek-v4-flash via Ollama Cloud
- **Frontend:** Vanilla HTML/CSS/JS
- **Deployment:** Docker, Caddy, Cloudflare

## Setup

```bash
# 1. Copy .env.example to .env and add your Ollama Cloud API key
cp .env.example .env

# 2. Build and run
docker compose up --build -d

# 3. Access at http://localhost:9018
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OLLAMA_BASE_URL | Ollama Cloud API base URL | https://ollama.com/v1 |
| OLLAMA_API_KEY | Your Ollama Cloud API key | (required) |
| MODEL | LLM model name | deepseek-v4-flash |

## Live Demo

https://leads.betamaxgroup.tech

## License

MIT