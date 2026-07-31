"""
Test suite for Lead Enrichment Pipeline.
Tests cover: health endpoint, rate limiting, caching, batch processing,
and error handling.
"""
import pytest
import json
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

# Import the app
import sys
sys.path.insert(0, str(Path(__file__).parent / "app"))
from main import app, init_cache, CACHE_DB, check_rate_limit, rate_tracker


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_rate_limit():
    """Reset rate limiter before each test."""
    rate_tracker.clear()
    yield
    rate_tracker.clear()


class TestHealthEndpoint:
    """Tests for the /api/health endpoint."""

    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data

    def test_health_lists_features(self, client):
        resp = client.get("/api/health")
        data = resp.json()
        assert "caching" in data["features"]
        assert "batch-csv" in data["features"]
        assert "rate-limiting" in data["features"]

    def test_health_shows_rate_limit(self, client):
        resp = client.get("/api/health")
        data = resp.json()
        assert "rate_limit" in data
        assert "max_batch_size" in data
        assert data["max_batch_size"] == 20


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_allows_under_limit(self):
        ip = "192.168.1.1"
        for _ in range(10):
            assert check_rate_limit(ip) is True

    def test_rate_limit_blocks_over_limit(self):
        ip = "192.168.1.2"
        for _ in range(10):
            check_rate_limit(ip)
        assert check_rate_limit(ip) is False

    def test_rate_limit_independent_per_ip(self):
        ip1 = "10.0.0.1"
        ip2 = "10.0.0.2"
        for _ in range(10):
            check_rate_limit(ip1)
        # Different IP should still be allowed
        assert check_rate_limit(ip2) is True


class TestEnrichEndpoint:
    """Tests for the /api/enrich endpoint."""

    def test_missing_url_returns_400(self, client):
        resp = client.post("/api/enrich", json={})
        assert resp.status_code == 400
        assert "URL is required" in resp.json()["error"]

    def test_empty_url_returns_400(self, client):
        resp = client.post("/api/enrich", json={"url": ""})
        assert resp.status_code == 400

    @patch("main.scrape_website", new_callable=AsyncMock)
    @patch("main.extract_company_info", new_callable=AsyncMock)
    def test_successful_enrichment(self, mock_extract, mock_scrape, client):
        mock_scrape.return_value = {
            "text": "Stripe is a technology company that builds economic infrastructure for the internet.",
            "title": "Stripe",
            "meta_description": "Online payment processing",
            "social_links": {"GitHub": "https://github.com/stripe"},
            "emails": ["support@stripe.com"],
        }
        mock_extract.return_value = {
            "company_description": "Payment processing company",
            "industry": "Fintech",
            "company_size": "500+",
            "location": "San Francisco, CA",
            "key_contacts": [],
            "tech_stack": ["React", "Node.js"],
            "value_proposition": "Payments infrastructure",
            "target_market": "B2B",
        }

        resp = client.post("/api/enrich", json={"url": "https://stripe.com"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["company_name"] == "Stripe"
        assert data["data"]["industry"] == "Fintech"
        assert data["cached"] is False

    @patch("main.scrape_website", new_callable=AsyncMock)
    def test_blocked_site_returns_clear_error(self, mock_scrape, client):
        mock_scrape.return_value = {
            "text": "",
            "title": "",
            "meta_description": "",
            "social_links": {},
            "emails": [],
            "_error": "This website is protected by Cloudflare and blocks automated scraping.",
        }
        resp = client.post("/api/enrich", json={"url": "https://protected-site.com"})
        assert resp.status_code == 422
        assert "Cloudflare" in resp.json()["error"]


class TestBatchEndpoint:
    """Tests for the /api/batch endpoint."""

    def test_batch_with_no_urls_returns_400(self, client):
        resp = client.post("/api/batch", json={"urls": []})
        assert resp.status_code == 400

    def test_batch_over_20_returns_400(self, client):
        urls = [f"https://example{i}.com" for i in range(21)]
        resp = client.post("/api/batch", json={"urls": urls})
        assert resp.status_code == 400
        assert "Maximum 20" in resp.json()["error"]

    def test_batch_with_valid_urls_returns_results(self, client):
        # This test uses mocked scraping
        with patch("main.scrape_website", new_callable=AsyncMock) as mock_scrape, \
             patch("main.extract_company_info", new_callable=AsyncMock) as mock_extract:
            mock_scrape.return_value = {
                "text": "A company website with enough text content here.",
                "title": "Test Co",
                "meta_description": "Test company",
                "social_links": {},
                "emails": [],
            }
            mock_extract.return_value = {
                "company_description": "Test company",
                "industry": "Technology",
                "company_size": "1-10",
                "location": "Not found",
                "key_contacts": [],
                "tech_stack": [],
                "value_proposition": "Testing",
                "target_market": "B2B",
            }

            resp = client.post("/api/batch", json={"urls": ["https://test1.com"]})
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 1
            assert data["successful"] == 1


class TestCaching:
    """Tests for the SQLite caching system."""

    def test_cache_init_creates_db(self):
        assert CACHE_DB.exists()

    def test_set_and_get_cached(self):
        import main
        test_url = "https://test-caching.example.com"
        test_result = {"company_name": "Test", "data": {"industry": "Tech"}}
        main.set_cached(test_url, test_result)
        cached = main.get_cached(test_url)
        assert cached is not None
        assert cached["company_name"] == "Test"

    def test_get_nonexistent_returns_none(self):
        import main
        assert main.get_cached("https://nonexistent.example.com") is None