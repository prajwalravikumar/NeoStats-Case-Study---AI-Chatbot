import requests
from config.config import SERPAPI_API_KEY
import logging

logger = logging.getLogger(__name__)

def serpapi_search(query: str, num_results: int = 3):
    if not SERPAPI_API_KEY:
        return []
    try:
        params = {"q": query, "api_key": SERPAPI_API_KEY, "engine": "google", "num": num_results}
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        data = resp.json()
        out = []
        for r in data.get("organic_results", [])[:num_results]:
            out.append({"title": r.get("title"), "snippet": r.get("snippet"), "link": r.get("link")})
        return out
    except Exception as e:
        logger.exception("SerpAPI search failed: %s", e)
        return []