import os
from typing import List, Dict, Any

import requests

BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


def web_search_bing(api_key: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform a simple Bing Web Search and return top_k results as dicts with
    title, url, snippet.

    This is optional: if `api_key` is not provided or the call fails, returns []
    so the overall evaluation continues.
    """
    if not api_key or not query:
        return []
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "mkt": "en-US", "count": top_k}
    try:
        resp = requests.get(BING_ENDPOINT, headers=headers, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        results = []
        web_pages = data.get("webPages", {}).get("value", [])
        for item in web_pages[:top_k]:
            results.append({
                "title": item.get("name"),
                "url": item.get("url"),
                "snippet": item.get("snippet") or item.get("displayUrl") or "",
            })
        return results
    except Exception:
        return []


def web_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """High-level wrapper: read SEARCH_API_KEY from env and call provider.

    Set environment variable `SEARCH_API_KEY` to enable. Currently only Bing
    is supported (Azure Cognitive Bing Web Search) via `SEARCH_API_KEY`.
    """
    api_key = os.environ.get("SEARCH_API_KEY")
    if not api_key:
        return []
    return web_search_bing(api_key, query, top_k=top_k)
