import asyncio
from collections import Counter
import re
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from bs4 import BeautifulSoup
from fastapi.concurrency import run_in_threadpool
import requests

from .llm import generate_llm_summary

logger = logging.getLogger(__name__)
_MAX_PREVIEW_LENGTH = 500
_REQUEST_TIMEOUT = 45  # Increased from 15s to 45s for production sites that may be slower
_USER_AGENT = "CiteLabsCrawler/0.1 (+https://citelabs.local)"

# Debug flag for detailed crawler diagnostics
DEBUG_CRAWLER = os.getenv("DEBUG_CRAWLER", "False").lower() == "true"


async def crawl_url(url: str) -> Dict[str, Any]:
    try:
        html = await run_in_threadpool(_fetch_html, url)
    except requests.exceptions.Timeout as e:
        logger.error(f"Crawl timeout for {url}: {e}")
        return {"error": f"Failed to crawl: Timeout after {_REQUEST_TIMEOUT}s"}
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') and e.response else "unknown"
        logger.error(f"Crawl HTTP error for {url}: {status_code} - {e}")
        if status_code == 403:
            return {"error": f"Failed to crawl: HTTP 403 Forbidden (likely bot detection)"}
        elif status_code == 429:
            return {"error": f"Failed to crawl: HTTP 429 Too Many Requests (rate limited)"}
        elif status_code == 503:
            return {"error": f"Failed to crawl: HTTP 503 Service Unavailable"}
        else:
            return {"error": f"Failed to crawl: HTTP {status_code}"}
    except requests.exceptions.SSLError as e:
        logger.error(f"Crawl SSL error for {url}: {e}")
        return {"error": f"Failed to crawl: SSL certificate error"}
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Crawl connection error for {url}: {e}")
        return {"error": f"Failed to crawl: Connection error (network issue or site unreachable)"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Crawl request error for {url}: {type(e).__name__}: {e}")
        return {"error": f"Failed to crawl: {type(e).__name__} - {str(e)}"}
    except Exception as e:
        logger.error(f"Crawl unexpected error for {url}: {type(e).__name__}: {e}", exc_info=True)
        return {"error": f"Failed to crawl: {type(e).__name__} - {str(e)}"}

    soup = BeautifulSoup(html, "html.parser")
    _strip_unused_tags(soup)

    title = _safe_text(soup.title)
    h1 = _first_heading_text(soup)
    full_content = _extract_text(soup)
    page_type_guess = _guess_page_type(soup, full_content)
    keywords_guess = _guess_keywords(full_content)

    page_purpose_summary = await _generate_page_purpose_summary(full_content);

    return {
        "title": title,
        "h1": h1,
        "content_preview": full_content[:_MAX_PREVIEW_LENGTH] if full_content else "",
        "full_content": full_content,
        "page_type_guess": page_type_guess,
        "keywords_guess": keywords_guess,
        "page_purpose_summary": page_purpose_summary,
        "ai_summary": "skipping ai summary for now",
    }


async def crawl_text(url: str) -> Dict[str, Any]:
    """
    Fetch a page's title and text only. Unlike crawl_url, this makes no LLM call,
    for callers that only need the content (Stage 1 evidence extraction).
    Returns {"error": ...} on failure.
    """
    try:
        html = await run_in_threadpool(_fetch_html, url)
    except requests.exceptions.HTTPError as e:
        # Response is falsy for 4xx/5xx, so compare to None rather than truthiness
        status_code = e.response.status_code if e.response is not None else "unknown"
        return {"error": f"Failed to crawl: HTTP {status_code}"}
    except Exception as e:
        return {"error": f"Failed to crawl: {type(e).__name__} - {str(e)[:200]}"}

    soup = BeautifulSoup(html, "html.parser")
    _strip_unused_tags(soup)
    return {"title": _safe_text(soup.title), "h1": _first_heading_text(soup), "full_content": _extract_text(soup)}


def describe_crawler() -> str:
    """
    Returns a human-readable description of how this crawler appears to websites.
    This is for developer understanding only, not used in logic.
    """
    return (
        "Static HTML crawler using Python 'requests' library, no JavaScript execution, "
        "desktop Chrome-like headers, no cookies/sessions, follows redirects, "
        "non-Google user-agent. Classified as: Static HTML crawler (non-JS)."
    )


def _fetch_html(url: str) -> str:
    logger.info(f"Fetching HTML from {url} (timeout: {_REQUEST_TIMEOUT}s)")
    
    # Try with a more realistic browser User-Agent to avoid bot detection
    # Note: Removed "br" (Brotli) from Accept-Encoding as requests handles gzip/deflate automatically
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",  # requests handles these automatically
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    # Log crawler fingerprint if debug mode is enabled
    if DEBUG_CRAWLER:
        logger.info("=" * 80)
        logger.info("[CRAWLER DIAGNOSTICS] Crawler Fingerprint:")
        logger.info(f"  User-Agent: {headers.get('User-Agent')}")
        logger.info(f"  Accept: {headers.get('Accept')}")
        logger.info(f"  Accept-Language: {headers.get('Accept-Language')}")
        logger.info(f"  Accept-Encoding: {headers.get('Accept-Encoding')}")
        logger.info(f"  Referer: {headers.get('Referer', 'None (not set)')}")
        logger.info(f"  Cookies: None (no session/cookie handling)")
        logger.info(f"  IP/Proxy: Not configured (using system default)")
        logger.info(f"  Crawler Type: {describe_crawler()}")
        logger.info("=" * 80)
    
    try:
        logger.debug(f"Making request to {url} with headers: {headers.get('User-Agent')}")
        response = requests.get(
        url,
            timeout=_REQUEST_TIMEOUT,
            headers=headers,
            allow_redirects=True,
            stream=False,  # Don't stream, get full response
        )
        
        # Capture final URL after redirects
        final_url = response.url
        redirect_count = len(response.history)
        
        logger.info(f"Received response from {url}: Status {response.status_code}")
        
        if DEBUG_CRAWLER:
            logger.info("=" * 80)
            logger.info("[CRAWLER DIAGNOSTICS] Request Details:")
            logger.info(f"  Original URL: {url}")
            logger.info(f"  Final URL (after redirects): {final_url}")
            logger.info(f"  Redirect count: {redirect_count}")
            logger.info(f"  HTTP Status Code: {response.status_code}")
            logger.info("=" * 80)
        
        response.raise_for_status()
        
        # Log response details for debugging
        content_length = len(response.content)
        content_type = response.headers.get('Content-Type', 'unknown')
        detected_encoding = response.encoding or "unknown"
        logger.info(f"Response from {url}: Status {response.status_code}, Content-Length: {content_length}, Content-Type: {content_type}, Encoding: {detected_encoding}")
        
        if DEBUG_CRAWLER:
            logger.info("=" * 80)
            logger.info("[CRAWLER DIAGNOSTICS] Response Details:")
            logger.info(f"  HTTP Status Code: {response.status_code}")
            logger.info(f"  Final Resolved URL: {final_url}")
            logger.info(f"  Response Content-Length: {content_length} bytes")
            logger.info(f"  Response Content-Type: {content_type}")
            logger.info(f"  Detected Encoding: {detected_encoding}")
            logger.info(f"  Response Body Empty: {content_length == 0}")
            logger.info(f"  Response Body Blocked: {content_length == 0 and response.status_code == 200}")
            
            # Log all response headers for analysis
            logger.info("  Response Headers:")
            for header_name, header_value in response.headers.items():
                logger.info(f"    {header_name}: {header_value}")
            logger.info("=" * 80)
        
        # Check if response is suspiciously small (might be SPA/JavaScript)
        if content_length < 1000:
            logger.warning(f"Response from {url} is very small ({content_length} chars) - might be a JavaScript SPA or blocked")
            if DEBUG_CRAWLER:
                logger.warning(f"[CRAWLER DIAGNOSTICS] Small content detected - possible causes:")
                logger.warning(f"  - JavaScript SPA (content loaded client-side)")
                logger.warning(f"  - Bot detection returning empty/minimal response")
                logger.warning(f"  - Site requires authentication/session")
                logger.warning(f"  - Site blocks non-browser user agents")
        
        # Better encoding detection and handling
        if not response.encoding:
            # Try to detect encoding from Content-Type header
            if 'charset=' in content_type:
                try:
                    charset = content_type.split('charset=')[1].split(';')[0].strip().strip('"\'')
                    response.encoding = charset
                    logger.debug(f"Detected encoding from Content-Type: {charset}")
                except Exception:
                    pass
        
        # If still no encoding, try to detect from content (optional chardet)
        if not response.encoding:
            try:
                import chardet
                detected = chardet.detect(response.content)
                if detected and detected.get('encoding'):
                    response.encoding = detected['encoding']
                    logger.debug(f"Detected encoding from content: {detected['encoding']} (confidence: {detected.get('confidence', 0):.2f})")
            except ImportError:
                logger.debug("chardet not available, will use UTF-8 fallback")
            except Exception as e:
                logger.debug(f"Encoding detection failed: {e}")
        
        # Fallback to UTF-8 if still no encoding
        if not response.encoding:
            response.encoding = 'utf-8'
            logger.debug("Using UTF-8 as fallback encoding")
        
        # Get text and validate it's not garbled
        text = response.text
        text_length = len(text)
        
        # Check if text looks garbled (too many non-printable or invalid characters)
        if text_length > 0:
            # Sample first 2000 chars to check for garbled content
            sample = text[:2000] if text_length > 2000 else text
            # Count printable characters (ASCII printable + common Unicode)
            printable_count = sum(1 for c in sample if (32 <= ord(c) <= 126) or c.isspace() or (ord(c) > 127 and c.isprintable()))
            printable_ratio = printable_count / len(sample) if len(sample) > 0 else 0
            
            logger.debug(f"Content validation: {printable_count}/{len(sample)} printable characters ({printable_ratio:.2%})")
            
            if printable_ratio < 0.3:  # Less than 30% printable - likely garbled
                logger.warning(f"Response from {url} appears garbled ({printable_ratio:.1%} printable). Trying alternative decoding...")
                
                # Try different encoding approaches
                encoding_attempts = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for enc in encoding_attempts:
                    try:
                        test_text = response.content.decode(enc, errors='strict')
                        # Check if this encoding produces better results
                        test_sample = test_text[:2000] if len(test_text) > 2000 else test_text
                        test_printable = sum(1 for c in test_sample if (32 <= ord(c) <= 126) or c.isspace() or (ord(c) > 127 and c.isprintable()))
                        test_ratio = test_printable / len(test_sample) if len(test_sample) > 0 else 0
                        
                        if test_ratio > printable_ratio:
                            text = test_text
                            response.encoding = enc
                            logger.info(f"Found better encoding: {enc} ({test_ratio:.1%} printable)")
                            break
                    except (UnicodeDecodeError, Exception):
                        continue
                
                # If still garbled, try with error replacement
                if printable_ratio < 0.3:
                    try:
                        text = response.content.decode('utf-8', errors='replace')
                        logger.warning("Using UTF-8 with error replacement - some characters may be lost")
                    except Exception as e:
                        logger.error(f"Failed to decode content: {e}")
                        # Last resort: return raw bytes as string (will be garbled but at least we have something)
                        text = str(response.content)
        
        # Final validation - check if we got reasonable HTML structure
        if text and len(text) > 100:
            # Check for basic HTML tags
            has_html_tags = any(tag in text.lower() for tag in ['<html', '<body', '<div', '<head', '<title'])
            if not has_html_tags:
                logger.warning(f"Response from {url} doesn't appear to contain HTML tags - might be JSON, plain text, or still garbled")
        
        return text
        
    except requests.exceptions.Timeout as e:
        logger.error(f"Request to {url} timed out after {_REQUEST_TIMEOUT}s. This could indicate:")
        logger.error("  - Site is very slow or overloaded")
        logger.error("  - Site is actively blocking/throttling the connection")
        logger.error("  - Network/firewall issues")
        logger.error(f"  - Full error: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to {url} failed: {type(e).__name__}: {e}")
        raise


def _strip_unused_tags(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()


def _first_heading_text(soup: BeautifulSoup) -> Optional[str]:
    heading = soup.find("h1")
    if heading:
        text = heading.get_text(strip=True)
        return text or None
    return None


def _safe_text(element: Any) -> Optional[str]:
    if element is None:
        return None
    if hasattr(element, "get_text"):
        text = element.get_text(strip=True)
        return text or None
    if isinstance(element, str):
        text = element.strip()
        return text or None
    return None


def _extract_text(soup: BeautifulSoup) -> str:
    text = soup.get_text(separator=" ", strip=True)
    # Collapse multiple whitespaces to single spaces.
    normalized = " ".join(text.split())
    return normalized


def _guess_page_type(soup: BeautifulSoup, text: Optional[str]) -> str:
    text_lower = (text or "").lower()

    pricing_indicators = ["pricing", "price", "plans", "cost", "subscription"]
    if _contains_pricing_table(soup) or any(indicator in text_lower for indicator in pricing_indicators):
        return "Product"

    article_indicators = ["blog", "article", "read more", "published", "by ", " minutes read"]
    if soup.find("article") or any(indicator in text_lower for indicator in article_indicators):
        return "Article"

    paragraph_count = len(soup.find_all("p"))
    heading_count = len(soup.find_all(["h1", "h2", "h3"]))
    if paragraph_count > 5 and heading_count > 2:
        return "Article"

    return "Organization"


def _contains_pricing_table(soup: BeautifulSoup) -> bool:
    for table in soup.find_all("table"):
        table_text = table.get_text(separator=" ", strip=True).lower()
        if "price" in table_text or "plan" in table_text or "pricing" in table_text:
            return True
        class_list = " ".join(table.get("class", [])).lower()
        if "pricing" in class_list or "plans" in class_list:
            return True
    return False


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "your",
    "have",
    "will",
    "about",
    "into",
    "there",
    "such",
    "they",
    "their",
    "these",
    "those",
    "what",
    "when",
    "where",
    "which",
    "while",
    "whose",
    "would",
    "could",
    "should",
    "been",
    "were",
    "also",
    "them",
    "then",
    "than",
    "over",
    "upon",
    "through",
    "within",
    "between",
    "after",
    "before",
    "each",
    "other",
    "more",
    "most",
    "some",
    "many",
    "much",
    "very",
    "just",
    "like",
    "because",
    "while",
    "using",
    "under",
    "only",
    "every",
    "able",
    "make",
    "made",
    "does",
    "doesn",
    "don",
    "cant",
    "cannot",
    "here",
    "back",
    "page",
    "home",
    "contact",
    "login",
    "signup",
    "subscribe",
    "learn",
    "know",
    "info",
    "information",
}


def _guess_keywords(text: Optional[str]) -> List[str]:
    if not text:
        return []

    words = _tokenize(text)
    filtered_words = [word for word in words if word not in _STOPWORDS]
    frequency = Counter(filtered_words)
    most_common = [word for word, _count in frequency.most_common(7)]

    # Keep 3-7 keywords, remove duplicates while preserving order.
    unique_keywords: List[str] = []
    for word in most_common:
        if word not in unique_keywords and len(unique_keywords) < 7:
            unique_keywords.append(word)

    return unique_keywords[: max(3, min(len(unique_keywords), 7))]


def _tokenize(text: str) -> Iterable[str]:
    for match in re.finditer(r"\b[a-zA-Z]{3,}\b", text.lower()):
        yield match.group(0)


async def _generate_page_purpose_summary(text: Optional[str]) -> str:
    if not text:
        return ""

    return await generate_llm_summary(text, short=False)


async def _generate_ai_summary(text: Optional[str]) -> str:
    if not text:
        return ""

    return await generate_llm_summary(text, short=True)

