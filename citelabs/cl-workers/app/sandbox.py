"""
AEO/GEO Sandbox Evaluation Engine
Implements the full 7-step evaluation flow
"""
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from collections import Counter, defaultdict  # New imports
import re  # New import
try:
    from gliner import GLiNER  # New import
except ImportError:
    GLiNER = None  # Handle missing dependency gracefully

import uuid

import requests
import httpx
from fastapi.concurrency import run_in_threadpool

from . import crawl
from .llm_providers.custom_providers import get_multi_provider_llm
from .embeddings_gemini import embed_documents
from .vector_pinecone import PineconeVectorStore
from .chunkers.factory import get_chunker
from .chunkers import Document as ChunkerDocument
from .llm_providers.factory import get_llm_provider
from .constants import get_model_for_step, get_model_for_role
logger = logging.getLogger(__name__)

# Backend URL for chunk persistence
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:4000")


def _sanitize_text(text: Optional[str]) -> Optional[str]:
    """
    Remove null bytes from text (PostgreSQL cannot store them in UTF-8 text fields).
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text with null bytes removed, or None if input is None/empty
    """
    if not text:
        return None
    # Remove null bytes (0x00) which PostgreSQL cannot store
    return text.replace('\x00', '')


# ==========================================
# Domain Classification & Weighting (P2 - Brand Boost)
# ==========================================

# Domain weights for RAG retrieval (boost brand pages over editorial)
DOMAIN_WEIGHTS = {
    "sandbox_brand": 1.4,      # Highest priority: sandbox brand pages
    "business_competitor": 1.2, # Second priority: direct business competitors
    "editorial": 0.8,          # Lower priority: blogs, reviews, listicles
}


def classify_domain_type(
    url: str,
    sandbox_domain: Optional[str] = None,
    business_competitor_domains: Optional[List[str]] = None,
) -> str:
    """
    Classify a domain as sandbox_brand, business_competitor, or editorial.
    This is a lightweight, deterministic classification (no LLM).
    
    Args:
        url: URL to classify
        sandbox_domain: Sandbox domain (e.g., "zomato.com")
        business_competitor_domains: List of business competitor domains (e.g., ["swiggy.com", "ubereats.com"])
        
    Returns:
        One of: "sandbox_brand", "business_competitor", "editorial"
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        
        # Normalize sandbox domain for comparison
        if sandbox_domain:
            sandbox_domain_normalized = sandbox_domain.lower().replace("www.", "")
            if domain == sandbox_domain_normalized:
                return "sandbox_brand"
        
        # Check against business competitor domains
        if business_competitor_domains:
            for comp_domain in business_competitor_domains:
                comp_domain_normalized = comp_domain.lower().replace("www.", "")
                if domain == comp_domain_normalized:
                    return "business_competitor"
        
        # Default to editorial (blogs, reviews, etc.)
        return "editorial"
    except Exception as e:
        logger.warning(f"Error classifying domain type for {url}: {e}")
        # Default to editorial on error
        return "editorial"


def _calculate_category_alignment(questions: List[Dict[str, str]], category: str) -> int:
    """
    Calculate how many questions align with the business category.
    
    Args:
        questions: List of question dicts
        category: Business category string
        
    Returns:
        Number of aligned questions (0-15)
    """
    if not category or not questions:
        return 0
    
    category_lower = category.lower()
    category_keywords = set()
    
    # Extract meaningful keywords from category
    # Remove common words and extract business-specific terms
    stopwords = {"and", "or", "the", "a", "an", "for", "with", "on", "in", "at", "to", "of"}
    words = re.findall(r'\b\w+\b', category_lower)
    for word in words:
        if len(word) > 3 and word not in stopwords:
            category_keywords.add(word)
    
    # Also check for specific patterns
    if "food delivery" in category_lower or "restaurant marketplace" in category_lower:
        category_keywords.update(["delivery", "restaurant", "food", "order", "app"])
    if "subscription" in category_lower:
        category_keywords.update(["subscription", "monthly", "plan"])
    if "marketplace" in category_lower:
        category_keywords.update(["marketplace", "platform", "compare"])
    
    aligned_count = 0
    exclusion_keywords = set()
    
    # Define exclusion patterns based on category
    if "food delivery" in category_lower or "restaurant marketplace" in category_lower:
        exclusion_keywords = {"meal kit", "subscription", "prepared meal", "frozen", "weekly delivery"}
    
    for q_dict in questions:
        question = q_dict.get("q", "").lower()
        
        # Check for exclusion keywords (negative alignment)
        has_exclusion = any(excl in question for excl in exclusion_keywords)
        if has_exclusion:
            continue  # Don't count as aligned
        
        # Check for category keywords (positive alignment)
        has_category_keyword = any(kw in question for kw in category_keywords if len(kw) > 3)
        if has_category_keyword:
            aligned_count += 1
    
    return aligned_count


# SERP API Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERP_API_ENDPOINT = os.getenv("SERP_API_ENDPOINT", "https://serpapi.com/search")
_BLOCKED_DOMAINS = {
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "linkedin.com",
    "tiktok.com",
    "youtube.com",
    "pinterest.com",
    "reddit.com",
    "wikipedia.org",
    "cnn.com",
    "nytimes.com",
    "bbc.com",
    "foxnews.com",
    "theguardian.com",
    "reuters.com",
    "bloomberg.com",
}
_BLOCKED_EXTENSIONS = (".pdf", ".doc", ".docx", ".ppt", ".pptx")

# Initialize Pinecone
_pinecone_store: Optional[PineconeVectorStore] = None


def _get_pinecone_store() -> PineconeVectorStore:
    """Get or create Pinecone store instance."""
    global _pinecone_store
    if _pinecone_store is None:
        _pinecone_store = PineconeVectorStore()
    return _pinecone_store


# ==========================================
# STEP 1.5: Domain Semantic Summary (NEW)
# ==========================================


def _log_llm_call(
    step_name: str,
    requested_model: Optional[str],
    resolved_model: str,
    provider_name: str,
    requested_role: Optional[str] = None,
) -> None:
    """
    Structured logging for LLM usage with model verification.
    
    Args:
        step_name: Pipeline step name (e.g., "intent_extraction")
        requested_model: Model name requested (may be None)
        resolved_model: Actual model used
        provider_name: Provider name (e.g., "Gemini 2.5 Flash")
        requested_role: Optional semantic role (e.g., "CATEGORY_REASONING")
    """
    role_info = f" requested_role={requested_role}" if requested_role else ""
    logger.info(
        f"[llm_call] step={step_name}{role_info} requested_model={requested_model or 'default'} "
        f"resolved_model={resolved_model} provider={provider_name}"
    )
    
    # Verify model routing - warn if Flash is used for reasoning steps
    if requested_role and requested_role != "FAST_EXTRACTION":
        if "flash" in resolved_model.lower() or "gemini-2.5-flash" in resolved_model.lower():
            logger.warning(
                f"[model_verification] WARNING: {requested_role} step is using Flash model ({resolved_model}). "
                f"This may indicate model routing issue. Expected reasoning model."
            )
    
    # Warn if Gemini is unavailable but was requested
    if requested_model and "gemini" in requested_model.lower() and "gemini" not in provider_name.lower():
        logger.warning(
            f"[model_verification] WARNING: Gemini model requested ({requested_model}) but provider is {provider_name}. "
            f"Check GEMINI_API_KEY and Gemini provider availability."
        )


# Initialize GLiNER model globally (lazy loaded if needed, but simple global for now)
# We use a try/except block to avoid crashing if dependencies are missing during dev
gliner_model = None

def _get_gliner_model():
    global gliner_model
    if gliner_model is None and GLiNER:
        try:
            logger.info("Loading GLiNER model 'urchade/gliner_small'...")
            gliner_model = GLiNER.from_pretrained("urchade/gliner_small")
            logger.info("GLiNER model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load GLiNER model: {e}")
    return gliner_model

# --- GLiNER Helper Functions ---
REMOVE_SUFFIXES = {"inc", "llc", "ltd", "corp", "co", "gmbh", "plc", "billing", "solutions", "systems", "technologies", "software"}
PUNCT_RE = re.compile(r'^[\W_]+|[\W_]+$')

def normalize_name(name: str) -> str:
    """Basic normalization: strip, trim punctuation, lower, remove common suffix tokens."""
    if not name:
        return ""
    n = name.strip()
    n = PUNCT_RE.sub("", n)
    parts = [p for p in re.split(r'[\s\-_/,&]+', n) if p]
    # drop suffix tokens like "Inc", "Ltd" heuristically
    if parts and parts[-1].lower() in REMOVE_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts).strip().lower()

def whole_word_count(text: str, phrase: str) -> int:
    # allow phrase to contain spaces, punctuation handled by escaping
    if not phrase:
        return 0
    pat = re.compile(rf'(?<!\w){re.escape(phrase)}(?!\w)', re.IGNORECASE)
    return len(pat.findall(text))

def canonicalize_candidates(raw_candidates: List[str]) -> Dict[str, List[str]]:
    """
    Merge variants by substring heuristic.
    Returns mapping canonical -> list(aliases)
    """
    norm_map = {}
    for r in raw_candidates:
        norm_map.setdefault(normalize_name(r), []).append(r)

    norms = list(norm_map.keys())
    merged = {}
    used = set()

    # sort norms by length ascending
    norms_sorted = sorted(norms, key=len)
    for n in norms_sorted:
        if n in used:
            continue
        canonical = n
        merged[canonical] = set([n])
        used.add(n)
        for m in norms_sorted:
            if m in used:
                continue
            if canonical in m or m in canonical:
                merged[canonical].add(m)
                used.add(m)

    result = {}
    for canon, alias_norms in merged.items():
        aliases_raw = []
        for an in alias_norms:
            aliases_raw.extend(norm_map.get(an, []))
        seen = []
        for a in aliases_raw:
            if a not in seen:
                seen.append(a)
        result[canon] = seen
    return result

def extract_competitors_gliner_improved(text: str, brand: str, threshold: float = 0.5) -> Dict:
    model = _get_gliner_model()
    if not model:
        logger.warning("GLiNER model not available, returning empty competitor metrics.")
        return {
            "brand": brand,
            "brand_mentions": whole_word_count(text, brand),
            "competitors_detected": [],
            "competitor_counts": {},
            "total_competitor_mentions": 0,
            "unique_competitors_count": 0,
        }

    brand_norm = normalize_name(brand)

    # safe GLiNER call
    try:
        ents = model.predict_entities(text, labels=["ORG", "PRODUCT"], threshold=threshold)
    except Exception as exc:
        logger.error(f"GLiNER prediction failed: {exc}")
        ents = []

    raw_candidates = []
    for ent in ents:
        if ent["label"] != "ORG":     # skip product names
            continue
        tok = ent["text"].strip()
        if not tok or len(tok) < 2:
            continue
        # skip brand matches (normalized)
        if normalize_name(tok) == brand_norm:
            continue
        raw_candidates.append(tok)

    # dedupe raw candidates preserving first occurrence order
    seen_raw = []
    seen_norm = set()
    for r in raw_candidates:
        rn = normalize_name(r)
        if rn and rn not in seen_norm:
            seen_norm.add(rn)
            seen_raw.append(r)

    # canonicalize / merge variants
    canon_map = canonicalize_candidates(seen_raw)

    # compute counts per canonical
    canon_counts = {}
    for canon_norm, aliases in canon_map.items():
        total = 0
        for alias in aliases:
            total += whole_word_count(text, alias)
            # Only count the space-stripped version if the alias actually has spaces
            if " " in alias:
                total += whole_word_count(text, alias.replace(" ", ""))
        canon_counts[canon_norm] = total

    # prepare output
    out_competitors = []
    counts_display = {}
    for canon_norm, aliases in canon_map.items():
        display = aliases[0] if aliases else canon_norm.title()
        out_competitors.append(display)
        counts_display[display] = canon_counts.get(canon_norm, 0)

    return {
        "brand": brand,
        "brand_mentions": whole_word_count(text, brand),
        "competitors_detected": out_competitors,
        "competitor_counts": counts_display,
        "total_competitor_mentions": sum(counts_display.values()),
        "unique_competitors_count": sum(1 for v in counts_display.values() if v > 0),
        # "debug_raw_gliner_entities": ents, # Omitted to save space
    }


async def generate_domain_summary(
    page_data: Dict[str, Any],
    model_name: Optional[str] = None,
) -> str:
    """
    Generate a concise, factual domain semantic summary (2-3 sentences or 3-5 bullets).
    This summary is reused across downstream LLM steps to reduce token usage.
    
    Args:
        page_data: Crawled page data with title, content, etc.
        model_name: Optional model override (uses MODEL_MAP if None)
        
    Returns:
        Concise domain summary string (2-3 sentences or 3-5 bullets)
    """
    # Use Flash model for simple summarization task
    if model_name is None:
        model_name = get_model_for_role("FAST_EXTRACTION")  # Use FAST_EXTRACTION for crawl summarization
    llm = get_llm_provider(model_name)
    
    # Log LLM call
    _log_llm_call(
        step_name="domain_summary_generation",
        requested_model=model_name,
        resolved_model=model_name,
        provider_name=llm.get_provider_name(),
    )
    
    title = page_data.get("title", "")
    h1 = page_data.get("h1", "")
    full_content = page_data.get("full_content", "")
    
    # Use first 4000 chars for efficiency (summary doesn't need full content)
    content_sample = full_content[:4000] if full_content else ""
    
    prompt = (
        "You are a factual content analyzer. Generate a concise, factual summary of this website's core business and purpose.\n\n"
        "REQUIREMENTS:\n"
        "• 2-3 sentences OR 3-5 bullet points\n"
        "• Focus on facts only - no marketing language, no opinions\n"
        "• Describe what the business does, who it serves, and its core value proposition\n"
        "• Be specific and concrete\n"
        "• Avoid generic phrases like 'leading platform' or 'best solution'\n\n"
        f"Page Title: {title}\n"
        f"Page H1: {h1}\n"
        f"Page Content (sample): {content_sample}\n\n"
        "Return ONLY the summary. No explanations, no markdown formatting."
    )
    
    # Summary generation is concise - lower token limit
    max_tokens_options = [300, 500, 800]
    summary = None
    last_error = None
    
    for max_tokens in max_tokens_options:
        try:
            summary = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            if summary and summary.strip():
                summary = summary.strip()
                break
            else:
                logger.warning(f"Empty summary response with max_tokens={max_tokens}, retrying with higher limit")
        except ValueError as e:
            error_str = str(e)
            last_error = e
            is_max_tokens_error = (
                "MAX_TOKENS" in error_str or 
                "max_tokens" in error_str.lower() or
                "max_tokens limit" in error_str.lower()
            )
            if is_max_tokens_error and max_tokens < max_tokens_options[-1]:
                next_max = max_tokens_options[max_tokens_options.index(max_tokens) + 1]
                logger.warning(f"Hit max_tokens limit with {max_tokens}, retrying with {next_max}")
                continue
            else:
                raise
        except Exception as e:
            logger.error(f"Error generating domain summary: {e}")
            raise
    
    if not summary or not summary.strip():
        if last_error:
            raise last_error
        raise ValueError("Failed to generate domain summary: Empty response after all retries")
    
    logger.info(f"Generated domain summary (length: {len(summary)})")
    return summary.strip()


# ==========================================
# STEP 1-2: Analyze Sandbox Page
# ==========================================


async def classify_business_category(
    page_data: Dict[str, Any],
    page_intent: str,
    domain_summary: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    NEW STEP: Classify business into a single primary category.
    This grounds the simulation before question generation.
    
    Args:
        page_data: Crawled page data with title, content, etc.
        page_intent: Extracted page intent
        domain_summary: Optional pre-generated domain summary (reduces token usage)
        model_name: Optional model override (uses MODEL_MAP if None)
        
    Returns:
        Single primary business category string
    """
    # Use CATEGORY_REASONING model (NOT Flash)
    if model_name is None:
        model_name = get_model_for_role("CATEGORY_REASONING")
    llm = get_llm_provider(model_name)
    
    # Verify provider actually supports the requested model
    actual_provider = llm.get_provider_name()
    # Check if Gemini model was requested but provider doesn't match
    if "gemini" in model_name.lower() and "gemini" not in actual_provider.lower():
        logger.error(
            f"[model_verification] ERROR: Category classification requested Gemini model ({model_name}) "
            f"but got provider {actual_provider}. Falling back may cause quality issues."
        )
    
    # Log LLM call with role
    _log_llm_call(
        step_name="category_classification",
        requested_model=model_name,
        resolved_model=model_name,
        provider_name=actual_provider,
        requested_role="CATEGORY_REASONING",
    )
    
    title = page_data.get("title", "")
    h1 = page_data.get("h1", "")
    
    # Prefer domain_summary over raw content to reduce tokens
    content_source = domain_summary if domain_summary else page_data.get("full_content", "")[:2000]
    content_label = "Domain Summary" if domain_summary else "Page Content (sample)"
    
    prompt = (
        "You are a business analyst classifying websites into specific business categories.\n\n"
        "CONTEXT:\n"
        "Assume the user is located in India.\n"
        "Prefer Indian brands, pricing models, delivery patterns, and regulations.\n\n"
        "TASK: Classify this website into ONE primary business category.\n\n"
        "REQUIREMENTS:\n"
        "• Return ONLY the category name - be specific, not generic\n"
        "• Focus on core business model and user intent\n"
        "• Avoid generic labels like 'SaaS' or 'E-commerce'\n"
        "• Include business model specifics (e.g., 'on-demand', 'subscription', 'marketplace')\n"
        "• Include target user type if relevant (e.g., 'B2B', 'consumer', 'enterprise')\n"
        "• Include geography if relevant (e.g., 'hyperlocal', 'regional', 'India')\n\n"
        "EXAMPLES:\n"
        "• 'Food delivery & restaurant marketplace (on-demand, consumer, hyperlocal, India)'\n"
        "• 'Prepared meal subscription service (consumer, weekly delivery)'\n"
        "• 'SaaS testing platform (B2B, browser automation)'\n"
        "• 'Online learning platform (B2C, course marketplace)'\n"
        "• 'Real estate listing platform (marketplace, residential sales)'\n\n"
        f"Page Title: {title}\n"
        f"Page H1: {h1}\n"
        f"Page Intent: {page_intent}\n"
        f"{content_label}: {content_source}\n\n"
        "Return ONLY the category name. No explanations, no markdown, no quotes."
    )
    
    logger.info("[region_context] region=India")
    
    # Reduced token limit since we're using summary
    # Start at 600 (reduced from 800) with 1000 as backup when using summary
    max_tokens_options = [600, 1000] if domain_summary else [800, 1200, 1500]
    category = None
    last_error = None
    
    for max_tokens in max_tokens_options:
        try:
            category = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            if category and category.strip():
                category = category.strip().strip('"\'')  # Remove quotes if present
                break
            else:
                logger.warning(f"Empty category response with max_tokens={max_tokens}, retrying with higher limit")
        except ValueError as e:
            error_str = str(e)
            last_error = e
            is_max_tokens_error = (
                "MAX_TOKENS" in error_str or 
                "max_tokens" in error_str.lower() or
                "max_tokens limit" in error_str.lower()
            )
            if is_max_tokens_error and max_tokens < max_tokens_options[-1]:
                next_max = max_tokens_options[max_tokens_options.index(max_tokens) + 1]
                logger.warning(f"Hit max_tokens limit with {max_tokens}, retrying with {next_max}")
                continue
            else:
                raise
        except Exception as e:
            logger.error(f"Error classifying business category: {e}")
            raise
    
    if not category or not category.strip():
        if last_error:
            raise last_error
        raise ValueError("Failed to classify business category: Empty response after all retries")
    
    category = category.strip()
    
    # Log category lock (CRITICAL for downstream steps)
    logger.info(f"[category_classification] locked_category = \"{category}\"")
    
    return category


async def analyze_sandbox_page(
    url: str,
    domain_summary: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Step 1-2: Crawl sandbox page and extract page intent using Gemini 2.5 Flash.
    
    Args:
        url: URL to analyze
        domain_summary: Optional pre-generated domain summary (reduces token usage)
        model_name: Optional model override (uses MODEL_MAP if None)

    Returns:
        Dict with page data and extracted intent
    """
    # Step 1: Crawl the page
    page_data = await crawl.crawl_url(url)
    
    if "error" in page_data:
        return {"error": page_data["error"]}

    full_text = page_data.get("full_content", "")
    
    # Step 2: Extract page intent using LLM with fallback
    # Use FAST_EXTRACTION model for intent extraction (can use Flash)
    if model_name is None:
        model_name = get_model_for_role("FAST_EXTRACTION")
    llm = get_llm_provider(model_name)
    
    # Log LLM call with role
    _log_llm_call(
        step_name="intent_extraction",
        requested_model=model_name,
        resolved_model=model_name,
        provider_name=llm.get_provider_name(),
        requested_role="FAST_EXTRACTION",
    )
    
    # Prefer domain_summary over raw content to reduce tokens
    content_source = domain_summary if domain_summary else full_text[:4000]
    content_label = "Domain Summary" if domain_summary else "Page Content"
    
    # Updated Prompt: Request JSON Output with Inteet and Page Brand Name
    intent_prompt = (
        "You are an AEO/GEO analysis engine. Your task is to extract *generic user intent patterns* from a webpage, focusing only on how the ideal audience thinks and searches. Do NOT generate business-specific phrasing. Do NOT describe what the business claims about itself. Identify the underlying goals, problems, desires, questions, and decision-making triggers of the target audience.\n"
        "ALSO: Identify the specific brand name of the company/website from the content.\n\n"
        "RETURN YOUR RESPONSE AS A VALID JSON OBJECT ONLY. NO MARKDOWN. NO EXPLANATION.\n"
        "Format:\n"
        "{\n"
        '  "intent": "YOUR_INTENT_ANALYSIS_HERE",\n'
        '  "page_brand_name": "BRAND_NAME_HERE"\n'
        "}\n\n"
        "Analyze this webpage and extract the *general underlying user intent* behind it. Focus on:\n"
        "• Real goals of the target audience\n"
        "• Problems the audience is trying to solve\n"
        "• What they would search for in AI/LLM systems\n"
        "• Jobs-to-be-done\n"
        "• Travel/use-case motivations\n"
        "• Pain points and decision criteria\n"
        "• Experience expectations\n"
        "• Buying or booking triggers\n"
        "\n"
        "Do NOT include business names in the 'intent' field.\n"
        "Do NOT describe the company's features directly in the 'intent' field.\n"
        "Interpret the page from the perspective of:\n"
        "\"What type of person would look for a page like this, and what would they ask an AI assistant?\"\n"
        f"Page Title: {page_data.get('title', 'N/A')}\n"
        f"Page H1: {page_data.get('h1', 'N/A')}\n"
        f"{content_label}:\n{content_source}\n"
    )

    # Reduced token limit since we're using summary instead of full content
    # Optimized: Start at 1200 (reduced from 1500) with 1800 as backup when using summary
    max_tokens_options = [1200, 1800] if domain_summary else [1500, 2000]
    intent_str = None
    intent_json = {}
    last_error = None
    
    for max_tokens in max_tokens_options:
        try:
            raw_response = await llm.chat(
                messages=[{"role": "user", "content": intent_prompt}],
                max_tokens=max_tokens,
            )
            # Try to clean markdown code blocks if present
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            import json
            try:
                intent_json = json.loads(cleaned_response)
                intent_str = intent_json.get("intent", "").strip()
                if intent_str:
                    break  # Success
            except json.JSONDecodeError:
                # Fallback: Treat entire response as intent if JSON parsing fails
                logger.warning("Failed to parse intent JSON, falling back to raw text")
                intent_str = raw_response.strip()
                intent_json = {"intent": intent_str, "page_brand_name": "Unknown"}
                break

            if not intent_str:
                 logger.warning(f"Empty intent response with max_tokens={max_tokens}, retrying with higher limit")

        except ValueError as e:
            error_str = str(e)
            last_error = e
            # Check if it's a MAX_TOKENS or safety filter issue
            # Check for both "MAX_TOKENS" and "max_tokens" (case-insensitive)
            is_max_tokens_error = (
                "MAX_TOKENS" in error_str or 
                "max_tokens" in error_str.lower() or
                "max_tokens limit" in error_str.lower() or
                "generated" in error_str.lower() and "tokens but no text" in error_str.lower()
            )
            
            if is_max_tokens_error and max_tokens < max_tokens_options[-1]:
                next_max = max_tokens_options[max_tokens_options.index(max_tokens) + 1]
                logger.warning(f"Hit max_tokens limit with {max_tokens}, retrying with {next_max}")
                continue  # Try next higher max_tokens
            elif "safety" in error_str.lower() or "SAFETY" in error_str:
                # Safety filter - don't retry, just raise
                logger.error(f"Intent extraction blocked by safety filters: {e}")
                raise
            else:
                # Other error - don't retry
                raise
    
    # If we still don't have intent after all retries
    if not intent_str or not intent_str.strip():
        if last_error:
            raise last_error
        raise ValueError("Failed to extract intent: Empty response after all retries")

    page_brand_name = intent_json.get("page_brand_name", "")

    return {
        "url": url,
        "title": page_data.get("title"),
        "h1": page_data.get("h1"),
        "intent": intent_str.strip(),
        "page_brand_name": page_brand_name.strip() if page_brand_name else None,  # NEW
        "page_summary": page_data.get("page_purpose_summary"),
        "ai_summary": page_data.get("ai_summary"),
        "full_text": full_text,
    }


# ==========================================
# STEP 3: Generate 15 User Questions
# ==========================================


async def generate_user_questions(
    page_intent: str,
    business_category: str = None,
    domain_summary: Optional[str] = None,
    model_name: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Step 3: Generate 15 real-world user questions (5 intent, 5 experience, 5 transaction).
    Now category-conditioned to ensure questions match the actual business.

    Args:
        page_intent: Extracted page intent
        business_category: Classified business category (NEW - grounds question generation)
        domain_summary: Optional pre-generated domain summary (reduces token usage)
        model_name: Optional model override (uses MODEL_MAP if None)

    Returns:
        List of question dicts with 'type' and 'q' keys
    """
    # Use QUESTION_GENERATION model (NOT Flash)
    if model_name is None:
        model_name = get_model_for_role("QUESTION_GENERATION")
    llm = get_llm_provider(model_name)
    
    # Verify provider
    actual_provider = llm.get_provider_name()
    # Check if Gemini model was requested but provider doesn't match
    if "gemini" in model_name.lower() and "gemini" not in actual_provider.lower():
        logger.error(
            f"[model_verification] ERROR: Question generation requested Gemini model ({model_name}) "
            f"but got provider {actual_provider}."
        )
    
    # Log LLM call with role
    _log_llm_call(
        step_name="question_generation",
        requested_model=model_name,
        resolved_model=model_name,
        provider_name=actual_provider,
        requested_role="QUESTION_GENERATION",
    )
    
    logger.info("[region_context] region=India")

    # Build category-conditioned prompt with hard exclusions
    category_context = ""
    category_exclusions = ""
    
    if business_category:
        category_lower = business_category.lower()
        
        # Generate category-specific exclusions
        if "food delivery" in category_lower or "restaurant marketplace" in category_lower or "on-demand" in category_lower:
            category_exclusions = (
                "\nHARD EXCLUSIONS (DO NOT generate questions about these):\n"
                "• Meal kits\n"
                "• Weekly subscriptions\n"
                "• Prepared meals\n"
                "• Frozen food delivery\n"
                "• Meal planning services\n"
                "• Grocery delivery (unless explicitly part of food delivery)\n"
            )
        elif "subscription" in category_lower and "meal" in category_lower:
            category_exclusions = (
                "\nHARD EXCLUSIONS (DO NOT generate questions about these):\n"
                "• On-demand food delivery\n"
                "• Restaurant ordering\n"
                "• Real-time delivery apps\n"
            )
        
        category_context = (
            f"\nCRITICAL: Business Category: {business_category}\n\n"
            "CATEGORY ALIGNMENT RULES:\n"
            "• ALL questions MUST match this exact business category\n"
            "• Exclude questions about adjacent but incorrect categories\n"
            "• Questions must reflect how users compare similar platforms in THIS category\n"
            f"{category_exclusions}"
        )
    
    prompt = (
        "You are an AI assistant that generates highly realistic user questions for AEO/GEO simulation.\n\n"
        "CONTEXT:\n"
        "Assume the user is located in India.\n"
        "Prefer Indian brands, pricing models (INR), delivery patterns, and regulations.\n"
        "Questions should reflect Indian market context (e.g., km-based delivery, Indian dining habits).\n\n"
        "Your job is to produce the SAME TYPE of questions real people ask AI systems like ChatGPT, Gemini, Perplexity, and Siri when they are exploring a product category, a service category, a problem they want solved, or a decision they are trying to make — BEFORE they know any specific brand exists.\n\n"
        "ABSOLUTE RULES:\n"
        "• NEVER include the business name.\n"
        "• NEVER indirectly describe the business (e.g., \"the company on this page…\").\n"
        "• ALL questions must be category-level only.\n"
        "• Questions must reflect real human phrasing, not SEO-style or blog-style prompts.\n"
        "• Questions should be natural, conversational, and often imperfect or contextual, just like real user queries.\n"
        "• Questions should reflect Indian market context (e.g., 'Which app delivers fastest in my area?', 'What are the delivery charges in Mumbai?')\n"
        f"{category_context}"
        "WHAT TO GENERATE:\n"
        "Using the intent summary below, generate EXACTLY 15 questions across three categories:\n\n"
        "1. Intent Questions (5)\n"
        "   – Discovery questions about the category, available options, what to look for, how to choose, or understanding the landscape.\n\n"
        "2. Experience Questions (5)\n"
        "   – Questions about features, capabilities, quality, comparisons, performance, suitability for specific needs, or expected outcomes.\n\n"
        "3. Transaction Questions (5)\n"
        "   – Questions about pricing, availability, cost structures, booking/sign-up logistics, hidden fees, value, or decision-making factors.\n\n"
        "STYLE REQUIREMENTS (Industry-Agnostic):\n"
        "• Make questions specific, not generic or vague.\n"
        "• Use real-world scenarios based on the intent summary (e.g., small teams, families, high-budget buyers, safety concerns, feature needs, performance issues, convenience factors, etc.).\n"
        "• Avoid robotic or templated phrasing.\n"
        "• Questions MUST adapt to whatever domain the intent summary implies (travel, SaaS, healthcare, finance, e-commerce, home services, real estate, etc.).\n\n"
        f"INPUT:\n"
        f"Page Intent:\n{page_intent}\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON array. No markdown, no code blocks, no explanations.\n"
        "The JSON must be valid and parseable. Escape any quotes in question text with backslash.\n\n"
        "Example format:\n"
        '[{"type": "intent", "q": "What are the best options for..."}, {"type": "experience", "q": "How does this work?"}, {"type": "transaction", "q": "How much does it cost?"}]\n\n'
        "CRITICAL: Return ONLY the JSON array, nothing else. Ensure all quotes in question text are properly escaped."
    )

    # Question generation needs significant tokens - prompt is ~500-600 tokens,
    # and we need to generate 15 questions in JSON format (~1500-2000 tokens)
    # Try with increasing max_tokens if we get empty responses
    # Optimized: Start at 2000 (already optimal) with 2500 as backup
    max_tokens_options = [2000, 2500]
    response = None
    last_error = None
    
    for max_tokens in max_tokens_options:
        try:
            response = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            if response and response.strip():
                # Check if JSON appears complete (has closing bracket)
                # If not, it might be truncated and we should retry with higher max_tokens
                if max_tokens < max_tokens_options[-1]:
                    # Quick check: does the response end with a closing bracket?
                    response_stripped = response.strip()
                    if not response_stripped.endswith(']'):
                        # Might be truncated - check if we can parse at least some questions
                        try:
                            import json
                            # Try to find complete objects
                            first_bracket = response_stripped.find('[')
                            if first_bracket != -1:
                                # Count complete objects by counting closing braces
                                complete_braces = response_stripped.count('},')
                                if complete_braces < 14:  # We need 15 questions, so at least 14 complete ones
                                    logger.warning(f"Response appears truncated (only {complete_braces} complete questions), retrying with higher max_tokens")
                                    continue  # Retry with higher max_tokens
                        except:
                            pass
                
                break  # Success, exit retry loop
            else:
                logger.warning(f"Empty question response with max_tokens={max_tokens}, retrying with higher limit")
        except ValueError as e:
            error_str = str(e)
            last_error = e
            # Check if it's a MAX_TOKENS issue and we can retry
            if "MAX_TOKENS" in error_str or "max_tokens" in error_str.lower():
                if max_tokens < max_tokens_options[-1]:
                    next_max = max_tokens_options[max_tokens_options.index(max_tokens) + 1]
                    logger.warning(f"Hit max_tokens limit with {max_tokens}, retrying with {next_max}")
                    continue  # Try next higher max_tokens
                else:
                    # Already at max, raise
                    raise
            elif "safety" in error_str.lower() or "SAFETY" in error_str:
                # Safety filter - don't retry, just raise
                logger.error(f"Question generation blocked by safety filters: {e}")
                raise
            else:
                # Other error - don't retry
                raise
    
    # If we still don't have response after all retries
    if not response or not response.strip():
        if last_error:
            raise last_error
        raise ValueError("Failed to generate questions: Empty response after all retries")

    # Parse JSON from response with robust error handling
    import json
    
    # Log full response for debugging
    logger.debug(f"LLM response for questions (first 1000 chars): {response[:1000]}")
    
    try:
        # Step 1: Try to extract JSON from markdown code blocks
        json_text = response.strip()
        
        # Remove markdown code blocks if present
        markdown_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if markdown_match:
            json_text = markdown_match.group(1)
            logger.debug("Extracted JSON from markdown code block")
        
        # Step 2: Try to find JSON array in response
        if not json_text.startswith('['):
            json_match = re.search(r'\[.*?\]', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                logger.debug("Extracted JSON array using regex")
        
        # Step 3: Try to fix common JSON issues
        # Remove trailing commas before closing brackets/braces
        json_text = re.sub(r',\s*\]', ']', json_text)
        json_text = re.sub(r',\s*\}', '}', json_text)
        
        # Step 4: Parse JSON
        questions = json.loads(json_text.strip())
        
        # Validate format
        if not isinstance(questions, list):
            logger.warning(f"Questions response is not a list, got: {type(questions)}")
            return []
            
        validated = []
        for q in questions:
            if isinstance(q, dict) and "type" in q and "q" in q:
                if q["type"] in ["intent", "experience", "transaction"]:
                    validated.append({"type": q["type"], "q": str(q["q"])})
        
        if len(validated) < 15:
            logger.warning(f"Only generated {len(validated)} valid questions, expected 15")
        
        # Validate category alignment
        if business_category and validated:
            alignment_score = _calculate_category_alignment(validated, business_category)
            logger.info(f"[question_generation] category_alignment_score = {alignment_score}/15")
            
            if alignment_score < 10.5:  # Less than 70% (10.5/15)
                logger.warning(
                    f"[question_generation] Low category alignment ({alignment_score}/15). "
                    f"Category: {business_category}. Some questions may drift from business category."
                )
        
        return validated[:15]  # Ensure max 15
        
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse questions JSON: {exc}")
        logger.error(f"JSON error at position {exc.pos}: {exc.msg}")
        logger.error(f"Response (first 2000 chars): {response[:2000]}")
        
        # Check if the error is due to truncated JSON (unterminated string)
        if "Unterminated string" in str(exc.msg) or exc.pos > 0:
            # Try to fix truncated JSON by finding complete objects
            logger.info("Attempting to fix truncated JSON by extracting complete question objects")
            try:
                # Find all complete JSON objects (those that end with })
                # Use a more robust regex to find complete objects
                complete_objects = []
                bracket_count = 0
                current_obj = ""
                in_string = False
                escape_next = False
                
                # Find the start of the array
                start_idx = response.find('[')
                if start_idx == -1:
                    raise ValueError("No array start found")
                
                # Parse character by character to find complete objects
                for i, char in enumerate(response[start_idx + 1:], start=start_idx + 1):
                    if escape_next:
                        escape_next = False
                        current_obj += char
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        current_obj += char
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        current_obj += char
                        continue
                    
                    current_obj += char
                    
                    if not in_string:
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                # Complete object found
                                try:
                                    obj = json.loads(current_obj.strip().rstrip(','))
                                    if isinstance(obj, dict) and "type" in obj and "q" in obj:
                                        complete_objects.append(obj)
                                except:
                                    pass
                                current_obj = ""
                
                # If we found complete objects, use them
                if complete_objects:
                    validated = []
                    for q in complete_objects:
                        if q["type"] in ["intent", "experience", "transaction"]:
                            validated.append({"type": q["type"], "q": str(q["q"])})
                    if validated:
                        logger.info(f"Successfully extracted {len(validated)} complete questions from truncated JSON")
                        return validated[:15]
            except Exception as fix_exc:
                logger.debug(f"Failed to fix truncated JSON: {fix_exc}")
        
        # Try one more time with a simpler extraction
        try:
            # Find the first [ and last ] and try to parse that
            first_bracket = response.find('[')
            last_bracket = response.rfind(']')
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                simple_json = response[first_bracket:last_bracket + 1]
                logger.debug(f"Trying simple extraction: {simple_json[:500]}")
                questions = json.loads(simple_json)
                if isinstance(questions, list):
                    validated = []
                    for q in questions:
                        if isinstance(q, dict) and "type" in q and "q" in q:
                            if q["type"] in ["intent", "experience", "transaction"]:
                                validated.append({"type": q["type"], "q": str(q["q"])})
                    if validated:
                        logger.info(f"Successfully parsed {len(validated)} questions using simple extraction")
                        return validated[:15]
        except Exception as retry_exc:
            logger.error(f"Retry parsing also failed: {retry_exc}")
            
    except Exception as exc:
        logger.error(f"Unexpected error parsing questions: {exc}")
        logger.error(f"Response (first 1000 chars): {response[:1000]}")

    # Fallback: return empty list
    logger.error("Failed to parse questions, returning empty list")
    return []


# ==========================================
# STEP 4: Extract Competitors (Hybrid: SERP + LLM)
# ==========================================


async def generate_serp_queries(
    page_intent: str,
    sandbox_title: str = None,
    sandbox_url: str = None,
    model_name: Optional[str] = None,
) -> List[str]:
    """
    Generate 2-3 strategic SERP API search queries based on page intent.
    
    Args:
        page_intent: The extracted page intent from analyze_sandbox_page
        sandbox_title: Optional page title for context
        sandbox_url: Optional URL for context
        
    Returns:
        List of 1-3 search query strings for SERP API
    """
    # Use COMPETITOR_REASONING model for SERP query generation
    if model_name is None:
        model_name = get_model_for_role("COMPETITOR_REASONING")
    llm = get_llm_provider(model_name)
    
    # Verify provider
    actual_provider = llm.get_provider_name()
    # Check if Gemini model was requested but provider doesn't match
    if "gemini" in model_name.lower() and "gemini" not in actual_provider.lower():
        logger.error(
            f"[model_verification] ERROR: SERP query generation requested Gemini model ({model_name}) "
            f"but got provider {actual_provider}."
        )
    
    # Log LLM call with role
    _log_llm_call(
        step_name="competitor_discovery_serp_queries",
        requested_model=model_name,
        resolved_model=model_name,
        provider_name=actual_provider,
        requested_role="COMPETITOR_REASONING",
    )
    
    logger.info("[region_context] region=India")
    
    prompt = (
        "You are an SEO/GEO expert creating strategic Google search queries to find competitors.\n\n"
        "Your task is to generate 10 highly effective search queries that would help discover "
        "the top competitors and alternatives in the same market/category.\n\n"
        "GUIDELINES:\n"
        "• Generate queries that real users would type into Google when searching for alternatives/competitors\n"
        "• Use natural, conversational search language (not keyword-stuffed)\n"
        "• Include location-specific terms if the intent suggests a local service (e.g., hotels, restaurants, services)\n"
        "• Include category/industry terms from the intent\n"
        "• Include qualifiers like 'best', 'top', 'alternatives', 'vs', 'comparison' when appropriate\n"
        "• For SaaS/software: Include terms like 'platform', 'software', 'tool', 'solution'\n"
        "• For local services: Include location names, nearby landmarks, specific features\n"
        "• NEVER include the specific business name or brand from the sandbox page\n"
        "• Focus on category-level searches that would surface competitors\n\n"
        "QUERY TYPES TO COVER (Make sure to generate 10 total):\n"
        "1. **Best/Top Alternatives Query**: 'best [category] alternatives' or 'top [category] software'\n"
        "2. **Category Search Query**: '[category] platform' or '[category] solutions'\n"
        "3. **Feature/Location-Specific Query**: '[feature] [category]' or '[location] [category]' (if applicable)\n"
        "4. **Comparison/Review Queries**: '[category] reviews' or '[category] comparison'\n"
        "5. **Long-tail Problem Queries**: 'how to solve [problem] with software'\n\n"
        "EXAMPLES:\n\n"
        "For Billing/Subscription Management:\n"
        "- 'best subscription billing software alternatives'\n"
        "- 'enterprise billing and revenue management platforms'\n"
        "- 'recurring billing software comparison'\n"
        "- 'top rated subscription management tools'\n\n"
        "For Hotels in Krabi:\n"
        "- 'best hotels in Krabi near Ao Nang beach'\n"
        "- 'family friendly hotels in Krabi Ao Nang'\n"
        "- 'luxury resorts Krabi beachfront'\n"
        "- 'resorts in Krabi with private pool'\n\n"
        "For E-commerce Platforms:\n"
        "- 'best e-commerce platform alternatives'\n"
        "- 'SaaS e-commerce solutions'\n"
        "- 'online store builder comparison'\n"
        "- 'top enterprise ecommerce software'\n\n"
        f"PAGE INTENT:\n{page_intent}\n\n"
    )
    
    # if sandbox_title:
    #     prompt += f"SANDBOX PAGE TITLE (for context only, DO NOT include in queries): {sandbox_title}\n\n"
    
    prompt += (
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON array with 10 search queries, like this:\n\n"
        '[\n'
        '  "best subscription billing software alternatives",\n'
        '  "enterprise revenue management platforms",\n'
        '  "recurring billing software comparison",\n'
        '  "top rated subscription management tools",\n'
        '  "billing automation software for saas",\n'
        '  "subscription management platform reviews",\n'
        '  "recurring payment processing solutions",\n'
        '  "best b2b subscription billing software",\n'
        '  "automated revenue recognition software",\n'
        '  "cloud based billing platform comparison"\n'
        ']\n\n'
        "Do not add explanations, markdown, or text outside the JSON array.\n"
        "Return ONLY the JSON array with 10 search query strings."
    )
    
    # SERP query generation needs more tokens - prompt can be 500+ tokens
    # We need to generate 10 short query strings, so 2000-2500 should be enough
    # Optimized: Start at 2500 (recommended) with 3000 as backup
    max_tokens_options = [2500, 3000]
    response = None
    last_error = None
    
    for max_tokens in max_tokens_options:
        try:
            response = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            if response and response.strip():
                # Check if JSON appears complete (has closing bracket)
                # If not, it might be truncated and we should retry with higher max_tokens
                if max_tokens < max_tokens_options[-1]:
                    response_stripped = response.strip()
                    if not response_stripped.endswith(']'):
                        # Might be truncated - try to extract what we can, but also retry
                        try:
                            # Count complete strings (those that end with ", or ])
                            # Simple check: count quotes that are properly closed
                            quote_count = response_stripped.count('"')
                            if quote_count % 2 != 0:
                                # Unclosed quote - definitely truncated
                                logger.warning(f"SERP query response appears truncated (unclosed quote), retrying with higher max_tokens")
                                continue
                            # Try to parse - if it fails, retry
                            first_bracket = response_stripped.find('[')
                            if first_bracket != -1:
                                # Try to extract complete strings
                                try:
                                    # Find all complete string values (re is imported at top of file)
                                    string_matches = re.findall(r'"([^"]+)"', response_stripped)
                                    if len(string_matches) < 2:  # We want at least 2 queries
                                        logger.warning(f"SERP query response appears truncated (only {len(string_matches)} complete queries), retrying with higher max_tokens")
                                        continue
                                except Exception:
                                    pass
                        except Exception:
                            pass
                
                break  # Success, exit retry loop
            else:
                logger.warning(f"Empty SERP query response with max_tokens={max_tokens}, retrying with higher limit")
        except ValueError as e:
            error_str = str(e)
            last_error = e
            # Check if it's a MAX_TOKENS issue and we can retry
            is_max_tokens_error = (
                "MAX_TOKENS" in error_str or 
                "max_tokens" in error_str.lower() or
                "max_tokens limit" in error_str.lower()
            )
            if is_max_tokens_error and max_tokens < max_tokens_options[-1]:
                next_max = max_tokens_options[max_tokens_options.index(max_tokens) + 1]
                logger.warning(f"Hit max_tokens limit with {max_tokens}, retrying with {next_max}")
                continue  # Try next higher max_tokens
            else:
                # Other error or already at max - raise
                raise
    
    # If we still don't have response after all retries
    if not response or not response.strip():
        if last_error:
            raise last_error
        raise ValueError("Failed to generate SERP queries: Empty response after all retries")
    
    try:
        # Step 1: Try to extract JSON from markdown code blocks
        # json is imported at top of file
        json_text = response.strip()
        
        # Remove markdown code blocks if present
        markdown_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if markdown_match:
            json_text = markdown_match.group(1)
            logger.debug("Extracted JSON from markdown code block")
        
        # Step 2: Try to find JSON array in response
        if not json_text.startswith('['):
            json_match = re.search(r'\[.*?\]', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                logger.debug("Extracted JSON array using regex")
        
        # Step 3: Try to fix common JSON issues
        # Remove trailing commas before closing brackets
        json_text = re.sub(r',\s*\]', ']', json_text)
        json_text = re.sub(r',\s*\}', '}', json_text)
        
        # Step 4: Parse JSON
        queries = json.loads(json_text.strip())
        
        # Validate and clean
        if not isinstance(queries, list):
            logger.warning("SERP queries response is not a list, using fallback")
            return _fallback_serp_queries(page_intent, sandbox_title)
        
        validated_queries = []
        for q in queries:
            if isinstance(q, str) and len(q.strip()) > 10:
                validated_queries.append(q.strip())
        
        # Ensure we have at least 1 query, max 3
        if len(validated_queries) < 1:
            fallback = _fallback_serp_queries(page_intent, sandbox_title)
            validated_queries.extend(fallback)
        
        return validated_queries[:10]  # Max 10 queries
        
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse SERP queries JSON: {exc}")
        logger.error(f"JSON error at position {exc.pos}: {exc.msg}")
        logger.error(f"Response (first 1000 chars): {response[:1000]}")
        
        # Check if the error is due to truncated JSON (unterminated string or missing closing bracket)
        if "Unterminated string" in str(exc.msg) or "Expecting value" in str(exc.msg) or exc.pos > 0:
            # Try to extract complete query strings from truncated JSON
            logger.info("Attempting to extract complete query strings from truncated JSON")
            try:
                # Extract all complete string values using regex (re is imported at top of file)
                # Find all complete string matches (quoted strings)
                string_matches = re.findall(r'"([^"]+)"', response)
                if string_matches:
                    validated_queries = []
                    for q in string_matches:
                        q_clean = q.strip()
                        if len(q_clean) > 10:  # Valid query length
                            validated_queries.append(q_clean)
                    
                    if validated_queries:
                        logger.info(f"Successfully extracted {len(validated_queries)} complete queries from truncated JSON: {validated_queries}")
                        # Ensure we have at least 1, max 3
                        if len(validated_queries) < 1:
                            fallback = _fallback_serp_queries(page_intent, sandbox_title)
                            validated_queries.extend(fallback)
                        return validated_queries[:10]
            except Exception as extract_exc:
                logger.debug(f"Failed to extract queries from truncated JSON: {extract_exc}")
        
        # Try one more time with a simpler extraction
        try:
            first_bracket = response.find('[')
            last_bracket = response.rfind(']')
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                simple_json = response[first_bracket:last_bracket + 1]
                queries = json.loads(simple_json)
                if isinstance(queries, list):
                    validated = [q.strip() for q in queries if isinstance(q, str) and len(q.strip()) > 10]
                    if validated:
                        logger.info(f"Successfully parsed {len(validated)} SERP queries using simple extraction")
                        return validated[:10]
        except Exception as retry_exc:
            logger.debug(f"Retry parsing also failed: {retry_exc}")
        
        # Fallback to generated queries
        logger.warning("Using fallback SERP queries due to parsing failure")
        return _fallback_serp_queries(page_intent, sandbox_title)
    except Exception as exc:
        logger.error(f"Unexpected error generating SERP queries: {exc}")
        logger.error(f"Response (first 500 chars): {response[:500]}")
        return _fallback_serp_queries(page_intent, sandbox_title)


def _fallback_serp_queries(page_intent: str, sandbox_title: str = None) -> List[str]:
    """
    Fallback function to generate basic SERP queries if LLM fails.
    """
    queries = []
    intent_lower = page_intent.lower()
    
    # Extract category/keywords from intent
    if "billing" in intent_lower or "subscription" in intent_lower:
        queries.append("best subscription billing software alternatives")
        queries.append("recurring billing platform comparison")
    elif "hotel" in intent_lower or "resort" in intent_lower:
        if "krabi" in intent_lower:
            queries.append("best hotels in Krabi")
            queries.append("Krabi beachfront hotels")
        else:
            queries.append("best hotels")
    elif "e-commerce" in intent_lower or "ecommerce" in intent_lower:
        queries.append("best e-commerce platform alternatives")
        queries.append("online store builder comparison")
    else:
        # Generic fallback
        queries.append("best alternatives")
    
    return queries[:3]  # Max 3


def _serpapi_search(query: str, num_results: int = 10) -> List[str]:
    """
    Search Google via SERP API and return competitor URLs.
    
    Args:
        query: Search query string
        num_results: Number of results to return
        
    Returns:
        List of competitor URLs
    """
    if not SERPAPI_KEY:
        logger.warning("SERPAPI_KEY not set, skipping SERP search")
        return []
    
    try:
        response = requests.get(
            SERP_API_ENDPOINT,
            params={"q": query, "engine": "google", "num": num_results, "api_key": SERPAPI_KEY},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        
        organic_results = data.get("organic_results", []) or []
        competitors = []
        
        for item in organic_results:
            link = item.get("link")
            if not link:
                continue
            
            if not _is_valid_competitor_url(link):
                continue
            
            if link not in competitors:
                competitors.append(link)
            
            if len(competitors) >= num_results:
                break
        
        return competitors
        
    except Exception as exc:
        logger.error(f"SERP API search failed for query '{query}': {exc}")
        return []


def _is_valid_competitor_url(link: str) -> bool:
    """Validate if URL is a valid competitor URL."""
    try:
        parsed = urlparse(link)
        if not parsed.scheme.startswith("http"):
            return False
        
        domain = parsed.netloc.lower()
        if any(blocked in domain for blocked in _BLOCKED_DOMAINS):
            return False
        
        path_lower = parsed.path.lower()
        if path_lower.endswith(_BLOCKED_EXTENSIONS):
            return False
        
        return True
    except Exception:
        return False


async def extract_direct_competitors(
    business_category: str,
    page_intent: str = None,
    sandbox_title: str = None,
    sandbox_url: str = None,
    model_name: Optional[str] = None,
) -> List[str]:
    """
    NEW: Layer A - Extract direct business competitors.
    Finds companies offering the same core service to the same users.
    
    Args:
        business_category: Classified business category
        page_intent: Page intent for context
        sandbox_title: Optional page title
        sandbox_url: Optional URL
        model_name: Optional model override (uses MODEL_MAP if None)
        
    Returns:
        List of competitor domain names (e.g., ['swiggy.com', 'ubereats.com'])
    """
    # Use COMPETITOR_REASONING model (NOT Flash)
    if model_name is None:
        model_name = get_model_for_role("COMPETITOR_REASONING")
    llm = get_llm_provider(model_name)
    
    # Verify provider
    actual_provider = llm.get_provider_name()
    # Check if Gemini model was requested but provider doesn't match
    if "gemini" in model_name.lower() and "gemini" not in actual_provider.lower():
        logger.error(
            f"[model_verification] ERROR: Competitor reasoning requested Gemini model ({model_name}) "
            f"but got provider {actual_provider}."
        )
    
    # Log LLM call with role
    _log_llm_call(
        step_name="competitor_discovery_direct",
        requested_model=model_name,
        resolved_model=model_name,
        provider_name=actual_provider,
        requested_role="COMPETITOR_REASONING",
    )
    
    logger.info("[region_context] region=India")
    
    prompt = (
        "You are a business analyst identifying direct business competitors.\n\n"
        "CONTEXT:\n"
        "Assume the user is located in India.\n"
        "PREFER Indian brands: Swiggy, Zomato, Dunzo, Zepto > Uber Eats, DoorDash (US-only).\n"
        "If a US-only competitor appears, mark it as non_primary_region_competitor.\n\n"
        "TASK: List direct business competitors offering the SAME core service "
        "to the SAME users in the SAME geography (India).\n\n"
        "CRITICAL RULES:\n"
        "• Include ONLY companies/platforms that directly compete as businesses\n"
        "• These are actual competitors users would compare when making a decision\n"
        "• Exclude blogs, review sites, aggregators, and editorial content\n"
        "• Exclude adjacent but different business models\n"
        "• Return company/domain names, not URLs\n"
        "• Return 5-10 direct business competitors\n"
        "• PRIORITIZE Indian competitors (Swiggy, Zomato, Dunzo, Zepto) over international ones\n\n"
        f"Business Category: {business_category}\n"
        f"Page Intent: {page_intent or 'N/A'}\n"
        f"Page Title: {sandbox_title or 'N/A'}\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON array of competitor names/domains, like:\n"
        '["Competitor1", "competitor2.com", "Competitor 3"]\n\n'
        "Return ONLY the JSON array. No explanations, no markdown."
    )
    
    # Direct competitor extraction - start higher to avoid MAX_TOKENS issues
    max_tokens_options = [1500, 2500, 3000]
    response = None
    last_error = None
    
    for max_tokens in max_tokens_options:
        try:
            response = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            if response and response.strip():
                break
            else:
                logger.warning(f"Empty direct competitor response with max_tokens={max_tokens}, retrying")
        except ValueError as e:
            error_str = str(e)
            last_error = e
            is_max_tokens_error = (
                "MAX_TOKENS" in error_str or 
                "max_tokens" in error_str.lower() or
                "max_tokens limit" in error_str.lower()
            )
            if is_max_tokens_error and max_tokens < max_tokens_options[-1]:
                next_max = max_tokens_options[max_tokens_options.index(max_tokens) + 1]
                logger.warning(f"Hit max_tokens limit with {max_tokens}, retrying with {next_max}")
                continue
            else:
                raise
        except Exception as e:
            logger.error(f"Error extracting direct competitors: {e}")
            raise
    
    if not response or not response.strip():
        if last_error:
            raise last_error
        raise ValueError("Failed to extract direct competitors: Empty response")
    
    # Parse JSON response
    try:
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            competitors = json.loads(json_match.group(0))
        else:
            competitors = json.loads(response.strip())
        
        if not isinstance(competitors, list):
            logger.warning("Direct competitors response is not a list")
            return []
        
        # Clean and validate
        validated = []
        for comp in competitors:
            if isinstance(comp, str) and comp.strip():
                validated.append(comp.strip())
        
        logger.info(f"Extracted {len(validated)} direct competitors: {validated[:5]}...")
        return validated[:10]  # Max 10
        
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse direct competitors JSON: {exc}")
        logger.error(f"Response: {response[:500]}")
        return []


async def extract_competitors_from_questions(
    questions: List[Dict[str, str]],
    page_intent: str = None,
    business_category: str = None,
    sandbox_title: str = None,
    sandbox_url: str = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Step 4: Hybrid competitor extraction (SERP API + LLM answers).
    NOW SPLIT INTO TWO LAYERS:
    - Layer A: Direct Business Competitors (NEW)
    - Layer B: Authority & Influencer Sources (EXISTING - blogs, reviews, etc.)
    
    Phase 1: Extract direct competitors using business category
    Phase 2: Generate 1-3 strategic SERP queries and search for authority sources
    Phase 3: Extract authority sources from LLM answers to all 15 questions
    Phase 4: Combine, deduplicate, and filter out sandbox domain
    
    Returns:
        Dict with:
        - competitor_urls: List of dicts with 'url', 'type' ('direct' | 'authority'), and 'source' ('llm' | 'serp' | 'direct_extraction')
        - serp_queries: List of SERP queries generated (for storing as questions)
    """
    # New structure: url -> {"type": "direct"|"authority", "source": "llm"|"serp"|"direct_extraction"}
    all_urls_dict = {}  # Dict mapping url -> dict with type and source
    sandbox_domain = None
    serp_queries = []
    
    # Extract sandbox domain for filtering
    if sandbox_url:
        try:
            sandbox_domain = urlparse(sandbox_url).netloc.lower().replace("www.", "")
        except Exception:
            pass
    
    # ==========================================
    # LAYER A: Direct Business Competitors (NEW)
    # ==========================================
    # OLD SEQUENTIAL IMPLEMENTATION (Commented out for parallel refactor)
    # logger.info("Layer A: Extracting direct business competitors")
    # direct_competitor_names = []
    # try:
    #     if business_category:
    #         direct_competitor_names = await extract_direct_competitors(
    #             business_category=business_category,
    #             page_intent=page_intent,
    #             sandbox_title=sandbox_title,
    #             sandbox_url=sandbox_url,
    #             model_name=model_name,
    #         )
    #         
    #         # Convert competitor names to URLs (try common patterns)
    #         for comp_name in direct_competitor_names:
    #             # Try to construct URL from name
    #             comp_name_clean = comp_name.lower().replace(" ", "").replace(".com", "").replace(".", "")
    #             # Try www.{name}.com pattern
    #             potential_urls = [
    #                 f"https://www.{comp_name_clean}.com",
    #                 f"https://{comp_name_clean}.com",
    #                 f"https://www.{comp_name}.com" if "." not in comp_name else f"https://{comp_name}",
    #             ]
    #             
    #             # For now, store the name - we'll try to resolve to URL later or use in SERP
    #             # Mark as direct competitor type
    #             for url in potential_urls:
    #                 if _is_valid_competitor_url(url):
    #                     if sandbox_domain:
    #                         try:
    #                             url_domain = urlparse(url).netloc.lower().replace("www.", "")
    #                             if url_domain == sandbox_domain:
    #                                 continue
    #                         except Exception:
    #                             pass
    #                     all_urls_dict[url] = {"type": "business_competitor", "source": "direct_extraction"}
    #                     break
    #         
    #         business_competitor_count = len([u for u, d in all_urls_dict.items() if d.get("type") == "business_competitor"])
    #         logger.info(f"[competitor_extraction] business_competitors = {business_competitor_count}")
    # except Exception as exc:
    #     logger.error(f"Layer A (Direct Competitors) failed: {exc}, continuing with authority sources")
    # 
    # # ==========================================
    # # LAYER B: Authority & Influencer Sources (EXISTING)
    # # PHASE 1: Strategic SERP API Queries (1-3 queries)
    # # ==========================================
    # logger.info("Phase 1: Generating strategic SERP queries")
    # try:
    #     if page_intent:
    #         serp_queries = await generate_serp_queries(
    #             page_intent=page_intent,
    #             sandbox_title=sandbox_title,
    #             sandbox_url=sandbox_url, 
    #             model_name=model_name
    #         )
    #         
    #         logger.info(f"Generated {len(serp_queries)} SERP queries: {serp_queries}")
    #         
    #         # Run SERP API searches in parallel
    #         serp_tasks = [
    #             run_in_threadpool(_serpapi_search, query, num_results=10)
    #             for query in serp_queries
    #         ]
    #         serp_results = await asyncio.gather(*serp_tasks, return_exceptions=True)
    #         
    #         # Collect SERP competitors with source_type='serp'
    #         for result in serp_results:
    #             if isinstance(result, Exception):
    #                 logger.warning(f"SERP search error: {result}")
    #                 continue
    #             if isinstance(result, list):
    #                 for url in result:
    #                     # Filter out sandbox domain
    #                     if sandbox_domain:
    #                         try:
    #                             url_domain = urlparse(url).netloc.lower().replace("www.", "")
    #                             if url_domain == sandbox_domain:
    #                                 continue
    #                         except Exception:
    #                             pass
    #                     # Mark as authority source from SERP (blogs, reviews, etc.)
    #                     all_urls_dict[url] = {"type": "authority_source", "source": "serp"}
    #         
    #         authority_source_count = len([u for u, d in all_urls_dict.items() if d.get("type") == "authority_source"])
    #         logger.info(f"[competitor_extraction] authority_sources = {authority_source_count}")
    #     
    # except Exception as exc:
    #     logger.error(f"Phase 1 (SERP) failed: {exc}, continuing with LLM extraction only")
    # 
    # # ==========================================
    # # PHASE 2: Extract from LLM Answers (all 15 questions)
    # # ==========================================
    # logger.info("Phase 2: Extracting authority sources from LLM answers")
    # # Use COMPETITOR_REASONING model for LLM answer extraction
    # if model_name is None:
    #     model_name = get_model_for_role("COMPETITOR_REASONING")
    # llm = get_llm_provider(model_name)
    # 
    # # Verify provider
    # actual_provider = llm.get_provider_name()
    # # Check if Gemini model was requested but provider doesn't match
    # if "gemini" in model_name.lower() and "gemini" not in actual_provider.lower():
    #     logger.error(
    #         f"[model_verification] ERROR: Competitor LLM answer extraction requested Gemini model ({model_name}) "
    #         f"but got provider {actual_provider}."
    #     )
    # 
    # # Log LLM call with role
    # _log_llm_call(
    #     step_name="competitor_discovery_llm_answers",
    #     requested_model=model_name,
    #     resolved_model=model_name,
    #     provider_name=actual_provider,
    #     requested_role="COMPETITOR_REASONING",
    # )
    # 
    # logger.info("[region_context] region=India")
    # 
    # for question_dict in questions:
    #     question = question_dict.get("q", "")
    #     if not question:
    #         continue
    # 
    #     try:
    #         # Ask LLM normally (no RAG)
    #         # Competitor extraction needs more tokens for complete answers
    #         # Increased to handle long answers that might hit MAX_TOKENS
    #         max_tokens_options = [3000, 4000, 5000]
    #         answer = None
    #         last_error = None
    #         
    #         for max_tokens in max_tokens_options:
    #             try:
    #                 answer = await llm.chat(
    #                     messages=[
    #                         {
    #                             "role": "user",
    #                             "content": question,
    #                         }
    #                     ],
    #                     max_tokens=max_tokens,
    #                 )
    #                 if answer and answer.strip():
    #                     break  # Success, exit retry loop
    #                 else:
    #                     logger.warning(f"Empty competitor answer with max_tokens={max_tokens}, retrying with higher limit")
    #             except ValueError as e:
    #                 error_str = str(e)
    #                 last_error = e
    #                 # Check if it's a MAX_TOKENS issue and we can retry
    #                 is_max_tokens_error = (
    #                     "MAX_TOKENS" in error_str or 
    #                     "max_tokens" in error_str.lower() or
    #                     "max_tokens limit" in error_str.lower()
    #                 )
    #                 if is_max_tokens_error and max_tokens < max_tokens_options[-1]:
    #                     next_max = max_tokens_options[max_tokens_options.index(max_tokens) + 1]
    #                     logger.warning(f"Hit max_tokens limit with {max_tokens}, retrying with {next_max}")
    #                     continue  # Try next higher max_tokens
    #                 else:
    #                     # If we've exhausted all max_tokens options, log and continue to next question
    #                     if is_max_tokens_error:
    #                         logger.error(f"All max_tokens options exhausted for question '{question[:100]}...'. Skipping this question.")
    #                         last_error = None  # Reset to allow continue
    #                         break  # Exit retry loop, will continue to next question
    #                     else:
    #                         # Other error - raise
    #                         raise
    #         
    #         # If we still don't have answer after all retries
    #         if not answer or not answer.strip():
    #             if last_error:
    #                 logger.error(f"Error extracting competitors for question '{question[:100]}...': {last_error}. Skipping this question.")
    #                 continue  # Skip this question and continue with next
    #             logger.warning(f"Failed to get answer for question '{question}': Empty response after all retries")
    #             continue  # Skip this question and continue with next
    # 
    #         # Check if citations present
    #         has_citations = bool(re.search(r'\[.*?\]|http|www\.|\.com|\.org', answer, re.IGNORECASE))
    # 
    #         if not has_citations:
    #             # Ask again for source URLs
    #             source_prompt = (
    #                 f"List the 5-10 most likely source URLs that informed your previous answer to: {question}\n\n"
    #                 "Return only URLs, one per line."
    #             )
    #             # Source URL extraction needs enough tokens for multiple URLs (5-10 URLs)
    #             # Increased to handle cases where model generates long responses
    #             max_tokens_options = [2000, 3000, 4000]
    #             source_response = None
    #             last_source_error = None
    #             for max_tokens in max_tokens_options:
    #                 try:
    #                     source_response = await llm.chat(
    #                         messages=[{"role": "user", "content": source_prompt}],
    #                         max_tokens=max_tokens,
    #                     )
    #                     if source_response and source_response.strip():
    #                         break  # Success
    #                     else:
    #                         logger.warning(f"Empty source URL response with max_tokens={max_tokens}, retrying with higher limit")
    #                 except ValueError as e:
    #                     error_str = str(e)
    #                     last_source_error = e
    #                     is_max_tokens_error = (
    #                         "MAX_TOKENS" in error_str or 
    #                         "max_tokens" in error_str.lower() or
    #                         "max_tokens limit" in error_str.lower()
    #                     )
    #                     if is_max_tokens_error and max_tokens < max_tokens_options[-1]:
    #                         next_max = max_tokens_options[max_tokens_options.index(max_tokens) + 1]
    #                         logger.warning(f"Hit max_tokens limit with {max_tokens}, retrying with {next_max}")
    #                         continue  # Retry with higher max_tokens
    #                     else:
    #                         # Other error or already at max - raise
    #                         raise
    #             
    #             if source_response and source_response.strip():
    #                 answer = source_response
    #             else:
    #                 logger.warning(f"Failed to get source URLs for question '{question}', using original answer")
    # 
    #         # Extract URLs from answer
    #         urls = _extract_urls_from_text(answer)
    #         
    #         # Filter out sandbox domain and mark as LLM source
    #         for url in urls:
    #             if sandbox_domain:
    #                 try:
    #                     url_domain = urlparse(url).netloc.lower().replace("www.", "")
    #                     if url_domain == sandbox_domain:
    #                         continue
    #                 except Exception:
    #                     pass
    #             # Mark as authority source from LLM (blogs, reviews, etc.)
    #             # Only add if not already present (business competitors take precedence)
    #             if url not in all_urls_dict:
    #                 all_urls_dict[url] = {"type": "authority_source", "source": "llm"}
    # 
    #     except Exception as exc:
    #         logger.warning(f"Error extracting competitors for question '{question}': {exc}")
    #         continue
    # 
    #     await asyncio.sleep(10)
    # 
    # logger.info(f"Phase 2: Total competitors after LLM extraction: {len(all_urls_dict)}")

    # ==========================================
    # NEW PARALLEL IMPLEMENTATION (Phases 1, 2, 3)
    # ==========================================
    
    # Define inner async functions for each phase to separate logic and state
    
    async def _extract_layer_a_direct() -> List[tuple]:
        """Task for Layer A: Direct Business Competitors"""
        # ==========================================
        # LAYER A: Direct Business Competitors (NEW)
        # ==========================================
        logger.info("Layer A: Extracting direct business competitors (Parallel)")
        local_results = []
        try:
            if business_category:
                direct_competitor_names = await extract_direct_competitors(
                    business_category=business_category,
                    page_intent=page_intent,
                    sandbox_title=sandbox_title,
                    sandbox_url=sandbox_url,
                    model_name=model_name,
                )
                
                # Convert competitor names to URLs (try common patterns)
                for comp_name in direct_competitor_names:
                    # Try to construct URL from name
                    comp_name_clean = comp_name.lower().replace(" ", "").replace(".com", "").replace(".", "")
                    # Try www.{name}.com pattern
                    potential_urls = [
                        f"https://www.{comp_name_clean}.com",
                        f"https://{comp_name_clean}.com",
                        f"https://www.{comp_name}.com" if "." not in comp_name else f"https://{comp_name}",
                    ]
                    
                    # For now, store the name - we'll try to resolve to URL later or use in SERP
                    # Mark as direct competitor type
                    for url in potential_urls:
                        if _is_valid_competitor_url(url):
                            if sandbox_domain:
                                try:
                                    url_domain = urlparse(url).netloc.lower().replace("www.", "")
                                    if url_domain == sandbox_domain:
                                        continue
                                except Exception:
                                    pass
                            local_results.append((url, {"type": "business_competitor", "source": "direct_extraction"}))
                            break
                
                logger.info(f"[competitor_extraction] business_competitors = {len(local_results)}")
        except Exception as exc:
            logger.error(f"Layer A (Direct Competitors) failed: {exc}, continuing with authority sources")
        return local_results

    async def _extract_layer_b_serp() -> tuple:
        """Task for Layer B Phase 1: SERP Queries and Search"""
        # ==========================================
        # LAYER B: Authority & Influencer Sources (EXISTING)
        # PHASE 1: Strategic SERP API Queries (1-3 queries)
        # ==========================================
        logger.info("Phase 1: Generating strategic SERP queries (Parallel)")
        local_results = []
        local_queries = []
        try:
            if page_intent:
                local_queries = await generate_serp_queries(
                    page_intent=page_intent,
                    sandbox_title=sandbox_title,
                    sandbox_url=sandbox_url, 
                    model_name=model_name
                )
                
                logger.info(f"Generated {len(local_queries)} SERP queries: {local_queries}")
                
                # Run SERP API searches in parallel
                serp_tasks = [
                    run_in_threadpool(_serpapi_search, query, num_results=10)
                    for query in local_queries
                ]
                serp_results = await asyncio.gather(*serp_tasks, return_exceptions=True)
                
                # Collect SERP competitors with source_type='serp'
                for result in serp_results:
                    if isinstance(result, Exception):
                        logger.warning(f"SERP search error: {result}")
                        continue
                    if isinstance(result, list):
                        for url in result:
                            # Filter out sandbox domain
                            if sandbox_domain:
                                try:
                                    url_domain = urlparse(url).netloc.lower().replace("www.", "")
                                    if url_domain == sandbox_domain:
                                        continue
                                except Exception:
                                    pass
                            # Mark as authority source from SERP (blogs, reviews, etc.)
                            local_results.append((url, {"type": "authority_source", "source": "serp"}))
                
                logger.info(f"[competitor_extraction] authority_sources (SERP) = {len(local_results)}")
        except Exception as exc:
            logger.error(f"Phase 1 (SERP) failed: {exc}, continuing with LLM extraction only")
        return local_results, local_queries

    async def _extract_layer_b_llm() -> List[tuple]:
        """Task for Layer B Phase 2: LLM Answer Extraction"""
        # ==========================================
        # PHASE 2: Extract from LLM Answers (all 15 questions)
        # ==========================================
        logger.info("Phase 2: Extracting authority sources from LLM answers (Parallel)")
        local_results = []
        
        # Use COMPETITOR_REASONING model for LLM answer extraction
        target_model = model_name
        if target_model is None:
            target_model = get_model_for_role("COMPETITOR_REASONING")
        llm = get_llm_provider(target_model)
        
        # Verify provider
        actual_provider = llm.get_provider_name()
        # Check if Gemini model was requested but provider doesn't match
        if "gemini" in target_model.lower() and "gemini" not in actual_provider.lower():
            logger.error(
                f"[model_verification] ERROR: Competitor LLM answer extraction requested Gemini model ({target_model}) "
                f"but got provider {actual_provider}."
            )
        
        # Log LLM call with role
        _log_llm_call(
            step_name="competitor_discovery_llm_answers",
            requested_model=target_model,
            resolved_model=target_model,
            provider_name=actual_provider,
            requested_role="COMPETITOR_REASONING",
        )
        
        logger.info("[region_context] region=India")
        
        # OLD IMPLEMENTATION: Processed questions sequentially with a 10s sleep (keeps for reference)
        # for question_dict in questions:
        #     question = question_dict.get("q", "")
        #     if not question:
        #         continue
        #
        #     try:
        #         # Ask LLM normally (no RAG)
        #         # Competitor extraction needs more tokens for complete answers
        #         # Increased to handle long answers that might hit MAX_TOKENS
        #         max_tokens_options = [3000, 4000, 5000]
        #         answer = None
        #         last_error = None
        #         
        #         for max_tokens in max_tokens_options:
        #             try:
        #                 answer = await llm.chat(
        #                     messages=[
        #                         {
        #                             "role": "user",
        #                             "content": question,
        #                         }
        #                     ],
        #                     max_tokens=max_tokens,
        #                 )
        #                 if answer and answer.strip():
        #                     break  # Success, exit retry loop
        #                 else:
        #                     logger.warning(f"Empty competitor answer with max_tokens={max_tokens}, retrying with higher limit")
        #             except ValueError as e:
        #                 error_str = str(e)
        #                 last_error = e
        #                 # Check if it's a MAX_TOKENS issue and we can retry
        #                 is_max_tokens_error = (
        #                     "MAX_TOKENS" in error_str or 
        #                     "max_tokens" in error_str.lower() or
        #                     "max_tokens limit" in error_str.lower()
        #                 )
        #                 if is_max_tokens_error and max_tokens < max_tokens_options[-1]:
        #                     logger.warning(f"Hit max_tokens limit with {max_tokens}, retrying with next higher limit")
        #                     continue  # Try next higher max_tokens
        #                 else:
        #                     # If we've exhausted all max_tokens options, log and continue to next question
        #                     if is_max_tokens_error:
        #                         logger.error(f"All max_tokens options exhausted for question '{question[:30]}...'")
        #                         last_error = None  # Reset to allow continue
        #                         break  # Exit retry loop, will continue to next question
        #                     # Other error - raise
        #                     raise
        #         
        #         # If we still don't have answer after all retries
        #         if not answer or not answer.strip():
        #             if last_error:
        #                 logger.error(f"Error extracting competitors for question '{question[:30]}...': {last_error}")
        #                 continue
        #             logger.warning(f"Failed to get answer for question '{question[:30]}...'")
        #             continue
        #
        #         # Check if citations present
        #         has_citations = bool(re.search(r'\[.*?\]|http|www\.|\.com|\.org', answer, re.IGNORECASE))
        #
        #         if not has_citations:
        #             # Ask again for source URLs
        #             source_prompt = (
        #                 f"List the 5-10 most likely source URLs that informed your previous answer to: {question}\n\n"
        #                 "Return only URLs, one per line."
        #             )
        #             # Source URL extraction needs enough tokens for multiple URLs (5-10 URLs)
        #             # Increased to handle cases where model generates long responses
        #             max_tokens_options = [2000, 3000, 4000]
        #             source_response = None
        #             # last_source_error = None (unused variable in orig code but logic same)
        #             for max_tokens in max_tokens_options:
        #                 try:
        #                     source_response = await llm.chat(
        #                         messages=[{"role": "user", "content": source_prompt}],
        #                         max_tokens=max_tokens,
        #                     )
        #                     if source_response and source_response.strip():
        #                         break  # Success
        #                 except Exception:
        #                     continue
        #             
        #             if source_response and source_response.strip():
        #                 answer = source_response
        #             else:
        #                 logger.warning(f"Failed to get source URLs for question '{question[:30]}...', using original answer")
        #                 pass
        #
        #         # Extract URLs from answer
        #         urls = _extract_urls_from_text(answer)
        #         
        #         # Filter out sandbox domain and mark as LLM source
        #         for url in urls:
        #             if sandbox_domain:
        #                 try:
        #                     url_domain = urlparse(url).netloc.lower().replace("www.", "")
        #                     if url_domain == sandbox_domain:
        #                         continue
        #                 except Exception:
        #                     pass
        #             # Mark as authority source from LLM (blogs, reviews, etc.)
        #             # Only add if not already present (business competitors take precedence)
        #             local_results.append((url, {"type": "authority_source", "source": "llm"}))
        #
        #     except Exception as exc:
        #         logger.warning(f"Error extracting competitors for question '{question[:30]}...': {exc}")
        #         continue
        #
        #     await asyncio.sleep(10) # Rate limiting preserved

        # NEW IMPLEMENTATION: Process all 15 questions concurrently using asyncio.gather
        async def process_single_question(idx: int, q_dict: Dict[str, str]) -> List[tuple]:
            q_results = []
            question = q_dict.get("q", "")
            if not question:
                return []

            logger.info(f"Question {idx+1}/{len(questions)}: Processing '{question[:40]}...' (Parallel)")
            try:
                # Ask LLM normally (no RAG)
                # Competitor extraction needs more tokens for complete answers
                # Increased to handle long answers that might hit MAX_TOKENS
                max_tokens_options = [3000, 4000, 5000]
                answer = None
                last_error = None
                
                for max_tokens in max_tokens_options:
                    try:
                        answer = await llm.chat(
                            messages=[
                                {
                                    "role": "user",
                                    "content": question,
                                }
                            ],
                            max_tokens=max_tokens,
                        )
                        if answer and answer.strip():
                            break  # Success, exit retry loop
                        else:
                            logger.warning(f"Question {idx+1}: Empty competitor answer with max_tokens={max_tokens}, retrying with higher limit")
                    except ValueError as e:
                        error_str = str(e)
                        last_error = e
                        # Check if it's a MAX_TOKENS issue and we can retry
                        is_max_tokens_error = (
                            "MAX_TOKENS" in error_str or 
                            "max_tokens" in error_str.lower() or
                            "max_tokens limit" in error_str.lower()
                        )
                        if is_max_tokens_error and max_tokens < max_tokens_options[-1]:
                            logger.warning(f"Question {idx+1}: Hit max_tokens limit with {max_tokens}, retrying with next higher limit")
                            continue  # Try next higher max_tokens
                        else:
                            # If we've exhausted all max_tokens options, log and continue to next question
                            if is_max_tokens_error:
                                logger.error(f"Question {idx+1}: All max_tokens options exhausted for question '{question[:30]}...'")
                                last_error = None  # Reset to allow continue
                                break  # Exit retry loop, will continue to next question
                            # Other error - raise
                            raise
                
                # If we still don't have answer after all retries
                if not answer or not answer.strip():
                    if last_error:
                        logger.error(f"Question {idx+1}: Error extracting competitors for question '{question[:30]}...': {last_error}")
                        return []
                    logger.warning(f"Question {idx+1}: Failed to get answer for question '{question[:30]}...'")
                    return []

                # Check if citations present
                has_citations = bool(re.search(r'\[.*?\]|http|www\.|\.com|\.org', answer, re.IGNORECASE))

                if not has_citations:
                    # Ask again for source URLs
                    source_prompt = (
                        f"List the 5-10 most likely source URLs that informed your previous answer to: {question}\n\n"
                        "Return only URLs, one per line."
                    )
                    # Source URL extraction needs enough tokens for multiple URLs (5-10 URLs)
                    # Increased to handle cases where model generates long responses
                    max_tokens_options = [2000, 3000, 4000]
                    source_response = None
                    # last_source_error = None (unused variable in orig code but logic same)
                    for max_tokens in max_tokens_options:
                        try:
                            source_response = await llm.chat(
                                messages=[{"role": "user", "content": source_prompt}],
                                max_tokens=max_tokens,
                            )
                            if source_response and source_response.strip():
                                break  # Success
                        except Exception:
                            continue
                    
                    if source_response and source_response.strip():
                        answer = source_response
                    else:
                        logger.warning(f"Question {idx+1}: Failed to get source URLs for question '{question[:30]}...', using original answer")
                        pass

                # Extract URLs from answer
                urls = _extract_urls_from_text(answer)
                
                # Filter out sandbox domain and mark as LLM source
                for url in urls:
                    if sandbox_domain:
                        try:
                            url_domain = urlparse(url).netloc.lower().replace("www.", "")
                            if url_domain == sandbox_domain:
                                continue
                        except Exception:
                            pass
                    # Mark as authority source from LLM (blogs, reviews, etc.)
                    # Only add if not already present (business competitors take precedence)
                    q_results.append((url, {"type": "authority_source", "source": "llm"}))

            except Exception as exc:
                logger.warning(f"Question {idx+1}: Error extracting competitors for question '{question[:30]}...': {exc}")
                
            logger.info(f"Question {idx+1}: Completed extraction, found {len(q_results)} items")
            return q_results

        # Run questions with smooth concurrency control to prevent quota stampedes
        sem = asyncio.Semaphore(1)
        async def _bounded_process(i, q_dict):
            async with sem:
                res = await process_single_question(i, q_dict)
                await asyncio.sleep(1.0)
                return res

        question_tasks = [_bounded_process(i, q_dict) for i, q_dict in enumerate(questions)]
        results_per_question = await asyncio.gather(*question_tasks, return_exceptions=True)
        
        # Aggregate results
        for item in results_per_question:
            if isinstance(item, list):
                local_results.extend(item)
            elif isinstance(item, Exception):
                logger.error(f"A question processing task failed unexpectedly: {item}")
            
        logger.info(f"Phase 2: LLM extraction task completed, found {len(local_results)} items")
        return local_results

    # Execute phases in parallel
    logger.info("Executing phases 1, 2, and 3 in parallel...")
    results = await asyncio.gather(
        _extract_layer_a_direct(),
        _extract_layer_b_serp(),
        _extract_layer_b_llm(),
        return_exceptions=True
    )

    # Process Phase 1 Results (Direct Competitors)
    if isinstance(results[0], list):
        for url, meta in results[0]:
            all_urls_dict[url] = meta
    else:
        logger.error(f"Phase 1 task failed: {results[0]}")

    # Process Phase 2 Results (SERP)
    if isinstance(results[1], tuple):
        serp_urls, queries = results[1]
        serp_queries = queries
        for url, meta in serp_urls:
            # Only add if not already present (Direct Competitors take precedence)
            if url not in all_urls_dict:
                all_urls_dict[url] = meta
    elif isinstance(results[1], Exception):
        logger.error(f"Phase 2 task failed: {results[1]}")

    # Process Phase 3 Results (LLM)
    if isinstance(results[2], list):
        for url, meta in results[2]:
            # Only add if not already present (Direct/SERP take precedence)
            if url not in all_urls_dict:
                all_urls_dict[url] = meta
    else:
         logger.error(f"Phase 3 task failed: {results[2]}")

    logger.info(f"Parallel extraction completed. Total unique URLs: {len(all_urls_dict)}")
    
    # ==========================================
    # PHASE 3: Normalize and Deduplicate
    # ==========================================
    # Convert to list of dicts with url, type, and source
    # Handle both old format (string) and new format (dict) for backward compatibility
    competitor_list = []
    for url, data in all_urls_dict.items():
        if isinstance(data, dict):
            # New format: {"type": "business_competitor"|"authority_source", "source": "llm"|"serp"|"direct_extraction"}
            competitor_list.append({
                "url": url,
                "type": data.get("type", "authority_source"),  # Default to authority_source if missing
                "source": data.get("source", "llm"),  # Default to llm if missing
            })
        else:
            # Old format: just string source_type (backward compatibility)
            competitor_list.append({
                "url": url,
                "type": "authority_source",  # Assume authority_source for old format
                "source": data if isinstance(data, str) else "llm",
            })
    
    # Normalize URLs (deduplicate by domain)
    # Priority: business_competitor > authority_source, direct_extraction > serp > llm
    normalized = []
    seen_domains = {}  # domain -> competitor dict
    
    for comp in competitor_list:
        url = comp["url"]
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace("www.", "")
            
            # Final filter: remove sandbox domain
            if sandbox_domain and domain == sandbox_domain:
                continue
            
            # Deduplicate by domain (keep best occurrence)
            if domain not in seen_domains:
                seen_domains[domain] = comp
            else:
                # If we have a better competitor (business_competitor > authority_source), replace
                existing = seen_domains[domain]
                if comp.get("type") == "business_competitor" and existing.get("type") != "business_competitor":
                    seen_domains[domain] = comp
                elif comp.get("type") == existing.get("type"):
                    # Same type, prefer direct_extraction > serp > llm
                    source_priority = {"direct_extraction": 3, "serp": 2, "llm": 1}
                    comp_priority = source_priority.get(comp.get("source", "llm"), 0)
                    existing_priority = source_priority.get(existing.get("source", "llm"), 0)
                    if comp_priority > existing_priority:
                        seen_domains[domain] = comp
        except Exception:
            # Keep if parsing fails, but check if URL already exists
            if url not in [c["url"] for c in normalized]:
                normalized.append(comp)
    
    # Convert seen_domains back to list
    normalized = list(seen_domains.values())
    
    # Limit to 20 competitors (prioritize business competitors)
    business_competitors = [c for c in normalized if c.get("type") == "business_competitor"]
    authority_sources = [c for c in normalized if c.get("type") == "authority_source"]
    
    # Keep all business competitors, fill rest with authority sources
    final_list = business_competitors + authority_sources[:20 - len(business_competitors)]
    
    logger.info(f"Phase 3: Final competitor count: {len(final_list)} (Business Competitors: {len(business_competitors)}, Authority Sources: {len(authority_sources)})")
    
    return {
        "competitor_urls": final_list,  # List of dicts with 'url', 'type' ('direct'|'authority'), and 'source' ('llm'|'serp'|'direct_extraction')
        "serp_queries": serp_queries,  # List of SERP queries for storing as questions
    }


def _extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text using regex."""
    # Pattern for URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
    matches = re.findall(url_pattern, text, re.IGNORECASE)
    
    # Clean and validate
    urls = []
    for match in matches:
        if not match.startswith("http"):
            match = "https://" + match
        try:
            parsed = urlparse(match)
            if parsed.netloc:
                urls.append(match)
        except Exception:
            pass
    
    return urls


def _normalize_and_deduplicate_urls(urls: List[str]) -> List[str]:
    """Normalize domain names and deduplicate URLs."""
    seen_domains = set()
    normalized = []

    for url in urls:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix for comparison
            domain_key = domain.replace("www.", "")
            
            if domain_key not in seen_domains:
                seen_domains.add(domain_key)
                normalized.append(url)
        except Exception:
            continue

    return normalized


# ==========================================
# STEP 5: Crawl Competitors and Store in Pinecone
# ==========================================


async def crawl_and_store_competitors(
    run_id: str,
    sandbox_url: str,
    sandbox_data: Dict[str, Any],
    competitor_urls: List[Any],  # Can be List[str] for backward compat or List[Dict] with source_type
) -> Dict[str, Any]:
    """
    Step 5: Crawl competitors in parallel, chunk content, embed, and store in Pinecone.

    Args:
        competitor_urls: List of strings (backward compat) or List of dicts with 'url' and 'source_type'

    Returns:
        Dict with competitor data
    """
    # Normalize competitor_urls format (handle both old and new format)
    competitor_list = []
    for item in competitor_urls:
        if isinstance(item, str):
            # Old format: just URL string, default to 'competitor' source_type
            competitor_list.append({"url": item, "source_type": "competitor"})
        elif isinstance(item, dict):
            # New format: dict with url and source_type
            competitor_list.append(item)
        else:
            logger.warning(f"Invalid competitor format: {item}")
            continue
    
    # Extract just URLs for crawling
    urls = [comp["url"] for comp in competitor_list]
    
    # Crawl competitors in parallel
    competitor_tasks = [crawl.crawl_url(url) for url in urls]
    competitor_results = await asyncio.gather(*competitor_tasks, return_exceptions=True)

    competitors_data = []
    all_chunks = []

    # Extract sandbox domain and business competitor domains for classification
    sandbox_domain = urlparse(sandbox_url).netloc.lower().replace("www.", "")
    business_competitor_domains = []
    for comp_info in competitor_list:
        comp_type = comp_info.get("type", "")
        if comp_type == "business_competitor" or comp_type == "direct":
            comp_url = comp_info.get("url", "")
            if comp_url:
                try:
                    comp_domain = urlparse(comp_url).netloc.lower().replace("www.", "")
                    if comp_domain and comp_domain not in business_competitor_domains:
                        business_competitor_domains.append(comp_domain)
                except Exception:
                    pass

    # Get chunker instance based on CHUNKING_TYPE environment variable
    chunker = get_chunker()

    # Process sandbox page chunks
    sandbox_domain_type = classify_domain_type(sandbox_url, sandbox_domain, business_competitor_domains)
    sandbox_doc = ChunkerDocument(
        text=sandbox_data.get("full_text", ""),
        source=sandbox_url,
        metadata={
            "url": sandbox_url,
            "is_client": True,
            "domain": urlparse(sandbox_url).netloc,
            "source_type": "client",
            "domain_type": sandbox_domain_type,  # P2: Add domain_type for weighting
        }
    )
    sandbox_chunks = chunker.chunk_documents([sandbox_doc])
    # Ensure domain_type is propagated to all sandbox chunks
    for chunk in sandbox_chunks:
        if "metadata" not in chunk:
            chunk["metadata"] = {}
        chunk["metadata"]["domain_type"] = sandbox_domain_type
    all_chunks.extend(sandbox_chunks)

    # Process competitor pages
    for comp_info, result in zip(competitor_list, competitor_results):
        url = comp_info["url"]
        source_type = comp_info.get("source_type", "competitor")
        
        if isinstance(result, Exception):
            logger.warning(f"Failed to crawl competitor {url}: {result}")
            continue

        if "error" in result:
            continue

        full_text = result.get("full_content", "")
        domain = urlparse(url).netloc

        # Classify domain type for this competitor (P2: Before appending to data)
        comp_domain_type = classify_domain_type(url, sandbox_domain, business_competitor_domains)

        competitors_data.append({
            "url": url,
            "domain": domain,
            "title": _sanitize_text(result.get("title")),
            "h1": _sanitize_text(result.get("h1")),
            "ai_summary": "",
            "full_text": _sanitize_text(full_text),
            "source_type": source_type,  # Include source_type in competitor data
            "domain_type": comp_domain_type,  # P2: Include domain_type for database storage
        })
        
        # Chunk competitor content
        comp_doc = ChunkerDocument(
            text=full_text,
            source=url,
            metadata={
                "url": url,
                "is_client": False,
                "domain": domain,
                "source_type": source_type,
                "domain_type": comp_domain_type,  # P2: Add domain_type for weighting
            }
        )
        comp_chunks = chunker.chunk_documents([comp_doc])
        # Ensure domain_type is propagated to all competitor chunks
        for chunk in comp_chunks:
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            chunk["metadata"]["domain_type"] = comp_domain_type
        all_chunks.extend(comp_chunks)

    # Generate embeddings for all chunks
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    embeddings = await embed_documents(chunk_texts)

    # Store in Pinecone
    namespace = f"sandbox-{run_id}"
    pinecone_store = _get_pinecone_store()
    await pinecone_store.store_chunks(namespace, all_chunks, embeddings)

    # Persist chunks to Supabase (non-blocking, log errors only)
    await _persist_chunks_to_db(run_id, all_chunks)

    return {
        "competitors": competitors_data,
        "chunks_count": len(all_chunks),
    }


async def _persist_chunks_to_db(run_id: str, chunks: List[Dict[str, Any]]) -> None:
    """
    Persist chunks to Supabase database via backend API.
    This is non-blocking - failures are logged but do not crash the pipeline.
    
    Args:
        run_id: Sandbox run ID
        chunks: List of chunk dictionaries with url, chunk_id, text, metadata, etc.
    """
    if not chunks:
        logger.debug(f"No chunks to persist for run {run_id}")
        return
    
    # CRITICAL: domain_type is FINALIZED at ingestion and MUST NOT be reclassified later.
    # Rules:
    # - is_client = true → domain_type = 'sandbox_brand' (set during chunk creation)
    # - known competitors → domain_type = 'business_competitor' (set during chunk creation)
    # - everything else → domain_type = 'editorial' (set during chunk creation)
    # This domain_type is stored in sandbox_chunks and is the source of truth.
    # DO NOT reclassify chunks during chat retrieval - trust the persisted domain_type.
    chunk_records = []
    for chunk in chunks:
        # Extract metadata
        metadata = chunk.get("metadata", {})
        # domain_type is FINALIZED at chunk creation time (see classify_domain_type and crawl_and_store_competitors)
        domain_type = metadata.get("domain_type", "editorial")
        
        # Extract chunk fields
        url = chunk.get("url", "")
        chunk_id = chunk.get("chunk_id", 0)
        text = chunk.get("text", "")
        summary = chunk.get("summary")
        is_client = chunk.get("is_client", False)
        domain = chunk.get("domain")
        source_type = chunk.get("source_type", "competitor")
        
        # Sanitize text (remove null bytes for PostgreSQL)
        if text:
            text = text.replace('\x00', '')
        if summary:
            summary = summary.replace('\x00', '')
        
        chunk_records.append({
            "url": url,
            "chunk_id": chunk_id,
            "text": text,
            "summary": summary,
            "is_client": is_client,
            "domain": domain,
            "source_type": source_type,
            "domain_type": domain_type,  # FINALIZED - do not modify after persistence
        })
    
    # Send chunks to backend in batches (max 100 per batch to avoid payload size issues)
    batch_size = 100
    total_chunks = len(chunk_records)
    successful_batches = 0
    failed_batches = 0
    
    logger.info(f"[chunk_persistence] Starting persistence of {total_chunks} chunks for run {run_id}")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunk_records[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{BACKEND_BASE_URL}/api/sandbox/{run_id}/chunks",
                    json={"chunks": batch},
                )
                if response.status_code == 200:
                    successful_batches += 1
                    logger.debug(
                        f"[chunk_persistence] Batch {batch_num}/{total_batches} persisted successfully "
                        f"({len(batch)} chunks)"
                    )
                else:
                    failed_batches += 1
                    logger.warning(
                        f"[chunk_persistence] Batch {batch_num}/{total_batches} failed: "
                        f"{response.status_code} - {response.text[:200]}"
                    )
        except Exception as exc:
            failed_batches += 1
            logger.warning(
                f"[chunk_persistence] Batch {batch_num}/{total_batches} failed with exception: {exc}",
                exc_info=True
            )
    
    # Log summary
    if successful_batches == total_batches:
        logger.info(
            f"[chunk_persistence] Successfully persisted all {total_chunks} chunks "
            f"({successful_batches} batches) for run {run_id}"
        )
    elif successful_batches > 0:
        logger.warning(
            f"[chunk_persistence] Partially persisted chunks for run {run_id}: "
            f"{successful_batches}/{total_batches} batches succeeded, "
            f"{failed_batches} batches failed"
        )
    else:
        logger.error(
            f"[chunk_persistence] Failed to persist any chunks for run {run_id}: "
            f"all {failed_batches} batches failed"
        )


def _chunk_text(text: str, url: str, is_client: bool, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Chunk text into 300-500 token chunks.

    Args:
        text: Text to chunk
        url: Source URL
        is_client: Whether this is the client page
        domain: Domain name

    Returns:
        List of chunk dicts
    """
    if not text or not text.strip():
        return []

    # Rough estimate: 1 token ≈ 4 characters
    # Target: 300-500 tokens = 1200-2000 characters
    chunk_size = 1500  # Target ~375 tokens
    overlap = 200  # Overlap between chunks

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "url": url,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "summary": chunk_text[:200],  # Simple summary
                "is_client": is_client,
                "domain": domain or urlparse(url).netloc,
                "source_type": "client" if is_client else "competitor",
            })
            chunk_id += 1

        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ==========================================
# STEP 6: RAG Simulation per Question
# ==========================================


# Phrases the RAG model uses when the retrieved context cannot answer the question
_NO_INFO_PHRASES = (
    "does not contain information",
    "doesn't contain information",
    "does not contain enough information",
    "doesn't contain enough information",
    "does not contain any information",
    "not enough information",
    "insufficient information",
    "no information regarding",
    "no information about",
    "no information on",
    "does not provide information",
    "doesn't provide information",
    "cannot answer",
    "can't answer",
    "unable to answer",
)

_CITATION_TAG_PATTERN = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)
_MAX_QUOTE_CHARS = 200
_MAX_DIAGNOSED_SOURCES = 2


def _claim_for_citation(answer: str, cited_url: str) -> Optional[str]:
    """Return the answer sentence that a [Source:<cited_url>] tag is attached to."""
    segments = _CITATION_TAG_PATTERN.split(answer)
    # re.split with one group alternates: text, tag, text, tag, ...
    for i in range(1, len(segments), 2):
        if cited_url not in segments[i]:
            continue
        # Walk back over empty segments so "[Source:a][Source:b]" both map to the same claim
        for j in range(i - 1, -1, -2):
            text = segments[j].strip()
            if not text:
                continue
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            claim = sentences[-1].strip().lstrip("*-• ").strip() if sentences else ""
            if claim:
                if len(claim) > _MAX_QUOTE_CHARS:
                    claim = claim[:_MAX_QUOTE_CHARS].rstrip() + "…"
                return claim
        return None
    return None


def _diagnose_rag_result(
    did_sandbox_appear: bool,
    answer: str,
    sandbox_domain: str,
    business_competitor_citations: List[str],
    authority_source_citations: List[str],
    final_chunks: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    """
    Explain why the sandbox did not appear for a question. Purely deterministic: quotes
    the answer and counts retrieved chunks, no LLM calls.

    - competitor_advantage: at least one non-sandbox source (business competitor or
      editorial) was cited, so the information existed in the pool but not from the sandbox.
    - content_gap: no source was cited at all, so no crawled content answered the question.
    - None: the sandbox appeared, no diagnosis needed.
    """
    if did_sandbox_appear:
        return {"diagnosis_type": None, "diagnosis_detail": None}

    total_chunks = len(final_chunks)
    sandbox_chunks = 0
    for chunk in final_chunks:
        try:
            chunk_domain = urlparse(chunk.get("url", "")).netloc.lower().replace("www.", "")
        except Exception:
            continue
        if chunk_domain and chunk_domain == sandbox_domain:
            sandbox_chunks += 1

    if sandbox_chunks:
        retrieval_note = (
            f"{sandbox_chunks} of {total_chunks} retrieved chunks came from the sandbox but were not cited."
        )
    else:
        retrieval_note = f"0 of {total_chunks} retrieved chunks came from the sandbox."

    cited_sources = [("Business competitor", url) for url in business_competitor_citations] + [
        ("Editorial source", url) for url in authority_source_citations
    ]

    if cited_sources:
        parts = []
        for label, url in cited_sources[:_MAX_DIAGNOSED_SOURCES]:
            domain = urlparse(url).netloc.lower().replace("www.", "") or url
            claim = _claim_for_citation(answer, url)
            if claim:
                parts.append(f'{label} {domain} supplied: "{claim}"')
            else:
                parts.append(f"{label} {domain} was cited")
        detail = "; ".join(parts)
        remaining = len(cited_sources) - _MAX_DIAGNOSED_SOURCES
        if remaining > 0:
            detail += f" (+{remaining} more source{'s' if remaining > 1 else ''})"
        return {
            "diagnosis_type": "competitor_advantage",
            "diagnosis_detail": f"{detail}. {retrieval_note}",
        }

    answer_lower = (answer or "").lower()
    if any(phrase in answer_lower for phrase in _NO_INFO_PHRASES):
        gap_note = "No source was cited; the answer states the retrieved context lacks this information."
    else:
        gap_note = "No source was cited in the answer."
    return {
        "diagnosis_type": "content_gap",
        "diagnosis_detail": f"{gap_note} {retrieval_note}",
    }


async def run_rag_simulation(
    run_id: str,
    questions: List[Dict[str, str]],
    sandbox_url: str,
    business_category: Optional[str] = None,
    business_competitors: Optional[List[str]] = None,
    competitor_urls: Optional[List[Dict[str, Any]]] = None,
    sandbox_brand_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Step 6: For each question, perform RAG simulation using RAG_ANSWERING model.
    Now with business context injection for better grounding.

    Args:
        run_id: Sandbox run ID
        questions: List of question dicts
        sandbox_url: Sandbox URL
        business_category: Business category for context
        business_competitors: List of business competitor names/domains
        sandbox_brand_name: Sandbox brand name (extracted from URL/domain)
        model_name: Optional model override (uses MODEL_MAP if None)

    Returns:
        List of RAG result dicts
    """
    # Use RAG_ANSWERING model (NOT Flash)
    if model_name is None:
        model_name = get_model_for_role("RAG_ANSWERING")
    llm = get_llm_provider(model_name)
    
    # Verify provider
    actual_provider = llm.get_provider_name()
    # Check if Gemini model was requested but provider doesn't match
    if "gemini" in model_name.lower() and "gemini" not in actual_provider.lower():
        logger.error(
            f"[model_verification] ERROR: RAG answering requested Gemini model ({model_name}) "
            f"but got provider {actual_provider}."
        )
    
    # Log LLM call with role
    _log_llm_call(
        step_name="rag_answering",
        requested_model=model_name,
        resolved_model=model_name,
        provider_name=actual_provider,
        requested_role="RAG_ANSWERING",
    )
    
    logger.info("[region_context] region=India")
    
    # Extract sandbox brand name from URL if not provided
    if not sandbox_brand_name and sandbox_url:
        try:
            domain = urlparse(sandbox_url).netloc.lower().replace("www.", "")
            sandbox_brand_name = domain.split(".")[0].capitalize()
        except Exception:
            pass
    
    pinecone_store = _get_pinecone_store()
    namespace = f"sandbox-{run_id}"
    rag_results = []

    for question_dict in questions:
        question = question_dict.get("q", "")
        question_type = question_dict.get("type", "intent")
        
        # Skip SERP queries - they're stored as questions but not used in RAG simulation
        if question_type == "SERP":
            continue
        
        if not question:
            continue

        try:
            # Generate query embedding using the same documents API
            [query_vector] = await embed_documents([question])

            # Retrieve top-k chunks (mix of competitor + sandbox)
            # P2: Retrieve more chunks initially to allow for domain weighting and balancing
            top_k = 15  # Retrieve more than needed to allow for weighting/balancing
            retrieved_chunks = await pinecone_store.search(
                namespace=namespace,
                query_vector=query_vector,
                top_k=top_k,
            )

            if not retrieved_chunks:
                rag_results.append({
                    "question": question,
                    "type": question_type,
                    "answer": "No relevant content found.",
                    "did_sandbox_appear": False,
                    "chunks_used": 0,
                    "competitor_citations": [],
                    "sandbox_citations": [],
                    "diagnosis_type": "content_gap",
                    "diagnosis_detail": "No content from any source was retrieved for this question.",
                })
                continue

            # P2: Apply domain weights to similarity scores
            weighted_chunks = []
            for chunk in retrieved_chunks:
                domain_type = chunk.get("domain_type", "editorial")
                original_score = chunk.get("score", 0.0)
                weight = DOMAIN_WEIGHTS.get(domain_type, 0.8)  # Default to editorial weight
                weighted_score = original_score * weight
                weighted_chunks.append({
                    **chunk,
                    "original_score": original_score,
                    "weighted_score": weighted_score,
                    "domain_type": domain_type,
                })
            
            # Sort by weighted score (descending)
            weighted_chunks.sort(key=lambda x: x["weighted_score"], reverse=True)
            
            # P2: Retrieval balancing - ensure at least 1 brand chunk if available
            final_chunks = []
            brand_chunks = [c for c in weighted_chunks if c["domain_type"] in ["sandbox_brand", "business_competitor"]]
            editorial_chunks = [c for c in weighted_chunks if c["domain_type"] == "editorial"]
            
            # If we have brand chunks, ensure at least 1 is included
            if brand_chunks:
                # Take top brand chunk if not already in top results
                top_brand = brand_chunks[0]
                if top_brand not in weighted_chunks[:10]:
                    final_chunks.append(top_brand)
            
            # Add top weighted chunks (avoid duplicates)
            seen_urls = set()
            for chunk in weighted_chunks:
                if len(final_chunks) >= 10:
                    break
                chunk_url = chunk.get("url", "")
                if chunk_url not in seen_urls:
                    final_chunks.append(chunk)
                    seen_urls.add(chunk_url)
            
            # Log domain type distribution (once per question, not in tight loop)
            domain_type_counts = {}
            for chunk in final_chunks:
                dt = chunk.get("domain_type", "editorial")
                domain_type_counts[dt] = domain_type_counts.get(dt, 0) + 1
            logger.info(
                f"[rag_retrieval] question_id={question[:50]}... "
                f"domain_types={domain_type_counts} "
                f"total_chunks={len(final_chunks)}"
            )

            # Build RAG context
            context_parts = []
            for chunk in final_chunks:
                context_parts.append(f"[Source: {chunk['url']}]\n{chunk['text']}")

            context = "\n\n".join(context_parts)

            # Build business context for grounding
            business_context_parts = []
            if business_category:
                business_context_parts.append(f"Business Category: {business_category}")
            if business_competitors:
                competitors_str = ", ".join(business_competitors[:5])  # Limit to 5 for brevity
                business_context_parts.append(f"Key Business Competitors: {competitors_str}")
            if sandbox_brand_name:
                business_context_parts.append(f"Sandbox Brand: {sandbox_brand_name}")
            
            business_context = "\n".join(business_context_parts) if business_context_parts else None

            # Build system prompt with soft citation bias and India context
            # P2: Softer instruction - prefer brand sources but don't force
            system_prompt = (
                "You must answer ONLY using the given context. "
                "Cite every fact using [Source:<url>] format. "
                "IMPORTANT: If the context doesn't contain enough information to answer the question, "
                "state that clearly and DO NOT cite any sources. Only cite sources when you actually "
                "use information from them to answer the question.\n\n"
                "CONTEXT:\n"
                "Assume the user is located in India.\n"
                "Prefer Indian brands, pricing models (INR), delivery patterns, and regulations.\n\n"
                "CITATION PREFERENCE (Soft Guidance):\n"
                "If official brand sources (sandbox brand or direct business competitors) are relevant to the question, "
                "prefer citing them over blogs or listicles. Use editorial sources when brand pages do not answer the question "
                "or when they provide valuable complementary information."
            )
            
            if business_context:
                system_prompt += f"\n\nBusiness Context:\n{business_context}\n"
            
            # Inject category + brand into query for better alignment
            query_context = ""
            if business_category:
                query_context += f"\nBusiness Category: {business_category}\n"
            if sandbox_brand_name:
                query_context += f"Brand: {sandbox_brand_name}\n"
            
            rag_prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {question}{query_context}\n\n"
                "Answer the question using only the provided context. "
                "If you can answer the question using information from the context, cite sources using [Source:<url>] format. "
                "If the context does not contain enough information to answer the question, say so clearly and do not cite any sources.\n\n"
                "REMEMBER: Prefer citing sandbox brand and business competitors over editorial sources."
            )

            # RAG simulation needs more tokens for complete answers with citations
            # Optimized: Start at 3000 (recommended for large prompts) with 3500 as backup
            max_tokens_options = [3000, 3500]
            answer = None
            last_error = None

            logger.warning(f"[RUN_RAG_SIMULATION] in process for question={question}")
            
            for max_tokens in max_tokens_options:
                try:
                    answer = await llm.chat(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": rag_prompt},
                        ],
                        max_tokens=max_tokens,
                    )
                    if answer and answer.strip():
                        break  # Success, exit retry loop
                    else:
                        logger.warning(f"Empty RAG answer with max_tokens={max_tokens}, retrying with higher limit")
                except ValueError as e:
                    error_str = str(e)
                    last_error = e
                    # Check if it's a MAX_TOKENS issue and we can retry
                    is_max_tokens_error = (
                        "MAX_TOKENS" in error_str or 
                        "max_tokens" in error_str.lower() or
                        "max_tokens limit" in error_str.lower()
                    )
                    if is_max_tokens_error and max_tokens < max_tokens_options[-1]:
                        next_max = max_tokens_options[max_tokens_options.index(max_tokens) + 1]
                        logger.warning(f"Hit max_tokens limit with {max_tokens}, retrying with {next_max}")
                        continue  # Try next higher max_tokens
                    else:
                        # Other error or already at max - raise
                        raise
            
            # If we still don't have answer after all retries
            if not answer or not answer.strip():
                if last_error:
                    raise last_error
                raise ValueError("Failed to generate RAG answer: Empty response after all retries")

            # NEW: Compute metrics using the dedicated function
            rag_metrics = compute_rag_metrics(
                url=sandbox_url,
                brand_name=sandbox_brand_name,
                answer=answer
            )

            logger.warning(f"[RUN_RAG_SIMULATION] for question={question}, with rag_metrics={rag_metrics}")
            
            # Extract metrics for logging and result dict
            did_sandbox_appear = rag_metrics.get("is_brand_mentioned", False) or rag_metrics.get("is_brand_cited", False)
            
            # Extract URLs explicitly cited by the LLM in the answer (Standard extraction for 'citations' list) (Standard extraction for 'citations' list)
            cited_urls = _extract_urls_from_text(answer)
            
            # Initialize citation lists
            sandbox_citations = []
            business_competitor_citations = []
            authority_source_citations = []
            
            # Normalize sandbox domain for comparison
            sandbox_domain = urlparse(sandbox_url).netloc.lower().replace("www.", "")
            
            # Build lookup maps for competitor classification
            business_competitor_domains = set()
            if competitor_urls:
                for comp in competitor_urls:
                    comp_type = comp.get("type", "authority_source")
                    comp_url = comp.get("url", "")
                    if comp_url:
                        try:
                            comp_domain = urlparse(comp_url).netloc.lower().replace("www.", "")
                            if comp_type == "business_competitor":
                                business_competitor_domains.add(comp_domain)
                        except Exception:
                            pass
            
            # Categorize cited URLs
            for cited_url in cited_urls:
                try:
                    # Normalize the cited URL's domain for comparison
                    cited_domain = urlparse(cited_url).netloc.lower().replace("www.", "")
                    cited_url_normalized = cited_url.lower()
                    sandbox_url_normalized = sandbox_url.lower()
                    
                    # Check if this URL matches the sandbox URL or domain
                    if (cited_url_normalized == sandbox_url_normalized or 
                        cited_domain == sandbox_domain):
                        if cited_url not in sandbox_citations:
                            sandbox_citations.append(cited_url)
                    elif cited_domain in business_competitor_domains:
                        # This is a business competitor
                        if cited_url not in business_competitor_citations:
                            business_competitor_citations.append(cited_url)
                    else:
                        # This is an authority source (blog, review, etc.)
                        if cited_url not in authority_source_citations:
                            authority_source_citations.append(cited_url)
                except Exception:
                    # If URL parsing fails, treat as authority source
                    if cited_url not in authority_source_citations:
                        authority_source_citations.append(cited_url)
            
            # For backward compatibility, combine business and authority into competitor_citations
            competitor_citations = business_competitor_citations + authority_source_citations
            
            # Check if business competitors were mentioned
            business_competitor_mentioned = len(business_competitor_citations) > 0
            
            # Check for editorial dominance (warn if only editorial sources cited)
            if len(authority_source_citations) > 0 and len(business_competitor_citations) == 0 and not did_sandbox_appear:
                logger.warning(
                    f"[rag_warning] editorial_dominance_detected question_id={len(rag_results) + 1} "
                    f"editorial_citations={len(authority_source_citations)} business_competitor_citations=0 sandbox_citations=0"
                )
            
            # Validate answer alignment with category
            if business_category and answer:
                answer_lower = answer.lower()
                category_lower = business_category.lower()
                # Check if answer references category keywords
                category_keywords = [kw for kw in category_lower.split() if len(kw) > 4]
                has_category_reference = any(kw in answer_lower for kw in category_keywords)
                if not has_category_reference:
                    logger.warning(
                        f"[rag_warning] answer_drift_detected question_id={len(rag_results) + 1} "
                        f"category={business_category[:50]} answer_does_not_reference_category=True"
                    )
            
            # Per-question logging (not noisy - only key metrics)
            question_id = len(rag_results) + 1
            logger.info(
                f"[rag] question_id={question_id} sandbox_mentioned={did_sandbox_appear} "
                f"business_competitor_mentioned={business_competitor_mentioned} "
                f"business_competitor_citations={len(business_competitor_citations)} "
                f"editorial_citations={len(authority_source_citations)} "
                f"authority_citations={len(authority_source_citations)}"
            )

            diagnosis = _diagnose_rag_result(
                did_sandbox_appear=did_sandbox_appear,
                answer=answer,
                sandbox_domain=sandbox_domain,
                business_competitor_citations=business_competitor_citations,
                authority_source_citations=authority_source_citations,
                final_chunks=final_chunks,
            )

            # CRITICAL: Citation arrays are FINALIZED here and become the source of truth.
            # These arrays are persisted to sandbox_rag_results table and MUST be trusted for analytics.
            # DO NOT reclassify or modify citations during retrieval - they are finalized at ingestion.
            # Analytics queries MUST use sandbox_rag_results, NOT Pinecone or runtime chunk retrieval.
            rag_results.append({
                "question": question,
                "type": question_type,
                "answer": answer.strip(),
                "did_sandbox_appear": did_sandbox_appear,
                "chunks_used": len(final_chunks),
                "competitor_citations": competitor_citations,  # Backward compatibility
                "sandbox_citations": sandbox_citations,  # FINAL - source of truth for analytics
                "business_competitor_citations": business_competitor_citations,  # FINAL - source of truth for analytics
                "authority_source_citations": authority_source_citations,  # FINAL - source of truth for analytics
                "metrics": rag_metrics,  # Store the full metrics dict
                "diagnosis_type": diagnosis["diagnosis_type"],
                "diagnosis_detail": diagnosis["diagnosis_detail"],
            })
            #lets prinf a log which will print the rag_metrics 
            logger.info(f"[rag_metrics] question_id={question_id} rag_metrics={rag_metrics}")

        except Exception as exc:
            logger.error(f"Error in RAG simulation for question '{question}': {exc}")
            rag_results.append({
                "question": question,
                "type": question_type,
                "answer": f"Error: {str(exc)}",
                "did_sandbox_appear": False,
                "chunks_used": 0,
                "competitor_citations": [],
                "sandbox_citations": [],
            })

    return rag_results


# ==========================================
# STEP 7: Final Aggregation and Scoring
# ==========================================


def calculate_simulation_reality_score(
    business_category: str = None,
    questions: List[Dict[str, str]] = None,
    competitor_urls: List[Dict[str, Any]] = None,
) -> float:
    """
    NEW METRIC: Calculate Simulation Reality Score (0-100).
    
    Detects when simulations drift from real business reality.
    
    Inputs:
    - % of questions aligned with category keywords
    - % of competitors that are direct competitors
    - Category mismatch signals
    
    Returns:
        Simulation Reality Score: 0-100
    """
    if not business_category:
        # Without category, we can't validate - return neutral score
        return 50.0
    
    score_components = []
    
    # Component 1: Question-Category Alignment (0-40 points)
    if questions:
        category_keywords = set(business_category.lower().split())
        aligned_count = 0
        for q_dict in questions:
            question = q_dict.get("q", "").lower()
            # Check if question contains category-relevant keywords
            question_words = set(question.split())
            # Simple overlap check - if question shares meaningful words with category
            overlap = len(category_keywords.intersection(question_words))
            if overlap > 0 or any(kw in question for kw in category_keywords if len(kw) > 4):
                aligned_count += 1
        
        alignment_ratio = aligned_count / len(questions) if questions else 0.0
        question_score = alignment_ratio * 40.0
        score_components.append(("question_alignment", question_score, alignment_ratio))
    else:
        question_score = 0.0
    
    # Component 2: Business Competitor Ratio (0-40 points)
    if competitor_urls:
        business_competitor_count = sum(1 for c in competitor_urls if c.get("type") == "business_competitor")
        total_count = len(competitor_urls)
        business_competitor_ratio = business_competitor_count / total_count if total_count > 0 else 0.0
        competitor_score = business_competitor_ratio * 40.0
        score_components.append(("business_competitor_ratio", competitor_score, business_competitor_ratio))
    else:
        competitor_score = 0.0
    
    # Component 3: Category Consistency (0-20 points)
    # If we have both questions and competitors, check consistency
    consistency_score = 20.0  # Default: assume consistent
    if questions and competitor_urls:
        # Check if questions and competitors seem to match the category
        # This is a simple heuristic - could be enhanced with LLM
        category_lower = business_category.lower()
        has_direct_competitors = any(c.get("type") == "direct" for c in competitor_urls)
        
        # If category suggests a specific business model, check alignment
        if "marketplace" in category_lower and not has_direct_competitors:
            consistency_score *= 0.5  # Penalize if marketplace but no direct competitors
        elif "subscription" in category_lower:
            # Check if questions mention subscription-related terms
            sub_keywords = ["subscription", "monthly", "annual", "plan", "pricing"]
            has_sub_questions = any(
                any(kw in q.get("q", "").lower() for kw in sub_keywords)
                for q in questions
            )
            if not has_sub_questions:
                consistency_score *= 0.7
    
    score_components.append(("category_consistency", consistency_score, consistency_score / 20.0))
    
    # Calculate total score
    total_score = question_score + competitor_score + consistency_score
    total_score = min(100.0, max(0.0, total_score))  # Clamp to 0-100
    
    logger.info(f"Simulation Reality Score: {total_score:.1f} (Question alignment: {question_score:.1f}, "
                f"Direct competitors: {competitor_score:.1f}, Consistency: {consistency_score:.1f})")
    
    return round(total_score, 2)


async def calculate_final_scores(
    rag_results: List[Dict[str, Any]],
    questions: List[Dict[str, str]],
    business_category: str = None,
    competitor_urls: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Step 7: Calculate AEO/GEO scores and generate recommendations.
    Now includes Simulation Reality Score.

    Returns:
        Dict with scores and recommendations
    """
    if not rag_results:
        reality_score = calculate_simulation_reality_score(
            business_category=business_category,
            questions=questions,
            competitor_urls=competitor_urls,
        )
        return {
            "geo_score": 0.0,
            "aeo_score": 0.0,
            "simulation_reality_score": reality_score,
            "citation_rate": 0.0,
            "coverage": 0.0,
            "competitor_dominance": 0.0,
            "missing_content": [],
            "schema_gaps": [],
            "structured_data_ops": [],
            "recommendations": [],
        }

    # Calculate citation rate
    sandbox_appeared_count = sum(1 for r in rag_results if r.get("did_sandbox_appear", False))
    citation_rate = (sandbox_appeared_count / len(rag_results)) * 100.0 if rag_results else 0.0

    # Calculate competitor dominance
    total_competitor_citations = sum(len(r.get("competitor_citations", [])) for r in rag_results)
    total_sandbox_citations = sum(len(r.get("sandbox_citations", [])) for r in rag_results)
    total_citations = total_competitor_citations + total_sandbox_citations
    competitor_dominance = (
        (total_competitor_citations / total_citations * 100.0) if total_citations > 0 else 0.0
    )

    # Calculate coverage (based on chunks used)
    avg_chunks_used = sum(r.get("chunks_used", 0) for r in rag_results) / len(rag_results) if rag_results else 0.0
    coverage = min(100.0, (avg_chunks_used / 10.0) * 100.0)  # Normalize to 0-100

    # GEO Score (Generative Engine Optimization)
    # Based on citation rate and coverage
    geo_score = (citation_rate * 0.7) + (coverage * 0.3)

    # AEO Score (Answer Engine Optimization)
    # Based on citation rate, coverage, and low competitor dominance
    aeo_score = (citation_rate * 0.5) + (coverage * 0.3) + ((100.0 - competitor_dominance) * 0.2)

    # NEW: Simulation Reality Score
    reality_score = calculate_simulation_reality_score(
        business_category=business_category,
        questions=questions,
        competitor_urls=competitor_urls,
    )
    
    # NEW: Internal Metrics (logged only, not exposed in API)
    business_competitor_citation_count = sum(
        len(r.get("business_competitor_citations", [])) for r in rag_results
    )
    authority_source_citation_count = sum(
        len(r.get("authority_source_citations", [])) for r in rag_results
    )
    total_business_citations = business_competitor_citation_count + authority_source_citation_count
    
    business_competitor_citation_rate = (
        (business_competitor_citation_count / total_business_citations * 100.0)
        if total_business_citations > 0 else 0.0
    )
    authority_blog_citation_rate = (
        (authority_source_citation_count / total_business_citations * 100.0)
        if total_business_citations > 0 else 0.0
    )
    sandbox_visibility_score = citation_rate  # Same as citation_rate for now
    
    # Category drift score (from reality score calculation)
    category_drift_score = 100.0 - reality_score  # Inverse of reality score
    
    # Calculate citation mix
    total_sandbox_citations = sum(len(r.get("sandbox_citations", [])) for r in rag_results)
    total_business_competitor_citations = sum(len(r.get("business_competitor_citations", [])) for r in rag_results)
    total_authority_source_citations = sum(len(r.get("authority_source_citations", [])) for r in rag_results)
    
    # Log internal metrics with citation mix
    logger.info(
        f"[simulation_quality] sandbox_visibility={sandbox_visibility_score:.2f} "
        f"category_drift={category_drift_score:.2f} "
        f"business_competitor_citation_rate={business_competitor_citation_rate:.2f} "
        f"authority_blog_citation_rate={authority_blog_citation_rate:.2f}"
    )
    logger.info(
        f"[citation_mix] sandbox={total_sandbox_citations} business={total_business_competitor_citations} "
        f"editorial={total_authority_source_citations}"
    )
    
    # Calculate brand visibility score (0-100)
    brand_visibility_score = (
        (total_sandbox_citations / len(rag_results) * 100.0) if rag_results else 0.0
    )
    logger.info(f"[brand_visibility_score] score={brand_visibility_score:.2f}")
    
    # VALIDATION WARNING: Check for zero sandbox citations despite high alignment
    if business_category and competitor_urls:
        business_competitor_count = sum(
            1 for c in competitor_urls if c.get("type") == "business_competitor"
        )
        
        # Calculate category alignment (reuse logic from reality score)
        category_alignment = 0.0
        if questions:
            alignment_count = _calculate_category_alignment(questions, business_category)
            category_alignment = (alignment_count / len(questions) * 100.0) if questions else 0.0
        
        if (sandbox_appeared_count == 0 and 
            business_competitor_count > 3 and 
            category_alignment > 70.0):
            logger.warning(
                f"[warning] Simulation completed with zero sandbox citations despite high category alignment "
                f"(category_alignment={category_alignment:.1f}%, business_competitors={business_competitor_count}). "
                f"This may indicate content gaps or retrieval issues."
            )

    return {
        "geo_score": round(geo_score, 2),
        "aeo_score": round(aeo_score, 2),
        "simulation_reality_score": reality_score,  # NEW METRIC
        "citation_rate": round(citation_rate, 2),
        "coverage": round(coverage, 2),
        "competitor_dominance": round(competitor_dominance, 2),
        "missing_content": [],  # Could be enhanced with LLM analysis
        "schema_gaps": [],  # Could be enhanced with LLM analysis
        "structured_data_ops": [],  # Could be enhanced with LLM analysis
        "recommendations": '',
    }


def compute_rag_metrics(url: str, brand_name: str, answer: str) -> Dict[str, Any]:
    """
    Computes advanced RAG metrics including brand presence, competitor SOC (Share of Voice),
    and citation placement.

    Args:
        url: The sandbox page URL.
        brand_name: The extracted brand name of the sandbox page.
        answer: The generated RAG answer.
    
    Returns:
        Dict containing metrics.
    """
    if not answer:
        return {}

    # 1. Brand & Competitor Analysis using GLiNER
    # If no brand name is known, we can't do brand specific checks, but we can still find competitors
    safe_brand_name = brand_name if brand_name else "Unknown Brand"
    logger.warning(f"[COMPUTE_RAG_METRICS] brand_name={brand_name}")
    
    # Lazy load if not already loaded
    _get_gliner_model()
    
    # Remove citations from text before analysis to avoid counting matches in URLs
    citation_pattern = re.compile(r'\[Source:\s*(.*?)\]', re.IGNORECASE)
    text_for_analysis = citation_pattern.sub("", answer)
    
    classification_result = extract_competitors_gliner_improved(
        text=text_for_analysis,
        brand=safe_brand_name
    )
    
    # Extract metrics from GLiNER result
    brand_mentions_count = classification_result.get("brand_mentions", 0)
    is_brand_mentioned = brand_mentions_count > 0
    competitor_mention_count = classification_result.get("total_competitor_mentions", 0)
    
    # 2. Citation Analysis
    # Check for [Source: url] format
    # We parse the URL from the source tag to see if it matches our sandbox URL
    
    sandbox_domain = urlparse(url).netloc.lower().replace("www.", "")
    
    # Regex to find all citations: [Source: ...]
    citation_pattern = re.compile(r'\[Source:\s*(.*?)\]', re.IGNORECASE)
    citations = citation_pattern.findall(answer)
    
    sandbox_citations_count = 0
    is_brand_cited = False
    brand_cited_first = False
    
    if citations:
        # Check first citation
        first_citation_url = citations[0].strip()
        try:
             first_citation_domain = urlparse(first_citation_url).netloc.lower().replace("www.", "")
             if (first_citation_url == url) or (sandbox_domain and sandbox_domain in first_citation_domain):
                 brand_cited_first = True
        except Exception:
            pass

        # Check all citations
        for cit in citations:
            cit_url = cit.strip()
            # Exact URL match or domain match
            if cit_url == url:
                sandbox_citations_count += 1
            else:
                try:
                    cit_domain = urlparse(cit_url).netloc.lower().replace("www.", "")
                    if sandbox_domain and sandbox_domain in cit_domain:
                        sandbox_citations_count += 1
                except:
                    pass
        
        if sandbox_citations_count > 0:
            is_brand_cited = True

    # 3. Fallback logic for is_brand_mentioned (string matching) if GLiNER fails or misses
    # The user wanted a string matching check as well.
    if not is_brand_mentioned and safe_brand_name != "Unknown Brand":
         if whole_word_count(answer, safe_brand_name) > 0:
             is_brand_mentioned = True
             brand_mentions_count = max(brand_mentions_count, whole_word_count(answer, safe_brand_name))
    
    logger.warning(f"[COMPUTE_RAG_METRICS] for answer={answer}, with brand_name={brand_name}, brand_mentions_count={brand_mentions_count}, is_brand_mentioned={is_brand_mentioned}, brand_cited_first={brand_cited_first}")

    return {
        "brand_mention_count": brand_mentions_count,
        "competitor_mention_count": competitor_mention_count,
        "is_brand_mentioned": is_brand_mentioned,
        "is_brand_cited": is_brand_cited,
        "brand_cited_first": brand_cited_first,
        "competitors_list": classification_result.get("competitors_detected", []),
        "competitors_counts": classification_result.get("competitor_counts", {})
    }
