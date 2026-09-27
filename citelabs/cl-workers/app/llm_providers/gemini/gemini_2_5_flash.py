"""
Gemini LLM Provider with Rate Limiting using Google AI Studio API
"""
import asyncio
import logging
import os
from typing import List, Dict, Optional

from google import genai
from google.genai import types

from ...constants import MODEL_GEMINI_2_5_FLASH
from ..base import BaseLLMProvider, GroundedResponse, GroundingChunk, order_cited_chunks
from ..rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


FALLBACK_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
]


class Gemini25FlashProvider(BaseLLMProvider):
    """Gemini LLM provider using Google's Gemini API with multi-model pool fallback."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")
        
        self.fallback_models = FALLBACK_MODELS.copy()
        default_model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
        if default_model in self.fallback_models:
            self.fallback_models.remove(default_model)
        self.fallback_models.insert(0, default_model)
        
        self.model_index = 0
        self.model_name = self.fallback_models[0]
        self.client = genai.Client(api_key=self.api_key)
        self.rate_limiter = get_rate_limiter()
        logger.info(f"Initialized Gemini Flash provider (active: {self.model_name}, pool: {self.fallback_models})")

    async def wait_for_slot(self) -> None:
        """Wait for rate limit slot."""
        await self.rate_limiter.wait_for_slot(self.model_name)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send chat request to Gemini with rate limiting and automatic model pool rotation.
        """
        try:
            # Convert messages to Gemini contents format
            contents = []
            system_instruction = None
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    contents.append({"role": "user", "parts": [{"text": content}]})
                elif role in ("assistant", "model"):
                    contents.append({"role": "model", "parts": [{"text": content}]})

            if not contents:
                if system_instruction:
                    contents.append({"role": "user", "parts": [{"text": system_instruction}]})
                else:
                    return ""

            # Generation config
            config = types.GenerateContentConfig(
                temperature=temperature if temperature is not None else 0.7,
                max_output_tokens=max_tokens if max_tokens else 2048,
                system_instruction=system_instruction,
            )

            # Retry loop with multi-model pool rotation on 429/503
            max_retries = len(self.fallback_models) + 1
            for attempt in range(max_retries):
                await self.wait_for_slot()
                try:
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=contents,
                        config=config,
                    )
                    if response.text:
                        return response.text.strip()
                    else:
                        logger.warning(f"Gemini returned empty response for model {self.model_name}")
                        return ""
                except Exception as call_err:
                    err_str = str(call_err)
                    is_rate_or_spike = (
                        "503" in err_str
                        or "429" in err_str
                        or "UNAVAILABLE" in err_str
                        or "demand" in err_str
                        or "RESOURCE_EXHAUSTED" in err_str
                    )
                    if is_rate_or_spike and attempt < max_retries - 1:
                        prev_model = self.model_name
                        self.model_index = (self.model_index + 1) % len(self.fallback_models)
                        self.model_name = self.fallback_models[self.model_index]
                        logger.warning(
                            f"Gemini quota/spike on {prev_model} ({err_str[:60]}). "
                            f"Auto-switching to pool model: {self.model_name} (attempt {attempt + 1}/{max_retries})..."
                        )
                        await asyncio.sleep(1.5)
                        continue
                    raise

        except Exception as exc:
            logger.error(f"Gemini API error ({self.model_name}): {exc}")
            raise

    async def grounded_generate(self, prompt: str, model: Optional[str] = None) -> GroundedResponse:
        """
        Answer with Google Search grounding enabled.

        Unlike chat(), this never rotates to another model in the fallback pool on a
        quota error: repeated observations must all come from the same model, or the
        run-to-run variation being measured would mix in model differences. It
        retries the same model with backoff instead.
        """
        model = model or os.getenv("STAGE1_GROUNDING_MODEL", "gemini-2.5-flash")
        max_retries = int(os.getenv("STAGE1_GROUNDING_MAX_RETRIES", "3"))
        config = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])

        for attempt in range(max_retries + 1):
            await self.rate_limiter.wait_for_slot(model)
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model,
                    contents=prompt,
                    config=config,
                )
                break
            except Exception as call_err:
                err_str = str(call_err)
                retryable = any(s in err_str for s in ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                if retryable and attempt < max_retries:
                    delay = 5.0 * (2 ** attempt)
                    logger.warning(
                        f"Grounded call on {model} hit {err_str[:60]}; retrying in {delay:.0f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

        candidate = response.candidates[0] if response.candidates else None
        metadata = candidate.grounding_metadata if candidate else None
        # Keep every chunk, even ones without a web URI, so support indices stay aligned
        chunks = [
            GroundingChunk(uri=(c.web.uri or "") if c.web else "", title=c.web.title if c.web else None)
            for c in (metadata.grounding_chunks or [] if metadata else [])
        ]
        spans = [
            (s.segment.start_index if s.segment else None, s.grounding_chunk_indices or [])
            for s in (metadata.grounding_supports or [] if metadata else [])
        ]
        return GroundedResponse(
            text=response.text or "",
            model=model,
            chunks=chunks,
            cited_chunk_order=[i for i in order_cited_chunks(spans) if i < len(chunks)],
            search_queries=list(metadata.web_search_queries or []) if metadata else [],
        )

    def get_provider_name(self) -> str:
        return f"Gemini ({self.model_name})"