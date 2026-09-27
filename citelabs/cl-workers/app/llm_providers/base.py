"""
Base LLM Provider Interface
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, List, Dict, Any, Optional, Sequence, Tuple


def order_cited_chunks(spans: Iterable[Tuple[Optional[int], Sequence[int]]]) -> List[int]:
    """
    Chunk indices in order of first appearance in the answer.
    `spans` holds (answer_start_offset, chunk_indices) per grounded answer segment;
    a missing offset sorts first. Each chunk is listed once.
    """
    order: List[int] = []
    for _, indices in sorted(spans, key=lambda span: span[0] or 0):
        for index in indices:
            if index not in order:
                order.append(index)
    return order


@dataclass
class GroundingChunk:
    """One source the model retrieved while answering with search grounding."""

    uri: str  # may be a provider redirect link, not the source's real URL
    title: Optional[str] = None


@dataclass
class GroundedResponse:
    """Answer produced with web search grounding, plus the sources it drew on."""

    text: str
    model: str
    chunks: List[GroundingChunk] = field(default_factory=list)
    # Indices into `chunks` in order of first appearance in the answer text.
    # Chunks the answer never references are absent.
    cited_chunk_order: List[int] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a chat request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        pass

    async def grounded_generate(self, prompt: str, model: Optional[str] = None) -> GroundedResponse:
        """
        Answer `prompt` with web search grounding and return the cited sources.
        Optional capability: providers without search grounding keep this default.
        """
        raise NotImplementedError(f"{self.get_provider_name()} does not support search grounding")

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of this provider."""
        pass

    @abstractmethod
    async def wait_for_slot(self) -> None:
        """
        Wait for a rate limit slot to become available.
        Uses the model name stored in the provider instance.
        """
        pass

