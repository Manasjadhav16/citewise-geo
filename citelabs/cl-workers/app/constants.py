"""
Constants for the application
"""

# Chunking type constants
CHUNKING_TYPE_SEMENTIC = "SEMENTIC"

# LLM Model constants
MODEL_GEMINI_2_5_FLASH = "gemini-3.7-flash"
MODEL_GEMINI_1_5_PRO = "gemini-3.7-flash"
MODEL_MISTRAL_MEDIUM_LATEST = "mistral-medium-latest"
MODEL_GPT_4_O_MINI = "gpt-4o-mini"
MODEL_GPT_5_NANO = "gpt-5-nano"
MULTI_PROVIDER = "multi-provider"

# Environment variable names for API keys
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# ==========================================
# GEO Simulation Pipeline Model Configuration
# ==========================================
# Semantic model roles - change models here to update entire pipeline
# This provides clear semantic meaning for each model role

MODEL_ROLES = {
    # Fast, cheap, high-throughput
    "FAST_EXTRACTION": MODEL_GEMINI_2_5_FLASH,
    
    # Reasoning but still safe on rate limits
    "CATEGORY_REASONING": MODEL_GEMINI_2_5_FLASH,
    "QUESTION_GENERATION": MODEL_GEMINI_2_5_FLASH,
    "COMPETITOR_REASONING": MODEL_GEMINI_2_5_FLASH,
    "RAG_ANSWERING": MODEL_GEMINI_2_5_FLASH,

    # Stage 1 (real-world observation). GROUNDED_OBSERVATION picks the provider;
    # the grounded model itself is STAGE1_GROUNDING_MODEL (see grounded_generate).
    "GROUNDED_OBSERVATION": MODEL_GEMINI_2_5_FLASH,
    "SOURCE_CLASSIFICATION": MODEL_GEMINI_2_5_FLASH,
    "EVIDENCE_EXTRACTION": MODEL_GEMINI_2_5_FLASH,
}

# Legacy MODEL_MAP for backward compatibility
# Maps pipeline steps to semantic roles
MODEL_MAP = {
    "crawl": MODEL_ROLES["FAST_EXTRACTION"],
    "chunking": MODEL_ROLES["FAST_EXTRACTION"],  # Not used for LLM, but kept for consistency
    "category_classification": MODEL_ROLES["CATEGORY_REASONING"],
    "question_generation": MODEL_ROLES["QUESTION_GENERATION"],
    "competitor_discovery": MODEL_ROLES["COMPETITOR_REASONING"],
    "rag_answering": MODEL_ROLES["RAG_ANSWERING"],
    "intent_extraction": MODEL_ROLES["FAST_EXTRACTION"],
}

def get_model_for_step(step: str) -> str:
    """
    Get the configured model for a pipeline step.
    
    Args:
        step: Pipeline step name (e.g., "category_classification", "question_generation")
        
    Returns:
        Model name string
    """
    return MODEL_MAP.get(step, MODEL_ROLES["FAST_EXTRACTION"])

def get_model_for_role(role: str) -> str:
    """
    Get the configured model for a semantic role.
    
    Args:
        role: Semantic role (e.g., "CATEGORY_REASONING", "QUESTION_GENERATION")
        
    Returns:
        Model name string
    """
    return MODEL_ROLES.get(role, MODEL_ROLES["FAST_EXTRACTION"])

