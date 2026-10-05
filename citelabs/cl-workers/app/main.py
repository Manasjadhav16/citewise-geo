from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()  # ← Add this here
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any

from . import crawl, schema, simulation, sandbox, async_job, chat
from app.test_embeddings import router as test_embed_router
from .llm_providers.factory import get_llm_provider



class CrawlRequest(BaseModel):
    url: HttpUrl


class SchemaGenerateRequest(BaseModel):
    page_record: dict


class SimulationRequest(BaseModel):
    url: HttpUrl
    query: str


app = FastAPI(title="CiteLabs Worker API", version="0.1.0")

app.include_router(test_embed_router)

@app.post("/crawl")
async def crawl_endpoint(payload: CrawlRequest):
    return await crawl.crawl_url(str(payload.url))


@app.post("/schema/generate")
async def schema_endpoint(payload: SchemaGenerateRequest):
    return await schema.generate_schema(payload.page_record)


@app.post("/simulation/run")
async def simulation_endpoint(payload: SimulationRequest):
    return await simulation.run_mode_a(str(payload.url), payload.query)


@app.post("/simulate")
async def simulate_endpoint(payload: SimulationRequest):
    return await simulation.run_mode_a(str(payload.url), payload.query)


# ==========================================
# Sandbox AEO/GEO Evaluation Endpoints
# ==========================================


class SandboxAnalyzeRequest(BaseModel):
    url: HttpUrl


class SandboxQuestionsRequest(BaseModel):
    run_id: str
    page_intent: str
    page_content: str


class SandboxCompetitorsRequest(BaseModel):
    run_id: str
    questions: List[Dict[str, str]]


class SandboxCrawlAllRequest(BaseModel):
    run_id: str
    sandbox_url: str
    sandbox_data: Dict[str, Any]
    competitor_urls: List[str]


class SandboxRagRequest(BaseModel):
    run_id: str
    questions: List[Dict[str, str]]
    sandbox_url: str
    business_category: Optional[str] = None
    business_competitors: Optional[List[str]] = None
    competitor_urls: Optional[List[Dict[str, Any]]] = None
    sandbox_brand_name: Optional[str] = None
    model_name: Optional[str] = None


class SandboxFinalizeRequest(BaseModel):
    run_id: str
    rag_results: List[Dict[str, Any]]
    questions: List[Dict[str, str]]


@app.post("/sandbox/analyze")
async def sandbox_analyze_endpoint(payload: SandboxAnalyzeRequest):
    """Step 1-2: Analyze sandbox page and extract intent."""
    return await sandbox.analyze_sandbox_page(str(payload.url))


@app.post("/sandbox/questions")
async def sandbox_questions_endpoint(payload: SandboxQuestionsRequest):
    """Step 3: Generate 15 user questions."""
    questions = await sandbox.generate_user_questions(
        payload.page_intent,
        payload.page_content,
    )
    return {"questions": questions}


@app.post("/sandbox/competitors")
async def sandbox_competitors_endpoint(payload: SandboxCompetitorsRequest):
    """Step 4: Extract competitors from LLM answers."""
    result = await sandbox.extract_competitors_from_questions(
        questions=payload.questions,
        page_intent=None,  # Legacy endpoint doesn't have page_intent
        sandbox_title=None,
        sandbox_url=None,
    )
    # Return in old format for backward compatibility (just URLs)
    competitor_urls = [comp["url"] for comp in result.get("competitor_urls", [])]
    return {"competitors": competitor_urls}


@app.post("/sandbox/crawl-all")
async def sandbox_crawl_all_endpoint(payload: SandboxCrawlAllRequest):
    """Step 5: Crawl competitors and store in Pinecone."""
    result = await sandbox.crawl_and_store_competitors(
        payload.run_id,
        payload.sandbox_url,
        payload.sandbox_data,
        payload.competitor_urls,
    )
    return result


@app.post("/sandbox/rag")
async def sandbox_rag_endpoint(payload: SandboxRagRequest):
    """Step 6: Run RAG simulation for each question."""
    results = await sandbox.run_rag_simulation(
        run_id=payload.run_id,
        questions=payload.questions,
        sandbox_url=payload.sandbox_url,
        business_category=payload.business_category,
        business_competitors=payload.business_competitors,
        competitor_urls=payload.competitor_urls,
        sandbox_brand_name=payload.sandbox_brand_name,
        model_name=payload.model_name
    )
    return {"rag_results": results}


@app.post("/sandbox/finalize")
async def sandbox_finalize_endpoint(payload: SandboxFinalizeRequest):
    """Step 7: Calculate final scores and recommendations."""
    scores = await sandbox.calculate_final_scores(
        payload.rag_results,
        payload.questions,
    )
    return {"scores": scores}


# ==========================================
# Async Job Endpoint
# ==========================================


class StartJobRequest(BaseModel):
    run_id: str
    url: HttpUrl
    model_name: Optional[str] = None
    # Two-stage GEO seed (see async_job.run_full_job): mode "stage1" or "baseline"
    seed: Optional[Dict[str, Any]] = None


@app.post("/start-job")
async def start_job_endpoint(payload: StartJobRequest):
    """
    Start async sandbox analysis job.
    Returns immediately, job runs in background.
    """
    import asyncio

    # Start job in background task
    from fastapi import HTTPException

    seed = payload.seed
    if seed is not None:
        mode = seed.get("mode")
        if mode not in ("stage1", "baseline"):
            raise HTTPException(status_code=400, detail="seed.mode must be 'stage1' or 'baseline'")
        if not seed.get("questions"):
            raise HTTPException(status_code=400, detail="seed.questions is required")
        if mode == "baseline" and not seed.get("source_pool"):
            raise HTTPException(status_code=400, detail="seed.source_pool is required for a baseline seed")

    asyncio.create_task(async_job.run_full_job(payload.run_id, str(payload.url), payload.model_name, seed))

    return {"message": "Job started", "run_id": payload.run_id}


# ==========================================
# Stage 1: Real-World Observation
# ==========================================


class Stage1StartRequest(BaseModel):
    observation_id: str
    url: HttpUrl
    queries: Optional[List[str]] = None  # generated from the page when omitted
    runs_per_query: Optional[int] = None  # overrides STAGE1_RUNS_PER_QUERY
    query_count: Optional[int] = None  # overrides STAGE1_QUERY_COUNT
    fetcher: Optional[str] = None  # overrides STAGE1_FETCHER: gemini_grounding | serpapi_aio
    serpapi_gl: Optional[str] = None  # overrides STAGE1_SERPAPI_GL (serpapi_aio only)
    serpapi_hl: Optional[str] = None  # overrides STAGE1_SERPAPI_HL
    serpapi_device: Optional[str] = None  # overrides STAGE1_SERPAPI_DEVICE


@app.post("/stage1/start")
async def stage1_start_endpoint(payload: Stage1StartRequest):
    """
    Start a Stage 1 observation job in the background. The default fetcher is an
    approximate proxy for Google AI Overview citations, not real AI Overview data.
    """
    import asyncio

    from .observation.config import Stage1Config
    from .observation.stage1_job import run_stage1_job

    from fastapi import HTTPException

    overrides = {
        "runs_per_query": payload.runs_per_query,
        "query_count": payload.query_count,
        "fetcher": payload.fetcher,
        "serpapi_gl": payload.serpapi_gl,
        "serpapi_hl": payload.serpapi_hl,
        "serpapi_device": payload.serpapi_device,
    }
    try:
        config = Stage1Config.from_env(overrides)  # reject invalid config before starting
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if config.fetcher == "serpapi_aio":
        # Refuse a run the SerpApi account cannot pay for (the job checks again before fetching)
        from .observation.fetchers import get_observation_fetcher
        from .observation.serpapi_aio import SerpApiError

        planned = (len([q for q in payload.queries if q.strip()]) if payload.queries else config.query_count) * config.runs_per_query
        try:
            await get_observation_fetcher(config.fetcher, config.grounding_model, config.serpapi_settings).preflight(planned)
        except (SerpApiError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    asyncio.create_task(
        run_stage1_job(payload.observation_id, str(payload.url), payload.queries, overrides)
    )
    return {"message": "Stage 1 observation started", "observation_id": payload.observation_id}


class CompareRequest(BaseModel):
    before: Dict[str, Optional[float]]
    after: Dict[str, Optional[float]]


@app.post("/compare")
async def compare_endpoint(payload: CompareRequest):
    """Before/after deltas for two runs' stored metrics (GEO/AEO improvement)."""
    from .geo_metrics import compare_run_metrics

    return compare_run_metrics(payload.before, payload.after)


# ==========================================
# Test LLM Endpoint
# ==========================================


class ChatMessageRequest(BaseModel):
    session_id: str
    content: str
    scope: str  # "sandbox_only" | "sandbox_competitors" | "web_only"
    run_id: Optional[str] = None
    sandbox_url: Optional[str] = None


class TestLLMRequest(BaseModel):
    prompt: str
    model_name: str


# ==========================================
# Chat Interface Endpoint
# ==========================================

@app.post("/chat/message")
async def chat_message_endpoint(payload: ChatMessageRequest):
    """
    Process a chat message using RAG pipeline.
    Reuses existing RAG logic from sandbox.py.
    """
    try:
        result = await chat.process_chat_message(
            session_id=payload.session_id,
            content=payload.content,
            scope=payload.scope,
            run_id=payload.run_id,
            sandbox_url=payload.sandbox_url,
        )
        return result
    except Exception as e:
        import traceback
        return {
            "content": f"Error processing chat message: {str(e)}",
            "citations": [],
            "tokens_in": 0,
            "tokens_out": 0,
            "model_used": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.post("/test/llm")
async def test_llm_endpoint(payload: TestLLMRequest):
    """
    Test endpoint for multi-provider LLM.
    Receives a prompt and returns the LLM response.
    """
    # extract the model name from the payload
    model_name = payload.model_name
    print(f"Using model: {model_name}")
    try:
        # Get multi-provider LLM instance
        llm = get_llm_provider(model_name)
        
        # Prepare messages for chat
        messages = [{"role": "user", "content": payload.prompt}]
        
        # Get response from multi-provider
        response = await llm.chat(
            messages=messages,
        )

        return {
            "response": response,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)

