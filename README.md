# 🌐 CiteLabs — AEO / GEO Evaluation Platform

<div align="center">

![CiteLabs Banner](assets/citelabs-home.png)

[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![Fastify](https://img.shields.io/badge/Fastify-4-black?style=for-the-badge&logo=fastify)](https://fastify.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Python_3.12-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.7_Flash-4285F4?style=for-the-badge&logo=google)](https://ai.google.dev/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?style=for-the-badge&logo=pinecone)](https://www.pinecone.io/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=for-the-badge&logo=supabase)](https://supabase.com/)

**An end-to-end evaluation engine for Answer Engine Optimization (AEO) and Generative Engine Optimization (GEO).**  
Analyze how your domain and competitors perform in AI-powered search engines, generative summaries, and retrieval-augmented answer engines.

</div>

---

## 📌 Overview

Traditional SEO optimizes for 10 blue links; **AEO & GEO** optimize for inclusion and citation in AI-generated answers (e.g. Gemini, ChatGPT, Perplexity).

**CiteLabs** crawls a target website, analyzes its core business intent, synthesizes realistic multi-intent user queries, retrieves competing brand data, embeds domain content into vector indices, and simulates AI search answering engines. It computes measurable brand visibility, citation likelihood, and competitor dominance scores with actionable recommendations.

---

## 📸 Screenshots

<div align="center">

### 1. Analysis Launchpad
*Submit any sandbox URL to initiate comprehensive multi-step AEO/GEO analysis.*

![CiteLabs Launchpad](assets/citelabs-home.png)

<br/>

### 2. Live Pipeline Execution
*Real-time tracking of crawling, semantic classification, competitor retrieval, vector indexing, and RAG simulation.*

![CiteLabs Progress](assets/citelabs-progress.png)

</div>

---

## 🚀 Key Features

- **Automated Web Crawling & Content Extraction**: Fast scraping and parsing of title, heading, and body semantics.
- **AI Domain Summarization & Intent Extraction**: Extracts core business intent and categorizes niche domains using Google Gemini LLMs.
- **Multi-Intent Query Generation**: Generates contextual user questions across three vital search dimensions:
  - **Intent**: Informational questions assessing direct topic relevance.
  - **Experience**: Exploratory queries on user trust, safety, and operational usage.
  - **Transaction**: Buying/deployment intent queries comparing alternatives.
- **Dynamic Competitor Discovery**: Detects direct industry competitors and crawls their public knowledge base for comparison.
- **Vector Search & Semantic Retrieval**: Embeds and indexes content using Pinecone / FAISS vector stores and sentence transformers (`all-MiniLM-L6-v2`).
- **RAG Simulation Engine**: Simulates generative search answer engines under strict citation criteria.
- **Resilient Multi-Model Failover**: Intelligent rate-limiting and automatic model pool rotation across Google Gemini models (`gemini-3.7-flash`, `gemini-3.8-flash`, `gemini-3.5-flash-lite`) to avoid free-tier quota stalls.
- **Two-Stage GEO Validation**: Grounds the competitor source pool in repeated real-world observations (Google AI Overview results via SerpApi, or Gemini with Google Search grounding as an approximate proxy) and measures before/after GEO improvement on a fixed question set and source pool. See [Two-Stage GEO Evaluation](#-two-stage-geo-evaluation).
- **Interactive Analytics Dashboard**: Real-time polling progress bar, citation distributions, chunk analysis, and GEO/AEO scoring breakdown.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[Frontend Dashboard\nNext.js + TailwindCSS\n:3001] -->|REST / Polling| B[Backend API\nFastify + Prisma\n:4000]
    B -->|PostgreSQL| C[(Supabase DB)]
    B -->|Dispatch Job| D[AI Worker Service\nFastAPI + Python 3.12\n:8001]
    D -->|Inference & Embeddings| E[Google Gemini GenAI\nModel Pool Fallback]
    D -->|Vector Indexing| F[(Pinecone / FAISS)]
    D -->|Step Progress| B
```

### Monorepo Components

| Service | Technology | Port | Description |
| :--- | :--- | :--- | :--- |
| **`cl-frontend`** | Next.js 14, React 18, TailwindCSS, Recharts | `3001` | User interface, run initiator, and real-time analytics dashboard. |
| **`cl-backend`** | Fastify 4, TypeScript, Prisma ORM | `4000` | REST API layer, job manager, and Supabase event store. |
| **`cl-workers`** | Python 3.12, FastAPI, Uvicorn, Google GenAI SDK | `8001` | Core AI computation engine, crawler, vector search, and RAG simulator. |

---

## 🔄 7-Step Evaluation Pipeline

```mermaid
graph TD
    S1[1. Crawl Target Site] --> S2[2. Domain Summary & Intent Extraction]
    S2 --> S3[3. Business Category Classification]
    S3 --> S4[4. Question Generation - 15 Queries]
    S4 --> S5[5. Competitor Discovery & Crawling]
    S5 --> S6[6. Vector Indexing Pinecone / FAISS]
    S6 --> S7[7. RAG Simulation & Scoring Calculation]
```

1. **Target Crawl**: Scrapes sandbox URL, extracting metadata, semantic text, and structure.
2. **Domain Summary**: Generates concise, factual semantic summaries to optimize prompt tokens downstream.
3. **Intent & Category**: Infers user search intent and classifies the business domain.
4. **Question Generation**: Produces 15 multi-intent queries targeting typical user decision journeys.
5. **Competitor Discovery**: Identifies competing market entities and retrieves their reference content.
6. **Vector Indexing**: Chunks and embeds text into vector storage for semantic retrieval.
7. **RAG Simulation & Scoring**: Queries the indexed knowledge base through the LLM, tracking mention rates, first-source citations, and competitor dominance.

---

## ⚙️ Getting Started

### Prerequisites

- **Node.js** v18+ and **npm**
- **Python** 3.10+ (recommended 3.12)
- **Supabase** PostgreSQL instance
- **Pinecone** API Key & Index
- **Google Gemini** API Key

---

### Environment Setup

#### 1. Backend (`citelabs/cl-backend/.env`)
```env
PORT=4000
DATABASE_URL="postgresql://postgres:[PASSWORD]@db.[PROJECT_ID].supabase.co:5432/postgres"
WORKER_BASE_URL="http://localhost:8001"
```

#### 2. Worker Service (`citelabs/cl-workers/.env`)
```env
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_MODEL="gemini-3.7-flash"
GEMINI_PRO_MODEL="gemini-3.7-flash"
GEMINI_EMBED_MODEL="gemini-embedding-001"
LLM_PROVIDER="gemini"

PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="citelabs-sandbox"
PINECONE_REGION="us-east-1"
PINECONE_DIMENSION=3072

BACKEND_BASE_URL="http://localhost:4000"
WORKER_BASE_URL="http://localhost:8001"
SUPABASE_URL="https://[PROJECT_ID].supabase.co"
VECTOR_DB_PROVIDER="faiss" # or "pinecone"
```

---

### Running Locally

Run each service in a separate terminal:

#### 1. Start Backend API
```bash
cd citelabs/cl-backend
npm install
npm run prisma:generate
npm run dev
# Server listening on http://localhost:4000
```

#### 2. Start Python AI Worker
```bash
cd citelabs/cl-workers
# Create & activate virtual environment if not already done:
# python -m venv env
# .\env\Scripts\activate  (Windows) or source env/bin/activate (macOS/Linux)
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
# Worker listening on http://localhost:8001
```

#### 3. Start Frontend Dashboard
```bash
cd citelabs/cl-frontend
npm install
npm run dev
# Dashboard available on http://localhost:3001
```

---

## 📊 Evaluation Metrics

- **AEO Score (Answer Engine Optimization)**: Measures likelihood of the target domain being cited as an authoritative direct source in synthesized answers.
- **GEO Score (Generative Engine Optimization)**: Measures aggregate brand visibility, entity mentions, and contextual relevance across generative search responses.
- **Citation Rate**: Percentage of simulated queries where the target domain is explicitly credited as a primary reference.
- **Competitor Dominance**: Share of citations captured by direct competitors in the same niche.

---

## 🧭 Two-Stage GEO Evaluation

The 7-step pipeline builds its competitor knowledge base from LLM reasoning and SERP discovery alone. Nothing ties that knowledge base to what AI search answers actually cite, so there is no real-world check that an optimization closed a gap a search engine itself recognizes. The two-stage system adds that grounding:

```mermaid
flowchart LR
    W[Current website] --> S1[Stage 1: Real-world observation<br/>repeated search-grounded answers]
    S1 --> B[Stage 2: Controlled GEO score BEFORE<br/>source pool seeded from Stage 1]
    B --> O[Website optimization<br/>human step]
    O --> A[Stage 2: Controlled GEO score AFTER<br/>same questions and source pool]
    A --> C[Comparison<br/>GEO Improvement = after − before]
    O -.-> R[Stage 1 again, optional and delayed<br/>did real-world citations change?]
    R -.-> C
```

Stage 1 is a validation layer, not a blocker: the controlled score can always be computed without it (as an unseeded run), and Stage 1 results are used for seeding and for comparison.

> [!IMPORTANT]
> **What Stage 1 data is depends on the fetcher**, which is stored with every observation and shown next to every result:
> - **`gemini_grounding`: an approximate proxy, not Google AI Overview data.** It records Gemini's own search-and-cite behaviour with Google Search grounding. That uses the same underlying Google Search index that AI Overviews draw from, but it is not AI Overview output. Treat these results as an indication of which sources AI search answers tend to cite, never as real AI Overview citations.
> - **`serpapi_aio`: Google AI Overview results as returned by SerpApi** for the run's fixed country (`gl`), language (`hl`) and device. SerpApi is a third-party SERP API; Google offers no official public AI Overview API.
>
> Fetchers implement `ObservationFetcher` (`cl-workers/app/observation/fetchers.py`), so another data source can be added.

### Stage 1: Real-World Observation

Code: `cl-workers/app/observation/`

1. **Queries**: supplied by the caller, or a balanced selection (intent / experience / transaction) from the existing question generator, conditioned on the page's intent, category and domain summary.
2. **Repeated observation**: each query is observed N times through the configured fetcher (see *Stage 1 fetchers* below). Every run's raw ordered citation list is stored.
3. **Union source pool**: per query, `Sq = S1 ∪ S2 ∪ … ∪ SN` over successful runs. Nothing is averaged or dropped; each source keeps its stability (share of runs that cited it), and every run's raw ordered citation list is stored.
4. **Categorisation**: every source domain gets one of `target_company, direct_competitor, indirect_competitor, government, reference, media, community, industry_organization, academic, other`. The target's own domain is matched deterministically; otherwise an LLM classifier decides, with domain heuristics (`.gov`, `.edu`, `.ac.in`, Wikipedia, Reddit, …) as the fallback and a stored cross-check. Categories roll up into **direct business competitors** (sell a competing product) and **AI visibility competitors** (everyone else occupying citation space).
5. **Evidence-preserving compression**: for each (query, source) pair, the model extracts the evidence that could influence the answer or citation decision for that query (facts, statistics, definitions, entities, product details, claims, specs, comparisons, expert statements, quoted excerpts), not a generic summary. Per-source budget: `min(1000, 20000 / sources)`, never below a 200-token floor.

A single failed observation, crawl or compression is recorded and skipped; it never fails the run.

### Stage 1 fetchers

Selected with `STAGE1_FETCHER`, or per request with `"fetcher"` in `POST /api/observation/run`.

**`gemini_grounding`** (default, the proxy): Gemini with Google Search grounding. Grounding source links are redirects, each resolved to the real URL with one `HEAD` request; search utility results (e.g. "current time") are kept in the raw data but flagged as not being web sources.

**`serpapi_aio`** (Google AI Overviews via SerpApi; code: `observation/serpapi_aio.py`):

- **Flow:** `engine=google` with `gl`, `hl`, `device` and `no_cache=true`. If `ai_overview` holds only a `page_token` (it expires in about a minute), `engine=google_ai_overview` is called with it immediately, also with `no_cache=true`. Settings come from `STAGE1_SERPAPI_GL` / `_HL` / `_DEVICE` (defaults `in`, `en`, `desktop`), can be overridden per request, and are fixed for the whole run.
- **Citations** are `ai_overview.references`, in order, and nothing else. **Inline links** inside the answer text (`snippet_links`) are stored per response with a `target_inline_linked` flag and reported as a separate, secondary *inline-linked* rate; they are never counted as citations.
- **Answer text** is every text block's title and snippet, list items, nested blocks, and each table cell (from the table's `detailed` rows).
- **Outcomes** per observation: `answer`; `no_aio` (Google showed no AI Overview: a valid observation, not a failure); `parse_error`; or a failure. **AIO activation rate** = runs with an AI Overview ÷ successful runs.
- **Rates:** mention %, cited %, citation positions and Jaccard are computed over runs that showed an AI Overview. Reported alongside: the activation rate, the *overall cited %* (cited runs ÷ all successful runs, including no-AI-Overview runs), and the number of runs each rate is based on.
- **Strict parsing:** a text block type other than `paragraph`, `heading`, `list`, `table` or `expandable`, or a missing required field, makes that observation a `parse_error`. Its credits still count, the run continues, and the run summary shows the parse-error count. After 3 consecutive parse errors the run stops; observations not yet started are skipped and spend nothing.
- **Credits:** before starting, the worker reads the account's `total_searches_left` (free) and refuses a run whose worst case (2 credits per observation) exceeds it; during the run, requests stop once that number is reached. Credits used are recorded per observation and per run.
- **Raw storage and re-parsing:** both raw responses of every observation are stored, sanitized (API key, account fields, and search-archive/markdown links removed). `research/reparse_serpapi_observation.py` rebuilds every parsed field from the stored responses with the same parser, without spending credits, and reports any difference from the stored values.

**Known limitation (tables):** SerpApi sometimes returns table rows with fewer cells than the header; the dropped cells appear to be ones that were links. Responses with such a table are flagged (`ragged_table`). A brand that appears only in a dropped cell is not in the answer text, so **the mention rate can undercount**.

**How "mentioned" is decided (Stage 1, both fetchers):** two fields are stored per response.

- `mentioned`: a case-insensitive, whole-word string match of the brand name (from intent extraction) or the target's domain against the answer text (`detect_visibility` in `observation/stage1_job.py`). No entity model is involved, and negation is not handled: "there is no mention of Razorpay" counts as a mention.
- `mentioned_excl_absence`: the same match, applied sentence by sentence (split at `.`, `!` or `?` followed by whitespace, and at line breaks). A sentence containing a brand term is an *absence-of-information statement* if it matches one of `ABSENCE_PATTERNS` ("no mention of", "does not mention", "not mentioned", "no information about/on/regarding", "does not contain (any) information", "not enough information", "cannot answer/find", including contracted forms). The brand counts as mentioned only if at least one brand-term sentence is not an absence statement. Ordinary negation is still a mention ("Razorpay does not charge setup fees"). The brand-term sentences are stored per response (`mention_sentences`, `absence_sentences`) for manual checking. Limitation: a sentence that names the brand *and* says something else lacks information (e.g. "There is no information about fees, but Razorpay is popular") is treated as an absence statement. Mention rates for both fields are over the same runs. Stage 2 scoring is unaffected.

"Cited" is separate: the target's domain is among the answer's citations.

### Stage 2: Controlled Generative Environment

The existing pipeline, with its source pool grounded in Stage 1:

- **Seeded "before" run** (`observation_id`): questions are Stage 1's queries, so real-world and controlled results can be compared per query. The source pool is Stage 1's union pool (one page per domain, the most stable; the target's own domain excluded because the sandbox page is always crawled) merged with the run's own SERP/LLM discovery, deduplicated by domain with Stage 1's labels winning. Capped at `STAGE2_MAX_SOURCES`, which trims discovered sources first.
- **"After" run** (`run_tag: "after"`, `baseline_run_id`): reuses the baseline's exact questions and source pool and skips discovery, re-crawling only the pages themselves. The score change then reflects the website, not setup variance.
- **Unseeded run**: unchanged behaviour (same pages crawled), with category labels added.
- Every RAG answer is tagged with the sandbox's citation position and the category of each cited and retrieved source.
- The GEO, AEO and Simulation Reality formulas are unchanged; the new metrics below are reported alongside them.

### Two-Stage Metrics

Implemented as small pure functions in `cl-workers/app/geo_metrics.py` (unit-tested). Rates are percentages.

| Metric | Definition |
| :--- | :--- |
| **Mention Rate** (AI visibility) | Responses that name the brand ÷ total responses |
| **Strict Citation Rate** | Responses that cite the target as a source ÷ total responses. The existing *Citation Rate* in the GEO/AEO formulas counts mentions **or** citations and is kept as is |
| **Mean / Median Citation Position** | The target's 1-based rank among the distinct domains an answer cites, over answers that cite it (lower is better) |
| **Source Stability** | Per source: runs that cited it ÷ N (Stage 1) |
| **Source-Set Diversity** | `\|S1 ∪ … ∪ SN\|` per query (Stage 1) |
| **Source-Set Stability** | Mean pairwise Jaccard `\|Si ∩ Sj\| / \|Si ∪ Sj\|` across a query's runs; 1 means identical sources every run (Stage 1) |
| **AI Visibility Competition** | Share of citations by category and by competitor group |
| **GEO / AEO Improvement** | `Score(after) − Score(before)`, with deltas on every metric above |

### Before/After Workflow

In the UI: start a real-world observation on the home page → **Run controlled GEO score (before)** on the observation page → change the website → **Re-measure after website changes** on the run page → **Compare with baseline**.

Via the API:

```bash
# 1. Stage 1 (small config for testing; omit the numbers to use the server defaults)
curl -X POST localhost:4000/api/observation/run -H 'content-type: application/json' \
  -d '{"url": "https://example.com/page", "runs_per_query": 3, "query_count": 3}'
# poll GET /api/observation/<observation_id> until "status": "completed"

# 2. Controlled score before the change, seeded from Stage 1
curl -X POST localhost:4000/api/sandbox/run -H 'content-type: application/json' \
  -d '{"url": "https://example.com/page", "observation_id": "<observation_id>", "run_tag": "before"}'

# 3. After changing the website: same questions and source pool
curl -X POST localhost:4000/api/sandbox/run -H 'content-type: application/json' \
  -d '{"url": "https://example.com/page", "run_tag": "after", "baseline_run_id": "<before run_id>"}'

# 4. Compare (add &before_observation=…&after_observation=… for real-world deltas
#    once a new Stage 1 observation has been made after the change is indexed)
curl 'localhost:4000/api/sandbox/compare?before=<before run_id>&after=<after run_id>'
```

Add `"dry_run": true` to a `/api/sandbox/run` request to validate it and see the seed without starting a run.

| Endpoint | Purpose |
| :--- | :--- |
| `POST /api/observation/run` | Start Stage 1 (`url`, optional `queries`, `runs_per_query`, `query_count`) |
| `GET /api/observation/:id` | Observation with per-query pools, categories, evidence and metrics (`?raw=true` adds answer texts and raw citation lists) |
| `GET /api/observation?url=` | Recent observations |
| `POST /api/sandbox/run` | Stage 2 run; optional `observation_id`, `run_tag`, `baseline_run_id`, `dry_run` |
| `GET /api/sandbox/compare` | Controlled deltas, per-question changes with Stage 1 metrics attached, optional real-world deltas |

### Configuration (`cl-workers/.env`)

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `STAGE1_RUNS_PER_QUERY` | `15` | N repeated observations per query |
| `STAGE1_QUERY_COUNT` | `10` | Queries selected when generating them |
| `STAGE1_EVIDENCE_TOKEN_BUDGET` | `20000` | Total evidence tokens per query |
| `STAGE1_PER_SOURCE_TOKEN_CAP` | `1000` | Maximum evidence tokens per source |
| `STAGE1_MIN_SOURCE_TOKENS` | `200` | Floor per source; also caps sources per query at budget ÷ floor |
| `STAGE1_COMPRESSION_BATCH_SIZE` | `5` | Evidence calls in flight at once |
| `STAGE1_GROUNDING_MODEL` | `gemini-2.5-flash` | Model for grounded observations (never rotated mid-run) |
| `STAGE1_FETCHER` | `gemini_grounding` | Observation fetcher: `gemini_grounding` (proxy) or `serpapi_aio` (Google AI Overviews via SerpApi) |
| `STAGE1_SKIP_EVIDENCE` | `false` | Observation-only runs: skip crawling cited pages and evidence extraction (saves LLM quota); LLM categorisation still runs. Also `"skip_evidence": true` per request. Shown on the Stage 1 page and in the export |
| `STAGE1_SERPAPI_GL` / `_HL` / `_DEVICE` | `in` / `en` / `desktop` | `serpapi_aio` country, language and device, fixed per run (needs `SERPAPI_KEY`) |
| `SOURCE_CATEGORIES` | built-in list | Comma-separated taxonomy override |
| `STAGE2_MAX_SOURCES` | `40` | Maximum merged source pool for seeded Stage 2 runs |
| `STAGE2_CRAWL_CONCURRENCY` | `8` | Concurrent page fetches in Stage 2 |

Also available: `STAGE1_OBSERVATION_CONCURRENCY`, `STAGE1_CRAWL_CONCURRENCY`, `STAGE1_MAX_PAGE_CHARS`, `STAGE1_CLASSIFICATION_BATCH_SIZE`, `STAGE1_GROUNDING_MAX_RETRIES`, `STAGE1_RESULT_DIR` (writes each raw Stage 1 result to a JSON file) and `STAGE1_SERPAPI_RAW_DIR` (also writes each sanitized raw SerpApi response to a file).

**Cost and time.** Every observation is one grounded call and every (query, source) pair one evidence call, all rate-limited (14 requests per minute per model on the free tier). The defaults (10 × 15) mean roughly 150 grounded calls and a few hundred evidence calls: expect 45+ minutes, and more than a free-tier key's daily quota. For testing, use `runs_per_query` and `query_count` of 3–5 per request.

### Tests

```bash
cd citelabs/cl-workers
pip install -r requirements-dev.txt
python -m pytest
```

Covers the token budget calculator, mean/median position, Jaccard similarity, mention vs citation rate, category breakdown percentages, source pools, seeding and tagging, and the Stage 2 orchestration in each seed mode (with every external call faked).

### Content gap analysis (planned)

`cl-workers/app/gap_analysis.py` defines `find_content_gaps(sandbox_evidence, competitor_evidence_list) -> List[Gap]`: the claims and topics that cited sources cover but the sandbox page does not, built from Stage 1's evidence. It is an interface only for now; the module docstring describes the intended approach.

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).
