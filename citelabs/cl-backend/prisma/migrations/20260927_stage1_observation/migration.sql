-- CreateTable
CREATE TABLE "observation_runs" (
    "id" TEXT NOT NULL,
    "sandbox_url" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "fetcher" TEXT,
    "data_disclaimer" TEXT,
    "config" JSONB,
    "context" JSONB,
    "requested_queries" TEXT[],
    "error" TEXT,
    "runs_attempted" INTEGER,
    "runs_succeeded" INTEGER,
    "unique_sources" INTEGER,
    "unique_domains" INTEGER,
    "mention_rate" DOUBLE PRECISION,
    "citation_rate" DOUBLE PRECISION,
    "mean_citation_position" DOUBLE PRECISION,
    "median_citation_position" DOUBLE PRECISION,
    "mean_source_set_diversity" DOUBLE PRECISION,
    "mean_source_set_stability" DOUBLE PRECISION,
    "mean_domain_set_stability" DOUBLE PRECISION,
    "category_breakdown" JSONB,
    "competitor_group_breakdown" JSONB,
    "source_categories" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "observation_runs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "observation_queries" (
    "id" TEXT NOT NULL,
    "observation_id" TEXT NOT NULL,
    "query_index" INTEGER NOT NULL,
    "query" TEXT NOT NULL,
    "query_type" TEXT,
    "successful_runs" INTEGER NOT NULL DEFAULT 0,
    "source_set_diversity" INTEGER,
    "domain_diversity" INTEGER,
    "source_set_stability" DOUBLE PRECISION,
    "domain_set_stability" DOUBLE PRECISION,
    "mention_rate" DOUBLE PRECISION,
    "citation_rate" DOUBLE PRECISION,
    "mean_citation_position" DOUBLE PRECISION,
    "median_citation_position" DOUBLE PRECISION,
    "target_evidence" JSONB,

    CONSTRAINT "observation_queries_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "observation_responses" (
    "id" TEXT NOT NULL,
    "observation_id" TEXT NOT NULL,
    "query_id" TEXT NOT NULL,
    "run_index" INTEGER NOT NULL,
    "ok" BOOLEAN NOT NULL,
    "error" TEXT,
    "model" TEXT,
    "latency_seconds" DOUBLE PRECISION,
    "answer_text" TEXT,
    "search_queries" TEXT[],
    "target_mentioned" BOOLEAN,
    "target_cited" BOOLEAN,
    "target_position" INTEGER,

    CONSTRAINT "observation_responses_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "observation_citations" (
    "id" TEXT NOT NULL,
    "response_id" TEXT NOT NULL,
    "position" INTEGER NOT NULL,
    "raw_uri" TEXT NOT NULL,
    "url" TEXT,
    "domain" TEXT NOT NULL,
    "title" TEXT,
    "cited_in_answer" BOOLEAN NOT NULL,
    "is_web_source" BOOLEAN NOT NULL,

    CONSTRAINT "observation_citations_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "observation_sources" (
    "id" TEXT NOT NULL,
    "observation_id" TEXT NOT NULL,
    "query_id" TEXT NOT NULL,
    "source_key" TEXT NOT NULL,
    "url" TEXT,
    "domain" TEXT NOT NULL,
    "title" TEXT,
    "category" TEXT NOT NULL,
    "category_source" TEXT NOT NULL,
    "llm_category" TEXT,
    "heuristic_category" TEXT,
    "competitor_group" TEXT NOT NULL,
    "stability" DOUBLE PRECISION NOT NULL,
    "appearances" INTEGER NOT NULL,
    "over_source_cap" BOOLEAN NOT NULL DEFAULT false,
    "crawl_error" TEXT,
    "evidence" TEXT,
    "evidence_token_budget" INTEGER,
    "evidence_token_estimate" INTEGER,
    "has_relevant_evidence" BOOLEAN,
    "evidence_error" TEXT,
    "evidence_latency_seconds" DOUBLE PRECISION,

    CONSTRAINT "observation_sources_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "observation_events" (
    "id" TEXT NOT NULL,
    "event_id" TEXT NOT NULL,
    "observation_id" TEXT NOT NULL,
    "step" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "payload" JSONB,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "observation_events_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "observation_queries_observation_id_idx" ON "observation_queries"("observation_id");

-- CreateIndex
CREATE INDEX "observation_responses_query_id_idx" ON "observation_responses"("query_id");

-- CreateIndex
CREATE INDEX "observation_responses_observation_id_idx" ON "observation_responses"("observation_id");

-- CreateIndex
CREATE INDEX "observation_citations_response_id_idx" ON "observation_citations"("response_id");

-- CreateIndex
CREATE INDEX "observation_citations_domain_idx" ON "observation_citations"("domain");

-- CreateIndex
CREATE INDEX "observation_sources_query_id_idx" ON "observation_sources"("query_id");

-- CreateIndex
CREATE INDEX "observation_sources_observation_id_idx" ON "observation_sources"("observation_id");

-- CreateIndex
CREATE INDEX "observation_sources_domain_idx" ON "observation_sources"("domain");

-- CreateIndex
CREATE UNIQUE INDEX "observation_events_event_id_key" ON "observation_events"("event_id");

-- CreateIndex
CREATE INDEX "observation_events_observation_id_idx" ON "observation_events"("observation_id");

