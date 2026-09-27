-- AlterTable
ALTER TABLE "sandbox_runs" ADD COLUMN     "baseline_run_id" TEXT,
ADD COLUMN     "observation_id" TEXT,
ADD COLUMN     "pool_counts" JSONB,
ADD COLUMN     "question_set" JSONB,
ADD COLUMN     "run_tag" TEXT,
ADD COLUMN     "seed_mode" TEXT,
ADD COLUMN     "source_pool" JSONB;

-- AlterTable
ALTER TABLE "sandbox_competitors" ADD COLUMN     "category_source" TEXT,
ADD COLUMN     "competitor_group" TEXT,
ADD COLUMN     "origin" TEXT,
ADD COLUMN     "source_category" TEXT;

-- AlterTable
ALTER TABLE "sandbox_rag_results" ADD COLUMN     "cited_source_categories" JSONB,
ADD COLUMN     "retrieved_source_categories" JSONB,
ADD COLUMN     "sandbox_citation_position" INTEGER;

-- AlterTable
ALTER TABLE "sandbox_scores" ADD COLUMN     "category_breakdown" JSONB,
ADD COLUMN     "competitor_group_breakdown" JSONB,
ADD COLUMN     "mean_citation_position" DOUBLE PRECISION,
ADD COLUMN     "median_citation_position" DOUBLE PRECISION,
ADD COLUMN     "mention_rate" DOUBLE PRECISION,
ADD COLUMN     "responses_evaluated" INTEGER,
ADD COLUMN     "strict_citation_rate" DOUBLE PRECISION;

-- CreateIndex
CREATE INDEX "sandbox_runs_observation_id_idx" ON "sandbox_runs"("observation_id");

-- CreateIndex
CREATE INDEX "sandbox_runs_baseline_run_id_idx" ON "sandbox_runs"("baseline_run_id");

