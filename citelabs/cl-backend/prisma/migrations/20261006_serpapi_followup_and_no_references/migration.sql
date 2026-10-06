-- AlterTable
ALTER TABLE "observation_runs" ADD COLUMN     "citation_rate_all_overviews" DOUBLE PRECISION,
ADD COLUMN     "followup_empty_runs" INTEGER,
ADD COLUMN     "overview_retrieved_runs" INTEGER,
ADD COLUMN     "page_token_runs" INTEGER,
ADD COLUMN     "runs_with_references" INTEGER;

-- AlterTable
ALTER TABLE "observation_queries" ADD COLUMN     "citation_rate_all_overviews" DOUBLE PRECISION,
ADD COLUMN     "followup_empty_runs" INTEGER,
ADD COLUMN     "overview_retrieved_runs" INTEGER,
ADD COLUMN     "page_token_runs" INTEGER,
ADD COLUMN     "runs_with_references" INTEGER;

-- AlterTable
ALTER TABLE "observation_responses" ADD COLUMN     "page_token_seen" BOOLEAN;

