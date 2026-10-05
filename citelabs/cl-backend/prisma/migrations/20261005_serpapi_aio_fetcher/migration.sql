-- AlterTable
ALTER TABLE "observation_runs" ADD COLUMN     "aio_activation_rate" DOUBLE PRECISION,
ADD COLUMN     "aio_runs" INTEGER,
ADD COLUMN     "credits_used" INTEGER,
ADD COLUMN     "fetcher_settings" JSONB,
ADD COLUMN     "overall_citation_rate" DOUBLE PRECISION,
ADD COLUMN     "preflight" JSONB,
ADD COLUMN     "rates_based_on_runs" INTEGER;

-- AlterTable
ALTER TABLE "observation_queries" ADD COLUMN     "aio_activation_rate" DOUBLE PRECISION,
ADD COLUMN     "aio_runs" INTEGER,
ADD COLUMN     "credits_used" INTEGER,
ADD COLUMN     "overall_citation_rate" DOUBLE PRECISION,
ADD COLUMN     "rates_based_on_runs" INTEGER;

-- AlterTable
ALTER TABLE "observation_responses" ADD COLUMN     "credits_used" INTEGER,
ADD COLUMN     "fetch_meta" JSONB,
ADD COLUMN     "outcome" TEXT;

