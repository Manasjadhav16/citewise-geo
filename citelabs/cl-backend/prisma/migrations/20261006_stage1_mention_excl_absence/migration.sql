-- AlterTable
ALTER TABLE "observation_runs" ADD COLUMN     "mention_rate_excl_absence" DOUBLE PRECISION;

-- AlterTable
ALTER TABLE "observation_queries" ADD COLUMN     "mention_rate_excl_absence" DOUBLE PRECISION;

-- AlterTable
ALTER TABLE "observation_responses" ADD COLUMN     "absence_sentences" JSONB,
ADD COLUMN     "mention_sentences" JSONB,
ADD COLUMN     "target_mentioned_excl_absence" BOOLEAN;

