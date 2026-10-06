-- AlterTable
ALTER TABLE "observation_runs" ADD COLUMN     "credits_assumed" INTEGER,
ADD COLUMN     "timeout_runs" INTEGER;

-- AlterTable
ALTER TABLE "observation_queries" ADD COLUMN     "credits_assumed" INTEGER,
ADD COLUMN     "timeout_runs" INTEGER;

-- AlterTable
ALTER TABLE "observation_responses" ADD COLUMN     "credits_assumed" INTEGER;

