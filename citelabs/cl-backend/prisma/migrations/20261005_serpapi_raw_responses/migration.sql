-- AlterTable
ALTER TABLE "observation_runs" ADD COLUMN     "inline_link_rate" DOUBLE PRECISION,
ADD COLUMN     "inline_linked_runs" INTEGER,
ADD COLUMN     "parse_errors" INTEGER;

-- AlterTable
ALTER TABLE "observation_queries" ADD COLUMN     "inline_link_rate" DOUBLE PRECISION,
ADD COLUMN     "inline_linked_runs" INTEGER,
ADD COLUMN     "parse_errors" INTEGER;

-- AlterTable
ALTER TABLE "observation_responses" ADD COLUMN     "parse_error" TEXT,
ADD COLUMN     "ragged_table" BOOLEAN,
ADD COLUMN     "raw_responses" JSONB,
ADD COLUMN     "snippet_links" JSONB,
ADD COLUMN     "tables_seen" INTEGER,
ADD COLUMN     "target_inline_linked" BOOLEAN;

