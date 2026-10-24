-- Competitive gap optimization reports (one row per analysis of a Stage 1 observation)
CREATE TABLE "gap_reports" (
    "id" TEXT NOT NULL,
    "observation_id" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "report" JSONB,
    "error" TEXT,
    "model_calls" INTEGER,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "gap_reports_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "gap_reports_observation_id_idx" ON "gap_reports"("observation_id");
