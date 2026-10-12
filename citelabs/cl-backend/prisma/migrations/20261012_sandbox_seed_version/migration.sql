-- Two-stage GEO: how a seeded Stage 2 pool was built (null = earlier seeding)
ALTER TABLE "sandbox_runs" ADD COLUMN "seed_version" TEXT;
