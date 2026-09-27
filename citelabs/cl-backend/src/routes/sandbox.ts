import { FastifyInstance } from 'fastify';
import { worker } from '../services/pythonClient';
import { Prisma, PrismaClient } from '@prisma/client';
import { resolveRunSeed, RunSeedError } from '../services/runSeed';

const prisma = new PrismaClient();

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL || 'http://localhost:4000';

/**
 * Sanitize text by removing null bytes (0x00) which PostgreSQL cannot store in UTF-8 text fields.
 */
function sanitizeText(text: string | null | undefined): string | null {
  if (!text) return null;
  // Remove null bytes (0x00) which PostgreSQL cannot store
  return text.replace(/\0/g, '');
}

type SandboxRunBody = {
  url: string;
  model_name?: string;
  // Two-stage GEO (see services/runSeed.ts)
  observation_id?: string;
  run_tag?: string;
  baseline_run_id?: string;
  dry_run?: boolean; // validate and return the seed without starting a run
};

type ProgressEventBody = {
  event_id: string;
  job_id: string;
  step: string;
  status: string;
  payload?: any;
  timestamp?: string;
};

type ResultBody = {
  sandbox_page?: any;
  questions?: Array<{ type: string; q: string }>;
  competitors?: Array<any>;
  rag_results?: Array<any>;
  scores?: any;
  // Two-stage GEO
  source_pool?: Array<any>;
  seed_mode?: string | null;
  pool_counts?: Record<string, unknown>;
};

function jsonOrNull(value: unknown): Prisma.InputJsonValue | typeof Prisma.DbNull {
  return value == null ? Prisma.DbNull : (value as Prisma.InputJsonValue);
}

export default async function sandboxRoutes(app: FastifyInstance) {
  // ==========================================
  // POST /api/sandbox/run - Start async job
  // ==========================================
  app.post('/api/sandbox/run', async (request, reply) => {
    const body = request.body as SandboxRunBody;
    const { url, model_name } = body;

    app.log.debug({ url }, 'Received sandbox run request');

    if (!url || typeof url !== 'string') {
      return reply.status(400).send({ error: 'Invalid URL provided' });
    }

    // Two-stage GEO: Stage 1 seeding / before-after tagging
    let resolved;
    try {
      resolved = await resolveRunSeed(prisma, body);
    } catch (err) {
      if (err instanceof RunSeedError) {
        return reply.status(err.statusCode).send({ error: err.message });
      }
      throw err;
    }
    if (body.dry_run) {
      return reply.status(200).send({
        dry_run: true,
        observation_id: resolved.observationId,
        run_tag: resolved.runTag,
        baseline_run_id: resolved.baselineRunId,
        seed: resolved.seed,
      });
    }

    try {
      // Generate run_id
      const runId = `run_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // Create run record
      const run = await prisma.sandboxRun.create({
        data: {
          id: runId,
          sandboxUrl: url,
          status: 'processing',
          observationId: resolved.observationId,
          runTag: resolved.runTag,
          baselineRunId: resolved.baselineRunId,
          seedMode: resolved.seed?.mode ?? null,
        },
      });

      // Fire-and-forget: Start job in Python worker
      worker.startJob({ run_id: runId, url, model_name, seed: resolved.seed ?? undefined }).catch((err) => {
        app.log.error({ err, runId }, 'Failed to start job in worker');
        // Update status to error
        prisma.sandboxRun
          .update({
            where: { id: runId },
            data: { status: 'error' },
          })
          .catch((updateErr) => {
            app.log.error({ err: updateErr, runId }, 'Failed to update run status to error');
          });
      });

      return reply.status(200).send({ run_id: runId });
    } catch (error) {
      app.log.error({ err: error, body }, 'Failed to create sandbox run');
      return reply.status(500).send({ error: 'Internal server error' });
    }
  });

  // ==========================================
  // GET /api/sandbox/:run_id - Get job status
  // ==========================================
  app.get('/api/sandbox/:run_id', async (request, reply) => {
    const { run_id } = request.params as { run_id: string };

    try {
      const run = await prisma.sandboxRun.findUnique({
        where: { id: run_id },
        include: {
          pages: true,
          questions: true,
          competitors: true,
          ragResults: {
            include: {
              questionRef: true,
              metrics: true, // Fetch metrics relation
            } as any, // Type assertion needed until TypeScript server picks up regenerated Prisma types
          },
          scores: true,
          events: {
            orderBy: { timestamp: 'desc' },
            take: 10, // Last 10 events
          },
        },
      }) as any; // Type assertion needed until TypeScript server picks up regenerated Prisma types

      if (!run) {
        return reply.status(404).send({ error: 'Run not found' });
      }

      // Calculate Advanced Metrics Aggregates
      let advancedMetrics = null;
      if (run.ragResults && run.ragResults.length > 0) {
        let totalBrandMentions = 0;
        let totalCompetitorMentions = 0;
        let answersWithBrandMention = 0;
        let answersWithBrandCited = 0;
        let answersWithBrandCitedFirst = 0;
        let answersWithAnyMention = 0; // brand > 0 OR competitor > 0

        run.ragResults.forEach((res: any) => {
          if (res.metrics) {
            totalBrandMentions += res.metrics.brandMentionCount;
            totalCompetitorMentions += res.metrics.competitorMentionCount;

            if (res.metrics.isBrandMentioned) answersWithBrandMention++;
            if (res.metrics.isBrandCited) answersWithBrandCited++;
            if (res.metrics.brandCitedFirst) answersWithBrandCitedFirst++;

            if (res.metrics.brandMentionCount > 0 || res.metrics.competitorMentionCount > 0) {
              answersWithAnyMention++;
            }
          }
        });

        const totalAnswers = run.ragResults.length;

        // 1. AI Share of Voice
        // (brand_mention_count / competitor_mention_count) * 100
        // If competitor mentions are 0, handle gracefully (e.g., if brand > 0 -> 100%, else 0%)
        let aiShareOfVoice = 0;
        if (totalCompetitorMentions > 0) {
          aiShareOfVoice = (totalBrandMentions / totalCompetitorMentions) * 100;
        } else if (totalBrandMentions > 0) {
          aiShareOfVoice = 100;
        }

        // 2. Mention Rate
        // (sum(is_brand_mentioned) / total number of answers) * 100
        const mentionRate = totalAnswers > 0 ? (answersWithBrandMention / totalAnswers) * 100 : 0;

        // 3. First Citation Rate
        // (brand_cited_first / total number of answers where either brand_mention_count > 0 + competitor_mention_count > 0 ) * 100
        const firstCitationRate = answersWithAnyMention > 0 ? (answersWithBrandCitedFirst / answersWithAnyMention) * 100 : 0;

        // 4. Citation Rate
        // (sum(is_brand_cited)/total number of answers )* 100
        const citationRate = totalAnswers > 0 ? (answersWithBrandCited / totalAnswers) * 100 : 0;

        advancedMetrics = {
          ai_share_of_voice: Number(aiShareOfVoice.toFixed(2)),
          mention_rate: Number(mentionRate.toFixed(2)),
          first_citation_rate: Number(firstCitationRate.toFixed(2)),
          citation_rate: Number(citationRate.toFixed(2)),
        };
      }

      // If completed, return full data
      // Two-stage GEO: how this run was seeded and what it reused
      const twoStage = {
        observation_id: run.observationId ?? null,
        run_tag: run.runTag ?? null,
        baseline_run_id: run.baselineRunId ?? null,
        seed_mode: run.seedMode ?? null,
        question_set: run.questionSet ?? null,
        source_pool: run.sourcePool ?? null,
        pool_counts: run.poolCounts ?? null,
      };

      if (run.status === 'completed') {
        return reply.status(200).send({
          run_id: run.id,
          status: run.status,
          sandbox: run.pages[0] || null,
          generated_questions: run.questions || [],
          competitors: run.competitors || [],
          rag_results: run.ragResults || [],
          scores: run.scores || null,
          advanced_metrics: advancedMetrics, // NEW: metrics object
          recommendations: run.scores?.recommendations || [],
          two_stage: twoStage,
        });
      }

      // Otherwise, return status and recent events
      return reply.status(200).send({
        run_id: run.id,
        status: run.status,
        recent_events: run.events || [],
        // Also verify intermediate progress if ragResults exist (optional but helpful)
        rag_results_count: run.ragResults.length,
        two_stage: twoStage,
      });
    } catch (error) {
      app.log.error({ err: error, run_id }, 'Failed to get sandbox run');
      return reply.status(500).send({ error: 'Internal server error' });
    }
  });

  // ==========================================
  // POST /api/sandbox/:run_id/progress - Receive progress from worker
  // ==========================================
  app.post('/api/sandbox/:run_id/progress', async (request, reply) => {
    const { run_id } = request.params as { run_id: string };
    const body = request.body as ProgressEventBody;

    app.log.debug({ run_id, body }, 'Received progress event');

    try {
      // Validate required fields
      if (!body.event_id || !body.step || !body.status) {
        return reply.status(400).send({ error: 'Missing required fields: event_id, step, status' });
      }

      // Check if event already exists (deduplication)
      const existingEvent = await prisma.jobEvent.findUnique({
        where: { eventId: body.event_id },
      });

      if (existingEvent) {
        app.log.debug({ event_id: body.event_id }, 'Duplicate event, skipping');
        return reply.status(200).send({ message: 'Event already processed' });
      }

      // Create event
      await prisma.jobEvent.create({
        data: {
          eventId: body.event_id,
          runId: run_id,
          step: body.step,
          status: body.status,
          payload: body.payload || null,
          timestamp: body.timestamp ? new Date(body.timestamp) : new Date(),
        },
      });

      // Update run status based on step
      // Step names now match status values directly
      let newStatus = run_id; // Default to run_id if step doesn't map
      const stepToStatus: Record<string, string> = {
        crawling_sandbox_site: 'crawling_sandbox_site',
        intent_extraction: 'intent_extraction',
        questions_generation: 'questions_generation',
        competitors_extraction: 'competitors_extraction',
        crawling_competitors: 'crawling_competitors',
        rag_simulation: 'rag_simulation',
        score_calculation: 'score_calculation',
      };

      if (stepToStatus[body.step]) {
        newStatus = stepToStatus[body.step];
      }

      // Update run status
      await prisma.sandboxRun.update({
        where: { id: run_id },
        data: { status: newStatus },
      });

      return reply.status(200).send({ message: 'Progress event recorded' });
    } catch (error) {
      app.log.error({ err: error, run_id, body }, 'Failed to process progress event');
      return reply.status(500).send({ error: 'Internal server error' });
    }
  });

  // ==========================================
  // POST /api/sandbox/:run_id/result - Receive final result from worker
  // ==========================================
  app.post('/api/sandbox/:run_id/result', async (request, reply) => {
    const { run_id } = request.params as { run_id: string };
    const body = request.body as ResultBody;

    app.log.debug({ run_id }, 'Received final result');

    try {
      // Verify run exists
      const run = await prisma.sandboxRun.findUnique({
        where: { id: run_id },
      });

      if (!run) {
        return reply.status(404).send({ error: 'Run not found' });
      }

      // Store sandbox page
      if (body.sandbox_page) {
        // Delete existing page for this run if any
        await prisma.sandboxPage.deleteMany({
          where: { runId: run_id },
        });

        // Create new page
        await prisma.sandboxPage.create({
          data: {
            id: `${run_id}_page`,
            runId: run_id,
            url: body.sandbox_page.url || run.sandboxUrl,
            title: sanitizeText(body.sandbox_page.title),
            h1: sanitizeText(body.sandbox_page.h1),
            intent: sanitizeText(body.sandbox_page.intent),
            pageSummary: sanitizeText(body.sandbox_page.page_summary),
            aiSummary: sanitizeText(body.sandbox_page.ai_summary),
            fullText: sanitizeText(body.sandbox_page.full_text),
            isClient: true,
          },
        });
      }

      // Store questions
      if (body.questions && body.questions.length > 0) {
        // Delete existing questions for this run
        await prisma.sandboxQuestion.deleteMany({
          where: { runId: run_id },
        });

        // Create new questions
        for (const q of body.questions) {
          await prisma.sandboxQuestion.create({
            data: {
              id: `q_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              runId: run_id,
              type: q.type,
              question: sanitizeText(q.q) || '',
            },
          });
        }
      }

      // Store competitors
      if (body.competitors && body.competitors.length > 0) {
        // Delete existing competitors
        await prisma.sandboxCompetitor.deleteMany({
          where: { runId: run_id },
        });

        // Create new competitors
        for (const comp of body.competitors) {
          await prisma.sandboxCompetitor.create({
            data: {
              id: `comp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              runId: run_id,
              url: comp.url,
              domain: comp.domain,
              title: sanitizeText(comp.title),
              h1: sanitizeText(comp.h1),
              aiSummary: sanitizeText(comp.ai_summary),
              fullText: sanitizeText(comp.full_text),
              sourceType: comp.source_type || 'competitor', // Use source_type from data (serp/llm), fallback to 'competitor'
              domainType: comp.domain_type || 'editorial', // P2: Store domain type (sandbox_brand/business_competitor/editorial)
              sourceCategory: comp.source_category ?? null,
              competitorGroup: comp.competitor_group ?? null,
              categorySource: comp.category_source ?? null,
              origin: comp.origin ?? null,
            } as any, // Type assertion needed until TypeScript server picks up regenerated Prisma types
          });
        }
      }

      // CRITICAL: Store RAG results - these become the SOURCE OF TRUTH for analytics.
      // Each row in sandbox_rag_results represents ONE question with FINAL citation arrays.
      // Citation arrays (sandbox_citations, business_competitor_citations, etc.) are FINALIZED at ingestion.
      // Analytics queries MUST count DISTINCT question_id from sandbox_rag_results, NOT chunks.
      // DO NOT rely on Pinecone or runtime chunk retrieval for analytics - use sandbox_rag_results only.
      if (body.rag_results && body.rag_results.length > 0) {
        // Get stored questions
        const storedQuestions = await prisma.sandboxQuestion.findMany({
          where: { runId: run_id },
        });

        // Delete existing RAG results (ensures idempotency)
        await prisma.sandboxRagResult.deleteMany({
          where: { runId: run_id },
        });

        // Create new RAG results - ONE ROW PER QUESTION with FINAL citation arrays
        for (let i = 0; i < body.rag_results.length && i < storedQuestions.length; i++) {
          const ragResult = body.rag_results[i];
          const question = storedQuestions[i];
          const ragResultId = `rag_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

          // Persist FINAL citation arrays - these are the source of truth for analytics
          await prisma.sandboxRagResult.create({
            data: {
              id: ragResultId,
              runId: run_id,
              questionId: question.id,  // Link to question for distinct counting
              question: sanitizeText(ragResult.question) || '',
              answer: sanitizeText(ragResult.answer) || '',
              didSandboxAppear: ragResult.did_sandbox_appear || false,
              chunksUsed: ragResult.chunks_used || 0,
              confidence: ragResult.confidence || null,
              competitorCitations: ragResult.competitor_citations || [],  // FINAL - source of truth
              sandboxCitations: ragResult.sandbox_citations || [],  // FINAL - source of truth
              businessCompetitorCitations: ragResult.business_competitor_citations || [],  // FINAL - source of truth
              authoritySourceCitations: ragResult.authority_source_citations || [],  // FINAL - source of truth
              diagnosisType: ragResult.diagnosis_type ?? null,
              diagnosisDetail: sanitizeText(ragResult.diagnosis_detail) ?? null,
              sandboxCitationPosition: ragResult.sandbox_citation_position ?? null,
              citedSourceCategories: jsonOrNull(ragResult.cited_source_categories),
              retrievedSourceCategories: jsonOrNull(ragResult.retrieved_source_categories),
              metrics: ragResult.metrics ? {
                create: {
                  id: `metrics_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                  // answerId is automatically set by Prisma from the parent relation
                  brandMentionCount: ragResult.metrics.brand_mention_count || 0,
                  competitorMentionCount: ragResult.metrics.competitor_mention_count || 0,
                  isBrandMentioned: ragResult.metrics.is_brand_mentioned || false,
                  isBrandCited: ragResult.metrics.is_brand_cited || false,
                  brandCitedFirst: ragResult.metrics.brand_cited_first || false,
                }
              } : undefined,
            } as any, // Type assertion needed until TypeScript server picks up regenerated Prisma types
          });
        }
      }

      // Store scores
      if (body.scores) {
        const twoStageScores = {
          responsesEvaluated: body.scores.responses_evaluated ?? null,
          mentionRate: body.scores.mention_rate ?? null,
          strictCitationRate: body.scores.strict_citation_rate ?? null,
          meanCitationPosition: body.scores.mean_citation_position ?? null,
          medianCitationPosition: body.scores.median_citation_position ?? null,
          categoryBreakdown: jsonOrNull(body.scores.category_breakdown),
          competitorGroupBreakdown: jsonOrNull(body.scores.competitor_group_breakdown),
        };
        await prisma.sandboxScore.upsert({
          where: { runId: run_id },
          create: {
            id: `score_${Date.now()}`,
            runId: run_id,
            geoScore: body.scores.geo_score || 0,
            aeoScore: body.scores.aeo_score || 0,
            citationRate: body.scores.citation_rate || 0,
            coverage: body.scores.coverage || null,
            competitorDominance: body.scores.competitor_dominance || null,
            missingContent: (body.scores.missing_content || []).map((item: string) => sanitizeText(item) || ''),
            schemaGaps: (body.scores.schema_gaps || []).map((item: string) => sanitizeText(item) || ''),
            structuredDataOps: (body.scores.structured_data_ops || []).map((item: string) => sanitizeText(item) || ''),
            recommendations: (body.scores.recommendations || []).map((item: string) => sanitizeText(item) || ''),
            ...twoStageScores,
          },
          update: {
            geoScore: body.scores.geo_score || 0,
            aeoScore: body.scores.aeo_score || 0,
            citationRate: body.scores.citation_rate || 0,
            coverage: body.scores.coverage || null,
            competitorDominance: body.scores.competitor_dominance || null,
            missingContent: (body.scores.missing_content || []).map((item: string) => sanitizeText(item) || ''),
            schemaGaps: (body.scores.schema_gaps || []).map((item: string) => sanitizeText(item) || ''),
            structuredDataOps: (body.scores.structured_data_ops || []).map((item: string) => sanitizeText(item) || ''),
            recommendations: (body.scores.recommendations || []).map((item: string) => sanitizeText(item) || ''),
            ...twoStageScores,
          },
        });
      }

      // Mark run as completed, storing the exact question set and source pool
      // so an "after" run can reuse them
      await prisma.sandboxRun.update({
        where: { id: run_id },
        data: {
          status: 'completed',
          questionSet: jsonOrNull((body.questions || []).filter((q) => q.type !== 'SERP')),
          sourcePool: jsonOrNull(body.source_pool),
          poolCounts: jsonOrNull(body.pool_counts),
        },
      });

      return reply.status(200).send({ message: 'Result stored successfully' });
    } catch (error) {
      app.log.error({ err: error, run_id }, 'Failed to store final result');
      return reply.status(500).send({ error: 'Internal server error' });
    }
  });

  // ==========================================
  // GET /api/sandbox/chunks - Query chunks by run_id and domain_type
  // ==========================================
  app.get('/api/sandbox/chunks', async (request, reply) => {
    const query = request.query as { run_id?: string; domain_type?: string; limit?: string };

    if (!query.run_id) {
      return reply.status(400).send({ error: 'run_id is required' });
    }

    try {
      const where: any = {
        runId: query.run_id,
      };

      if (query.domain_type) {
        where.domainType = query.domain_type;
      }

      const limit = query.limit ? parseInt(query.limit, 10) : 10;

      const chunks = await prisma.sandboxChunk.findMany({
        where,
        orderBy: {
          createdAt: 'desc',
        },
        take: limit,
        select: {
          id: true,
          runId: true,
          url: true,
          chunkId: true,
          text: true,
          summary: true,
          isClient: true,
          domain: true,
          sourceType: true,
          domainType: true,
          createdAt: true,
        },
      });

      return reply.status(200).send({
        chunks: chunks.map((chunk) => ({
          id: chunk.id,
          run_id: chunk.runId,
          url: chunk.url,
          chunk_id: chunk.chunkId,
          text: chunk.text,
          summary: chunk.summary,
          is_client: chunk.isClient,
          domain: chunk.domain,
          source_type: chunk.sourceType,
          domain_type: chunk.domainType,
          created_at: chunk.createdAt.toISOString(),
        })),
      });
    } catch (error) {
      app.log.error({ err: error, query }, 'Failed to query sandbox chunks');
      return reply.status(500).send({ error: 'Internal server error' });
    }
  });

  // ==========================================
  // POST /api/sandbox/:run_id/chunks - Receive chunks from worker
  // ==========================================
  app.post('/api/sandbox/:run_id/chunks', async (request, reply) => {
    const { run_id } = request.params as { run_id: string };
    const body = request.body as { chunks: Array<any> };

    app.log.debug({ run_id, chunk_count: body.chunks?.length }, 'Received chunks for persistence');

    if (!body.chunks || !Array.isArray(body.chunks)) {
      return reply.status(400).send({ error: 'Missing or invalid chunks array' });
    }

    try {
      // Verify run exists
      const run = await prisma.sandboxRun.findUnique({
        where: { id: run_id },
      });

      if (!run) {
        return reply.status(404).send({ error: 'Run not found' });
      }

      // Prepare chunks for batch insertion
      const chunkData = body.chunks.map((chunk) => ({
        id: `chunk_${run_id}_${chunk.chunk_id}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        runId: run_id,
        url: chunk.url || '',
        chunkId: chunk.chunk_id || 0,
        text: sanitizeText(chunk.text) || '',
        summary: sanitizeText(chunk.summary),
        isClient: chunk.is_client || false,
        domain: chunk.domain || null,
        sourceType: chunk.source_type || 'competitor',
        domainType: chunk.domain_type || 'editorial',
      }));

      // Batch insert chunks (Prisma handles batching internally, but we'll do it in batches of 100)
      const batchSize = 100;
      let insertedCount = 0;
      let skippedCount = 0;

      for (let i = 0; i < chunkData.length; i += batchSize) {
        const batch = chunkData.slice(i, i + batchSize);

        try {
          // Use createMany for better performance, but handle duplicates gracefully
          await prisma.sandboxChunk.createMany({
            data: batch as any, // Type assertion needed until TypeScript server picks up regenerated Prisma types
            skipDuplicates: true, // Skip if (runId, url, chunkId) already exists
          });
          insertedCount += batch.length;
        } catch (error: any) {
          // If createMany fails (e.g., unique constraint), try individual inserts
          app.log.warn({ err: error, batch_start: i }, 'Batch insert failed, trying individual inserts');

          for (const chunk of batch) {
            try {
              await prisma.sandboxChunk.create({
                data: chunk as any,
              });
              insertedCount += 1;
            } catch (individualError: any) {
              // Check if it's a duplicate error
              if (individualError.code === 'P2002' || individualError.message?.includes('Unique constraint')) {
                skippedCount += 1;
                app.log.debug({ chunk_id: chunk.chunkId, url: chunk.url }, 'Skipping duplicate chunk');
              } else {
                app.log.warn({ err: individualError, chunk }, 'Failed to insert individual chunk');
              }
            }
          }
        }
      }

      app.log.info(
        { run_id, total: chunkData.length, inserted: insertedCount, skipped: skippedCount },
        'Chunks persisted successfully'
      );

      return reply.status(200).send({
        message: 'Chunks persisted successfully',
        total: chunkData.length,
        inserted: insertedCount,
        skipped: skippedCount,
      });
    } catch (error) {
      app.log.error({ err: error, run_id, chunk_count: body.chunks?.length }, 'Failed to persist chunks');
      // Return 200 to not break worker execution, but log the error
      return reply.status(200).send({
        message: 'Chunk persistence attempted (may have failed)',
        error: 'Internal error occurred',
      });
    }
  });

  // ==========================================
  // GET /api/sandbox/analytics/questions-with-brand - Get questions where brand was cited
  // ==========================================
  // CRITICAL: Analytics queries MUST use sandbox_rag_results ONLY, never Pinecone or chunks.
  // Why:
  // 1. sandbox_rag_results contains FINAL citations from RAG generation (one row per question)
  // 2. Counting chunks would be incorrect - we need to count DISTINCT questions, not chunks
  // 3. Pinecone retrieval is non-deterministic and can change over time
  // 4. Citation arrays in sandbox_rag_results are the source of truth, finalized at ingestion
  app.get('/api/sandbox/analytics/questions-with-brand', async (request, reply) => {
    const query = request.query as { run_id?: string; brand_domain?: string };

    if (!query.run_id) {
      return reply.status(400).send({ error: 'run_id is required' });
    }

    try {
      // Query sandbox_rag_results directly - this is the source of truth for citations
      // Each row represents ONE question with FINAL citation arrays
      const ragResults = await prisma.sandboxRagResult.findMany({
        where: {
          runId: query.run_id,
        },
        select: {
          questionId: true,  // Include questionId to ensure we're counting distinct questions
          question: true,
          answer: true,
          sandboxCitations: true,
          businessCompetitorCitations: true,
        },
      });

      // Filter results where brand domain appears in citations
      // CRITICAL: We count DISTINCT questions, not chunks
      const questions_with_brand: Array<{ question: string; answer: string; citations: string[] }> = [];

      if (query.brand_domain) {
        const brand_domain_lower = query.brand_domain.toLowerCase();

        for (const result of ragResults) {
          // Check if brand domain is in any citation array (case-insensitive)
          // Using FINAL citation arrays from sandbox_rag_results, not runtime chunk retrieval
          const in_sandbox = result.sandboxCitations.some((url: string) =>
            url.toLowerCase().includes(brand_domain_lower)
          );
          const in_business_competitor = result.businessCompetitorCitations.some((url: string) =>
            url.toLowerCase().includes(brand_domain_lower)
          );

          if (in_sandbox || in_business_competitor) {
            questions_with_brand.push({
              question: result.question,
              answer: result.answer,
              citations: [
                ...result.sandboxCitations,
                ...result.businessCompetitorCitations,
              ],
            });
          }
        }
      } else {
        // If no brand_domain provided, return all questions with any sandbox/business_competitor citations
        for (const result of ragResults) {
          if (result.sandboxCitations.length > 0 || result.businessCompetitorCitations.length > 0) {
            questions_with_brand.push({
              question: result.question,
              answer: result.answer,
              citations: [
                ...result.sandboxCitations,
                ...result.businessCompetitorCitations,
              ],
            });
          }
        }
      }

      // Return count of DISTINCT questions (not chunks)
      return reply.status(200).send({
        questions: questions_with_brand,
        count: questions_with_brand.length,  // This is the count of distinct questions, not chunks
      });
    } catch (error) {
      app.log.error({ err: error, query }, 'Failed to query questions with brand');
      return reply.status(500).send({ error: 'Internal server error' });
    }
  });

  // ==========================================
  // GET /api/sandbox/analytics/total-questions - Get total question count
  // ==========================================
  // CRITICAL: Count questions from sandbox_questions table, not chunks or RAG results.
  // This ensures analytics counts are stable and based on the actual question set.
  app.get('/api/sandbox/analytics/total-questions', async (request, reply) => {
    const query = request.query as { run_id?: string };

    if (!query.run_id) {
      return reply.status(400).send({ error: 'run_id is required' });
    }

    try {
      // Count distinct questions from sandbox_questions table
      // This is the source of truth for the total question count
      const count = await prisma.sandboxQuestion.count({
        where: {
          runId: query.run_id,
        },
      });

      return reply.status(200).send({
        total: count,  // Total number of questions in the run
      });
    } catch (error) {
      app.log.error({ err: error, query }, 'Failed to get total question count');
      return reply.status(500).send({ error: 'Internal server error' });
    }
  });

  // ==========================================
  // Legacy endpoint (keep for backward compatibility, but mark as deprecated)
  // ==========================================
  app.post('/api/sandbox/analyze', async (request, reply) => {
    return reply.status(410).send({
      error: 'This endpoint is deprecated. Use POST /api/sandbox/run instead.',
    });
  });
}
