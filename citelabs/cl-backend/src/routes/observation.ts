import { randomUUID } from 'crypto';
import { FastifyInstance } from 'fastify';
import { Prisma, PrismaClient } from '@prisma/client';
import { worker } from '../services/pythonClient';

const prisma = new PrismaClient();

/**
 * Stage 1: Real-World Observation.
 *
 * The default worker fetcher records Gemini's answers with Google Search grounding:
 * an approximate proxy for Google AI Overview citations, NOT real AI Overview data.
 * Every response from these routes carries the fetcher's data_disclaimer.
 */

// The worker posts the full raw result (answer texts, citations, evidence) in one body
const RESULT_BODY_LIMIT = 50 * 1024 * 1024;

type ObservationRunBody = {
  url?: string;
  queries?: string[];
  runs_per_query?: number;
  query_count?: number;
  // Optional overrides of the worker's STAGE1_FETCHER / STAGE1_SERPAPI_* settings
  fetcher?: 'gemini_grounding' | 'serpapi_aio';
  serpapi_gl?: string;
  serpapi_hl?: string;
  serpapi_device?: 'desktop' | 'mobile' | 'tablet';
};

type ProgressBody = {
  event_id?: string;
  step?: string;
  status?: string;
  payload?: Record<string, unknown>;
  timestamp?: string;
};

type WorkerCitation = {
  raw_uri: string;
  url: string | null;
  domain: string;
  title: string | null;
  order: number;
  cited_in_answer: boolean;
  is_web_source: boolean;
};

type WorkerRun = {
  run_index: number;
  ok: boolean;
  error: string | null;
  outcome?: string; // answer | no_aio | parse_error | error
  credits_used?: number | null;
  fetch_meta?: Record<string, unknown> | null;
  parse_error?: string | null;
  tables_seen?: number | null;
  ragged_table?: boolean | null;
  inline_links?: { text: string; link: string }[] | null; // snippet_links: not citations
  target_inline_linked?: boolean | null;
  raw_responses?: Record<string, unknown>[] | null;
  model?: string;
  latency_seconds?: number;
  answer_text?: string;
  search_queries?: string[];
  target?: { mentioned: boolean; cited: boolean; citation_position: number | null } | null;
  mentioned_excl_absence?: boolean | null;
  mention_sentences?: string[] | null;
  absence_sentences?: string[] | null;
  citations: WorkerCitation[];
};

type WorkerSource = {
  key: string;
  url: string | null;
  domain: string;
  title: string | null;
  category: string;
  category_source: string;
  llm_category: string | null;
  heuristic_category: string | null;
  competitor_group: string;
  stability: number;
  appearances: number;
  over_source_cap: boolean;
  crawl_error: string | null;
  evidence: string | null;
  evidence_token_budget: number | null;
  evidence_token_estimate: number | null;
  has_relevant_evidence: boolean | null;
  evidence_error: string | null;
  evidence_latency_seconds: number | null;
};

type WorkerQuery = {
  query_index: number;
  query: string;
  query_type: string;
  runs: WorkerRun[];
  sources: WorkerSource[];
  target_evidence: Record<string, unknown> | null;
  metrics: {
    successful_runs: number;
    source_set_diversity: number;
    domain_diversity: number;
    source_set_stability: number | null;
    domain_set_stability: number | null;
    mention_rate: number | null; // null: no successful observations
    mention_rate_excl_absence?: number | null;
    citation_rate: number | null;
    mean_citation_position: number | null;
    median_citation_position: number | null;
    rates_based_on_runs?: number;
    aio_runs?: number | null;
    aio_activation_rate?: number | null;
    overall_citation_rate?: number | null;
    credits_used?: number | null;
    parse_errors?: number;
    inline_linked_runs?: number | null;
    inline_link_rate?: number | null;
  };
};

type WorkerResult = {
  status: 'completed' | 'failed';
  error?: string;
  fetcher?: string;
  data_disclaimer?: string;
  fetcher_settings?: Record<string, unknown>;
  preflight?: Record<string, unknown>;
  config?: Record<string, unknown>;
  context?: Record<string, unknown>;
  metrics?: Record<string, any>;
  source_categories?: Record<string, unknown>;
  queries?: WorkerQuery[];
};

/** PostgreSQL text cannot contain null bytes. */
function clean(text: string | null | undefined): string | null {
  return text == null ? null : text.replace(/\0/g, '');
}

function toJson(value: unknown): Prisma.InputJsonValue | typeof Prisma.DbNull {
  return value == null ? Prisma.DbNull : (value as Prisma.InputJsonValue);
}

function isPositiveInt(value: unknown): boolean {
  return value === undefined || (Number.isInteger(value) && (value as number) >= 1);
}

export default async function observationRoutes(app: FastifyInstance) {
  // ==========================================
  // POST /api/observation/run - Start a Stage 1 observation
  // ==========================================
  app.post('/api/observation/run', async (request, reply) => {
    const body = (request.body || {}) as ObservationRunBody;
    const { url, queries, runs_per_query, query_count, fetcher, serpapi_gl, serpapi_hl, serpapi_device } = body;

    try {
      new URL(url || '');
    } catch {
      return reply.status(400).send({ error: 'Invalid URL provided' });
    }
    if (queries !== undefined && (!Array.isArray(queries) || queries.some((q) => typeof q !== 'string'))) {
      return reply.status(400).send({ error: 'queries must be an array of strings' });
    }
    if (!isPositiveInt(runs_per_query) || !isPositiveInt(query_count)) {
      return reply.status(400).send({ error: 'runs_per_query and query_count must be positive integers' });
    }

    const observationId = `obs_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
    await prisma.observationRun.create({
      data: {
        id: observationId,
        sandboxUrl: url!,
        status: 'pending',
        requestedQueries: queries ?? [],
        config: toJson({ runs_per_query, query_count, fetcher, serpapi_gl, serpapi_hl, serpapi_device }),
      },
    });

    try {
      // Awaited (the worker returns immediately) so an invalid config comes back as an error
      await worker.startStage1({
        observation_id: observationId,
        url: url!,
        queries,
        runs_per_query,
        query_count,
        fetcher,
        serpapi_gl,
        serpapi_hl,
        serpapi_device,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      app.log.error({ err, observationId }, 'Failed to start Stage 1 in worker');
      await prisma.observationRun.update({ where: { id: observationId }, data: { status: 'failed', error: message } });
      return reply.status(502).send({ error: 'Failed to start Stage 1 observation', detail: message, observation_id: observationId });
    }

    return reply.status(200).send({ observation_id: observationId });
  });

  // ==========================================
  // POST /api/observation/:id/progress - Progress event from the worker
  // ==========================================
  app.post('/api/observation/:id/progress', async (request, reply) => {
    const { id } = request.params as { id: string };
    const body = (request.body || {}) as ProgressBody;
    if (!body.event_id || !body.step || !body.status) {
      return reply.status(400).send({ error: 'Missing required fields: event_id, step, status' });
    }

    const run = await prisma.observationRun.findUnique({ where: { id }, select: { id: true } });
    if (!run) {
      return reply.status(404).send({ error: 'Observation not found' });
    }
    const existing = await prisma.observationEvent.findUnique({ where: { eventId: body.event_id } });
    if (existing) {
      return reply.status(200).send({ message: 'Event already processed' });
    }

    await prisma.observationEvent.create({
      data: {
        eventId: body.event_id,
        observationId: id,
        step: body.step,
        status: body.status,
        payload: toJson(body.payload),
        timestamp: body.timestamp ? new Date(body.timestamp) : new Date(),
      },
    });
    // "completed" is set by the result callback, once the data is stored
    await prisma.observationRun.update({
      where: { id },
      data: { status: body.status === 'failed' ? 'failed' : 'running' },
    });
    return reply.status(200).send({ message: 'Progress event recorded' });
  });

  // ==========================================
  // POST /api/observation/:id/result - Full raw result from the worker
  // ==========================================
  app.post('/api/observation/:id/result', { bodyLimit: RESULT_BODY_LIMIT }, async (request, reply) => {
    const { id } = request.params as { id: string };
    const result = (request.body || {}) as WorkerResult;

    const run = await prisma.observationRun.findUnique({ where: { id }, select: { id: true } });
    if (!run) {
      return reply.status(404).send({ error: 'Observation not found' });
    }

    const m = result.metrics || {};
    const queryRows: Prisma.ObservationQueryCreateManyInput[] = [];
    const responseRows: Prisma.ObservationResponseCreateManyInput[] = [];
    const citationRows: Prisma.ObservationCitationCreateManyInput[] = [];
    const sourceRows: Prisma.ObservationSourceCreateManyInput[] = [];

    for (const q of result.queries || []) {
      const queryId = randomUUID();
      queryRows.push({
        id: queryId,
        observationId: id,
        queryIndex: q.query_index,
        query: clean(q.query) || '',
        queryType: q.query_type,
        successfulRuns: q.metrics.successful_runs,
        sourceSetDiversity: q.metrics.source_set_diversity,
        domainDiversity: q.metrics.domain_diversity,
        sourceSetStability: q.metrics.source_set_stability,
        domainSetStability: q.metrics.domain_set_stability,
        mentionRate: q.metrics.mention_rate,
        mentionRateExclAbsence: q.metrics.mention_rate_excl_absence ?? null,
        citationRate: q.metrics.citation_rate,
        meanCitationPosition: q.metrics.mean_citation_position,
        medianCitationPosition: q.metrics.median_citation_position,
        ratesBasedOnRuns: q.metrics.rates_based_on_runs ?? null,
        aioRuns: q.metrics.aio_runs ?? null,
        aioActivationRate: q.metrics.aio_activation_rate ?? null,
        overallCitationRate: q.metrics.overall_citation_rate ?? null,
        creditsUsed: q.metrics.credits_used ?? null,
        parseErrors: q.metrics.parse_errors ?? null,
        inlineLinkedRuns: q.metrics.inline_linked_runs ?? null,
        inlineLinkRate: q.metrics.inline_link_rate ?? null,
        targetEvidence: toJson(q.target_evidence),
      });

      for (const r of q.runs) {
        const responseId = randomUUID();
        responseRows.push({
          id: responseId,
          observationId: id,
          queryId,
          runIndex: r.run_index,
          ok: r.ok,
          error: clean(r.error),
          model: r.model ?? null,
          latencySeconds: r.latency_seconds ?? null,
          answerText: clean(r.answer_text),
          searchQueries: (r.search_queries || []).map((s) => clean(s) || ''),
          targetMentioned: r.target?.mentioned ?? null,
          targetCited: r.target?.cited ?? null,
          targetPosition: r.target?.citation_position ?? null,
          targetMentionedExclAbsence: r.mentioned_excl_absence ?? null,
          mentionSentences: toJson(r.mention_sentences),
          absenceSentences: toJson(r.absence_sentences),
          outcome: r.outcome ?? null,
          creditsUsed: r.credits_used ?? null,
          fetchMeta: toJson(r.fetch_meta),
          parseError: clean(r.parse_error ?? null),
          tablesSeen: r.tables_seen ?? null,
          raggedTable: r.ragged_table ?? null,
          snippetLinks: toJson(r.inline_links),
          targetInlineLinked: r.target_inline_linked ?? null,
          rawResponses: toJson(r.raw_responses),
        });
        for (const c of r.citations) {
          citationRows.push({
            id: randomUUID(),
            responseId,
            position: c.order,
            rawUri: c.raw_uri,
            url: c.url,
            domain: c.domain || '',
            title: clean(c.title),
            citedInAnswer: c.cited_in_answer,
            isWebSource: c.is_web_source,
          });
        }
      }

      for (const s of q.sources) {
        sourceRows.push({
          id: randomUUID(),
          observationId: id,
          queryId,
          sourceKey: s.key,
          url: s.url,
          domain: s.domain,
          title: clean(s.title),
          category: s.category,
          categorySource: s.category_source,
          llmCategory: s.llm_category,
          heuristicCategory: s.heuristic_category,
          competitorGroup: s.competitor_group,
          stability: s.stability,
          appearances: s.appearances,
          overSourceCap: s.over_source_cap,
          crawlError: clean(s.crawl_error),
          evidence: clean(s.evidence),
          evidenceTokenBudget: s.evidence_token_budget,
          evidenceTokenEstimate: s.evidence_token_estimate,
          hasRelevantEvidence: s.has_relevant_evidence,
          evidenceError: clean(s.evidence_error),
          evidenceLatencySeconds: s.evidence_latency_seconds,
        });
      }
    }

    // Replace any earlier result for this run, then store the new one atomically
    await prisma.$transaction([
      prisma.observationCitation.deleteMany({ where: { response: { observationId: id } } }),
      prisma.observationResponse.deleteMany({ where: { observationId: id } }),
      prisma.observationSource.deleteMany({ where: { observationId: id } }),
      prisma.observationQuery.deleteMany({ where: { observationId: id } }),
      prisma.observationRun.update({
        where: { id },
        data: {
          status: result.status === 'completed' ? 'completed' : 'failed',
          error: clean(result.error),
          fetcher: result.fetcher ?? null,
          dataDisclaimer: result.data_disclaimer ?? null,
          config: toJson(result.config),
          context: toJson(result.context),
          runsAttempted: m.runs_attempted ?? null,
          runsSucceeded: m.runs_succeeded ?? null,
          uniqueSources: m.unique_sources ?? null,
          uniqueDomains: m.unique_domains ?? null,
          mentionRate: m.mention_rate ?? null,
          mentionRateExclAbsence: m.mention_rate_excl_absence ?? null,
          citationRate: m.citation_rate ?? null,
          meanCitationPosition: m.mean_citation_position ?? null,
          medianCitationPosition: m.median_citation_position ?? null,
          meanSourceSetDiversity: m.mean_source_set_diversity ?? null,
          meanSourceSetStability: m.mean_source_set_stability ?? null,
          meanDomainSetStability: m.mean_domain_set_stability ?? null,
          fetcherSettings: toJson(result.fetcher_settings),
          preflight: toJson(result.preflight),
          creditsUsed: m.credits_used ?? null,
          ratesBasedOnRuns: m.rates_based_on_runs ?? null,
          aioRuns: m.aio_runs ?? null,
          aioActivationRate: m.aio_activation_rate ?? null,
          overallCitationRate: m.overall_citation_rate ?? null,
          parseErrors: m.parse_errors ?? null,
          inlineLinkedRuns: m.inline_linked_runs ?? null,
          inlineLinkRate: m.inline_link_rate ?? null,
          categoryBreakdown: toJson(m.category_breakdown),
          competitorGroupBreakdown: toJson(m.competitor_group_breakdown),
          sourceCategories: toJson(result.source_categories),
        },
      }),
      prisma.observationQuery.createMany({ data: queryRows }),
      prisma.observationResponse.createMany({ data: responseRows }),
      prisma.observationCitation.createMany({ data: citationRows }),
      prisma.observationSource.createMany({ data: sourceRows }),
    ]);

    app.log.info(
      { id, queries: queryRows.length, responses: responseRows.length, citations: citationRows.length, sources: sourceRows.length },
      'Stored Stage 1 result',
    );
    return reply.status(200).send({ message: 'Result stored' });
  });

  // ==========================================
  // GET /api/observation/:id - Observation with per-query pools and metrics
  // ?raw=true adds answer texts and every run's raw citation list
  // ==========================================
  app.get('/api/observation/:id', async (request, reply) => {
    const { id } = request.params as { id: string };
    const { raw } = request.query as { raw?: string };
    const includeRaw = raw === 'true';

    const observation = await prisma.observationRun.findUnique({
      where: { id },
      include: {
        events: { orderBy: { timestamp: 'desc' }, take: 20 },
        queries: {
          orderBy: { queryIndex: 'asc' },
          include: {
            sources: { orderBy: [{ stability: 'desc' }, { domain: 'asc' }] },
            responses: {
              orderBy: { runIndex: 'asc' },
              ...(includeRaw
                ? { include: { citations: { orderBy: { position: 'asc' } } } }
                : { select: { id: true, runIndex: true, ok: true, error: true, model: true, latencySeconds: true, targetMentioned: true, targetCited: true, targetPosition: true, targetMentionedExclAbsence: true, mentionSentences: true, absenceSentences: true, outcome: true, creditsUsed: true, parseError: true, tablesSeen: true, raggedTable: true, targetInlineLinked: true, snippetLinks: true } }),
            },
          },
        },
      },
    });
    if (!observation) {
      return reply.status(404).send({ error: 'Observation not found' });
    }
    return reply.status(200).send(observation);
  });

  // ==========================================
  // GET /api/observation?url=&limit= - Recent observations (e.g. to seed Stage 2)
  // ==========================================
  app.get('/api/observation', async (request, reply) => {
    const { url, limit } = request.query as { url?: string; limit?: string };
    const take = Math.min(Math.max(parseInt(limit || '20', 10) || 20, 1), 100);
    const observations = await prisma.observationRun.findMany({
      where: url ? { sandboxUrl: url } : undefined,
      orderBy: { createdAt: 'desc' },
      take,
      select: {
        id: true,
        sandboxUrl: true,
        status: true,
        fetcher: true,
        dataDisclaimer: true,
        runsAttempted: true,
        runsSucceeded: true,
        uniqueSources: true,
        mentionRate: true,
        citationRate: true,
        aioActivationRate: true,
        parseErrors: true,
        creditsUsed: true,
        createdAt: true,
      },
    });
    return reply.status(200).send({ observations });
  });
}
