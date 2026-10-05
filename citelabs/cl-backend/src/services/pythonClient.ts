import { DEFAULT_MODEL_NAME, DEFAULT_WORKER_BASE_URL } from '../constants';

export type CrawlResponse = {
  title?: string | null;
  h1?: string | null;
  content_preview?: string | null;
  full_content?: string | null;
  error?: string;
};

export class Worker {
  constructor(private readonly baseUrl: string = DEFAULT_WORKER_BASE_URL) { }

  async crawl(url: string): Promise<CrawlResponse> {
    const response = await fetch(`${this.baseUrl}/crawl`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /crawl request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }

    return (await response.json()) as CrawlResponse;
  }

  async generateSchema(payload: {
    url: string;
    page_type: string;
    title: string;
    ai_summary: string;
  }): Promise<{ schema_block?: string; error?: string }> {
    const response = await fetch(`${this.baseUrl}/schema/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /schema/generate request failed with status ${response.status}: ${errorBody || 'no body'
        }`,
      );
    }

    return response.json() as Promise<{ schema_block?: string; error?: string }>;
  }

  async runSimulation(payload: { url: string; query: string }): Promise<unknown> {
    const response = await fetch(`${this.baseUrl}/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /simulate request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }

    return response.json();
  }

  async sandboxAnalyze(url: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/sandbox/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /sandbox/analyze request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }

    return response.json();
  }

  async sandboxQuestions(payload: {
    run_id: string;
    page_intent: string;
    page_content: string;
  }): Promise<any> {
    const response = await fetch(`${this.baseUrl}/sandbox/questions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /sandbox/questions request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }

    return response.json();
  }

  async sandboxCompetitors(payload: {
    run_id: string;
    questions: Array<{ type: string; q: string }>;
  }): Promise<any> {
    const response = await fetch(`${this.baseUrl}/sandbox/competitors`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /sandbox/competitors request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }

    return response.json();
  }

  async sandboxCrawlAll(payload: {
    run_id: string;
    sandbox_url: string;
    sandbox_data: any;
    competitor_urls: string[];
  }): Promise<any> {
    const response = await fetch(`${this.baseUrl}/sandbox/crawl-all`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /sandbox/crawl-all request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }

    return response.json();
  }

  async sandboxRag(payload: {
    run_id: string;
    questions: Array<{ type: string; q: string }>;
    sandbox_url: string;
  }): Promise<any> {
    const response = await fetch(`${this.baseUrl}/sandbox/rag`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /sandbox/rag request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }

    return response.json();
  }

  async sandboxFinalize(payload: {
    run_id: string;
    rag_results: any[];
    questions: Array<{ type: string; q: string }>;
  }): Promise<any> {
    const response = await fetch(`${this.baseUrl}/sandbox/finalize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /sandbox/finalize request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }

    return response.json();
  }

  async compare(payload: {
    before: Record<string, number | null>;
    after: Record<string, number | null>;
  }): Promise<{
    geo_improvement: number | null;
    aeo_improvement: number | null;
    deltas: Record<string, { before: number | null; after: number | null; delta: number | null }>;
  }> {
    const response = await fetch(`${this.baseUrl}/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /compare request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }

    return (await response.json()) as any;
  }

  async startJob(payload: { run_id: string; url: string; model_name?: string; seed?: unknown }): Promise<void> {
    const finalPayload = {
      ...payload,
      model_name: payload.model_name || DEFAULT_MODEL_NAME,
    };

    const response = await fetch(`${this.baseUrl}/start-job`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(finalPayload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /start-job request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }
  }

  async startStage1(payload: {
    observation_id: string;
    url: string;
    queries?: string[];
    runs_per_query?: number;
    query_count?: number;
    fetcher?: string;
    serpapi_gl?: string;
    serpapi_hl?: string;
    serpapi_device?: string;
  }): Promise<void> {
    const response = await fetch(`${this.baseUrl}/stage1/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /stage1/start request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }
  }

  async processChatMessage(payload: {
    session_id: string;
    content: string;
    scope: 'sandbox_only' | 'sandbox_competitors' | 'web_only';
    run_id?: string;
    sandbox_url?: string;
  }): Promise<{
    content: string;
    citations: Array<{
      url: string;
      domain_type?: string;
      chunk_id?: number;
      relevance_score?: number;
    }>;
    tokens_in: number;
    tokens_out: number;
    model_used: string;
  }> {
    const response = await fetch(`${this.baseUrl}/chat/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Worker /chat/message request failed with status ${response.status}: ${errorBody || 'no body'}`,
      );
    }

    return response.json();
  }
}

export const worker = new Worker(
  process.env.WORKER_BASE_URL ?? DEFAULT_WORKER_BASE_URL,
);

