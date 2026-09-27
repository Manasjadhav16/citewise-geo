import { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
} from 'recharts';

type SandboxPage = {
  url: string;
  title?: string | null;
  h1?: string | null;
  intent?: string | null;
  pageSummary?: string | null;
  aiSummary?: string | null;
};

type SandboxQuestion = {
  id: string;
  type: string;
  question: string;
};

type SandboxCompetitor = {
  id: string;
  url: string;
  domain: string;
  title?: string | null;
  aiSummary?: string | null;
  sourceType?: string;
};

type SandboxRagResult = {
  id: string;
  question: string;
  answer: string;
  didSandboxAppear: boolean;
  chunksUsed: number;
  competitorCitations: string[];
  sandboxCitations: string[];
  diagnosisType?: 'competitor_advantage' | 'content_gap' | null;
  diagnosisDetail?: string | null;
  questionRef?: {
    type: string;
  };
};

type SandboxScores = {
  geoScore: number;
  aeoScore: number;
  citationRate: number;
  coverage?: number | null;
  competitorDominance?: number | null;
  recommendations: string[];
};


type AdvancedMetrics = {
  ai_share_of_voice: number;
  mention_rate: number;
  first_citation_rate: number;
  citation_rate: number;
};

type SandboxResponse = {
  run_id: string;
  status: string;
  sandbox: SandboxPage | null;
  generated_questions: SandboxQuestion[];
  competitors: SandboxCompetitor[];
  rag_results: SandboxRagResult[];
  scores: SandboxScores | null;
  advanced_metrics?: AdvancedMetrics | null;
  recommendations: string[];
};

const COLORS = ['#8b5cf6', '#3b82f6', '#ec4899', '#10b981', '#f59e0b'];

const RunDetails = () => {
  const router = useRouter();
  const { runId } = router.query;
  const [data, setData] = useState<SandboxResponse | null>(null);
  const [currentStep, setCurrentStep] = useState<string>('processing');

  // Redirect if runId is a reserved route like "chat"
  useEffect(() => {
    if (runId === 'chat') {
      router.replace('/ask');
    }
  }, [runId, router]);

  // Don't render if this is a reserved route
  if (runId === 'chat') {
    return null;
  }

  // Fetch job status repeatedly until completed or error
  useEffect(() => {
    if (!runId || typeof runId !== 'string' || runId === 'chat') {
      return;
    }

    let isSubscribed = true;
    let timerId: NodeJS.Timeout | null = null;

    const fetchJobStatus = async () => {
      try {
        const response = await fetch(`http://localhost:4000/api/sandbox/${runId}`);
        const responseData = await response.json();

        if (!isSubscribed) return;

        if (!response.ok) {
          console.error('Failed to fetch job status:', responseData);
          return;
        }

        // Update current step based on status
        const statusToStep: Record<string, string> = {
          processing: 'processing',
          crawling_sandbox_site: 'crawling_sandbox_site',
          intent_extraction: 'intent_extraction',
          questions_generation: 'questions_generation',
          competitors_extraction: 'competitors_extraction',
          crawling_competitors: 'crawling_competitors',
          rag_simulation: 'rag_simulation',
          score_calculation: 'score_calculation',
          completed: 'completed',
          error: 'error',
        };

        const step = statusToStep[responseData.status] || 'processing';
        setCurrentStep(step);

        // If completed, set data and stop polling
        if (responseData.status === 'completed') {
          setData(responseData as SandboxResponse);
          setCurrentStep('completed');
          if (timerId) clearInterval(timerId);
        } else if (responseData.status === 'error') {
          setCurrentStep('error');
          if (timerId) clearInterval(timerId);
        }
      } catch (err) {
        console.error('Error fetching job status:', err);
      }
    };

    fetchJobStatus();
    timerId = setInterval(fetchJobStatus, 2000);

    return () => {
      isSubscribed = false;
      if (timerId) clearInterval(timerId);
    };
  }, [runId]);

  const getStepInfo = (step: string) => {
    const steps = [
      { key: 'processing', label: 'Initializing', icon: '🚀', color: 'from-blue-500 to-cyan-500' },
      { key: 'crawling_sandbox_site', label: 'Crawling Sandbox Page', icon: '🕷️', color: 'from-purple-500 to-pink-500' },
      { key: 'intent_extraction', label: 'Extracting Intent', icon: '🧠', color: 'from-indigo-500 to-purple-500' },
      { key: 'questions_generation', label: 'Generating Questions', icon: '❓', color: 'from-blue-500 to-indigo-500' },
      { key: 'competitors_extraction', label: 'Extracting Competitors', icon: '🔍', color: 'from-pink-500 to-rose-500' },
      { key: 'crawling_competitors', label: 'Crawling Competitors', icon: '🕸️', color: 'from-orange-500 to-red-500' },
      { key: 'rag_simulation', label: 'RAG Simulation', icon: '🤖', color: 'from-green-500 to-emerald-500' },
      { key: 'score_calculation', label: 'Calculating Scores', icon: '📊', color: 'from-yellow-500 to-orange-500' },
    ];

    return steps.find((s) => s.key === step) || steps[0];
  };

  const getStepStatus = (stepKey: string) => {
    const stepIndex = [
      'processing',
      'crawling_sandbox_site',
      'intent_extraction',
      'questions_generation',
      'competitors_extraction',
      'crawling_competitors',
      'rag_simulation',
      'score_calculation',
    ].indexOf(stepKey);

    const currentIndex = [
      'processing',
      'crawling_sandbox_site',
      'intent_extraction',
      'questions_generation',
      'competitors_extraction',
      'crawling_competitors',
      'rag_simulation',
      'score_calculation',
    ].indexOf(currentStep);

    if (currentStep === 'error') return 'error';
    if (currentStep === 'completed') return stepIndex <= currentIndex ? 'completed' : 'pending';
    if (stepIndex < currentIndex) return 'completed';
    if (stepIndex === currentIndex) return 'active';
    return 'pending';
  };

  // Prepare chart data
  const questionTypeData = data?.generated_questions
    ? Object.entries(
      data.generated_questions.reduce((acc, q) => {
        acc[q.type] = (acc[q.type] || 0) + 1;
        return acc;
      }, {} as Record<string, number>)
    ).map(([type, count]) => ({ type: type.charAt(0).toUpperCase() + type.slice(1), count }))
    : [];

  const ragResultsData = data?.rag_results
    ? data.rag_results.map((r, i) => ({
      question: `Q${i + 1}`,
      chunks: r.chunksUsed,
      cited: r.didSandboxAppear ? 1 : 0,
    }))
    : [];

  const citationPieData = data?.scores
    ? [
      { name: 'Cited', value: data.scores.citationRate },
      { name: 'Not Cited', value: 100 - data.scores.citationRate },
    ]
    : [];

  const currentStepInfo = getStepInfo(currentStep);

  return (
    <>
      <Head>
        <title>Run Details - CiteLabs</title>
      </Head>
      <main className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
        <div className="container mx-auto px-4 py-8 max-w-7xl">
          {/* Header */}
          <header className="mb-8">
            <button
              onClick={() => router.push('/')}
              className="mb-4 text-gray-400 hover:text-white transition-colors flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              Back to Home
            </button>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              Analysis Run
            </h1>
            {runId && (
              <p className="mt-2 text-gray-400 font-mono text-sm">Run ID: {runId}</p>
            )}
          </header>

          {/* Progress Indicator */}
          {currentStep !== 'completed' && currentStep !== 'error' && (
            <section className="mb-8 bg-gray-800/50 backdrop-blur-lg rounded-2xl border border-gray-700 p-8 shadow-2xl">
              <div className="flex items-center justify-center mb-8">
                <div className={`relative w-32 h-32 rounded-full bg-gradient-to-r ${currentStepInfo.color} p-1 animate-spin-slow`}>
                  <div className="w-full h-full rounded-full bg-gray-900 flex items-center justify-center">
                    <span className="text-5xl">{currentStepInfo.icon}</span>
                  </div>
                </div>
              </div>
              <h2 className="text-2xl font-semibold text-center mb-2">{currentStepInfo.label}</h2>
              <p className="text-gray-400 text-center mb-8">Analysis in progress...</p>

              <div className="space-y-4">
                {[
                  { key: 'processing', label: 'Initializing', icon: '🚀' },
                  { key: 'crawling_sandbox_site', label: 'Crawling Sandbox Page', icon: '🕷️' },
                  { key: 'intent_extraction', label: 'Extracting Intent', icon: '🧠' },
                  { key: 'questions_generation', label: 'Generating Questions', icon: '❓' },
                  { key: 'competitors_extraction', label: 'Extracting Competitors', icon: '🔍' },
                  { key: 'crawling_competitors', label: 'Crawling Competitors', icon: '🕸️' },
                  { key: 'rag_simulation', label: 'RAG Simulation', icon: '🤖' },
                  { key: 'score_calculation', label: 'Calculating Scores', icon: '📊' },
                ].map((step) => {
                  const status = getStepStatus(step.key);
                  return (
                    <div key={step.key} className="flex items-center gap-4">
                      <div
                        className={`w-12 h-12 rounded-full flex items-center justify-center text-2xl transition-all ${status === 'completed'
                          ? 'bg-green-500'
                          : status === 'active'
                            ? 'bg-gradient-to-r from-purple-500 to-pink-500 animate-pulse'
                            : status === 'error'
                              ? 'bg-red-500'
                              : 'bg-gray-700'
                          }`}
                      >
                        {status === 'completed' ? '✓' : step.icon}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className={`font-medium ${status === 'active' ? 'text-purple-400' : status === 'completed' ? 'text-green-400' : 'text-gray-400'}`}>
                            {step.label}
                          </span>
                          {status === 'active' && (
                            <span className="text-xs text-purple-400 animate-pulse">Processing...</span>
                          )}
                        </div>
                        {status === 'active' && (
                          <div className="h-1 bg-gray-700 rounded-full overflow-hidden">
                            <div className={`h-full bg-gradient-to-r ${currentStepInfo.color} animate-progress`}></div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Results */}
          {data && data.status === 'completed' && (
            <div className="space-y-8">


              {/* Advanced Metrics */}
              {data.advanced_metrics && (
                <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl border border-gray-700 p-8 shadow-2xl">
                  <h2 className="text-3xl font-bold mb-6">Metrics</h2>
                  <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <div className="group relative bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl p-6 shadow-lg transition-transform hover:scale-[1.02]">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-semibold uppercase text-violet-100">AI Share of Voice</h3>
                        <div className="relative group/tooltip">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-violet-200 hover:text-white cursor-help" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <div className="absolute bottom-full right-0 mb-2 w-48 p-2 bg-gray-900 text-xs text-white rounded shadow-xl opacity-0 group-hover/tooltip:opacity-100 transition-opacity pointer-events-none z-10 border border-gray-700">
                            Shows how much of the conversation your brand occupies compared to competitors in a single answer.
                          </div>
                        </div>
                      </div>
                      <p className="text-4xl font-bold text-white">{data.advanced_metrics.ai_share_of_voice}%</p>
                    </div>

                    <div className="group relative bg-gradient-to-br from-fuchsia-500 to-pink-600 rounded-xl p-6 shadow-lg transition-transform hover:scale-[1.02]">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-semibold uppercase text-fuchsia-100">Mention Rate</h3>
                        <div className="relative group/tooltip">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-fuchsia-200 hover:text-white cursor-help" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <div className="absolute bottom-full right-0 mb-2 w-48 p-2 bg-gray-900 text-xs text-white rounded shadow-xl opacity-0 group-hover/tooltip:opacity-100 transition-opacity pointer-events-none z-10 border border-gray-700">
                            The percentage of all queries where the AI mentions your brand name in its text.
                          </div>
                        </div>
                      </div>
                      <p className="text-4xl font-bold text-white">{data.advanced_metrics.mention_rate}%</p>
                    </div>

                    <div className="group relative bg-gradient-to-br from-indigo-500 to-blue-600 rounded-xl p-6 shadow-lg transition-transform hover:scale-[1.02]">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-semibold uppercase text-indigo-100">First Citation Rate</h3>
                        <div className="relative group/tooltip">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-indigo-200 hover:text-white cursor-help" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <div className="absolute bottom-full right-0 mb-2 w-48 p-2 bg-gray-900 text-xs text-white rounded shadow-xl opacity-0 group-hover/tooltip:opacity-100 transition-opacity pointer-events-none z-10 border border-gray-700">
                            How often your brand is the number one recommendation when the AI cites sources.
                          </div>
                        </div>
                      </div>
                      <p className="text-4xl font-bold text-white">{data.advanced_metrics.first_citation_rate}%</p>
                    </div>

                    <div className="group relative bg-gradient-to-br from-cyan-500 to-teal-600 rounded-xl p-6 shadow-lg transition-transform hover:scale-[1.02]">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-semibold uppercase text-cyan-100">Citation Rate</h3>
                        <div className="relative group/tooltip">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-cyan-200 hover:text-white cursor-help" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <div className="absolute bottom-full right-0 mb-2 w-48 p-2 bg-gray-900 text-xs text-white rounded shadow-xl opacity-0 group-hover/tooltip:opacity-100 transition-opacity pointer-events-none z-10 border border-gray-700">
                            The percentage of answers that include a clickable link to your website (driving traffic).
                          </div>
                        </div>
                      </div>
                      <p className="text-4xl font-bold text-white">{data.advanced_metrics.citation_rate}%</p>
                    </div>
                  </div>

                  {/* Question Types Pie Chart */}
                  <div className="bg-gray-900/50 rounded-xl p-6">
                    <h3 className="text-lg font-semibold mb-4 text-center">Question Types Distribution</h3>
                    <div className="w-full h-[300px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={questionTypeData}
                            cx="50%"
                            cy="50%"
                            labelLine={true}
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            outerRadius={100}
                            fill="#8884d8"
                            dataKey="count"
                            nameKey="type"
                          >
                            {questionTypeData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }} />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </section>
              )}

              {/* Sandbox Page */}
              {data.sandbox && (
                <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl border border-gray-700 p-8 shadow-2xl">
                  <h2 className="text-3xl font-bold mb-6">Sandbox Page Analysis</h2>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-gray-900/50 rounded-xl p-6">
                      <h3 className="text-sm font-semibold uppercase text-gray-400 mb-2">Title</h3>
                      <p className="text-lg text-white">{data.sandbox.title || '—'}</p>
                    </div>
                    <div className="bg-gray-900/50 rounded-xl p-6">
                      <h3 className="text-sm font-semibold uppercase text-gray-400 mb-2">H1</h3>
                      <p className="text-lg text-white">{data.sandbox.h1 || '—'}</p>
                    </div>
                    <div className="bg-gray-900/50 rounded-xl p-6 md:col-span-2">
                      <h3 className="text-sm font-semibold uppercase text-gray-400 mb-2">Page Intent</h3>
                      <p className="text-white whitespace-pre-line">{data.sandbox.intent || '—'}</p>
                    </div>
                    <div className="bg-gray-900/50 rounded-xl p-6 md:col-span-2">
                      <h3 className="text-sm font-semibold uppercase text-gray-400 mb-2">AI Summary</h3>
                      <p className="text-white whitespace-pre-line">{data.sandbox.aiSummary || '—'}</p>
                    </div>
                  </div>
                </section>
              )}

              {/* Questions */}
              {data.generated_questions && data.generated_questions.length > 0 && (
                <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl border border-gray-700 p-8 shadow-2xl">
                  <h2 className="text-3xl font-bold mb-6">Generated Questions ({data.generated_questions.length})</h2>
                  <div className="grid md:grid-cols-3 gap-6">
                    {['intent', 'experience', 'transaction', 'SERP'].map((type) => {
                      const typeQuestions = data.generated_questions.filter((q) => q.type === type);
                      if (typeQuestions.length === 0) return null;
                      return (
                        <div key={type} className="bg-gray-900/50 rounded-xl p-6">
                          <h3 className="text-lg font-semibold mb-4 capitalize">
                            {type} ({typeQuestions.length})
                          </h3>
                          <ul className="space-y-3">
                            {typeQuestions.map((q) => (
                              <li key={q.id} className="text-sm text-gray-300 border-l-2 border-purple-500 pl-3">
                                {q.question}
                              </li>
                            ))}
                          </ul>
                        </div>
                      );
                    })}
                  </div>
                </section>
              )}

              {/* Competitors */}
              {data.competitors && data.competitors.length > 0 && (
                <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl border border-gray-700 p-8 shadow-2xl">
                  <h2 className="text-3xl font-bold mb-6">Competitors ({data.competitors.length})</h2>
                  <div className="grid md:grid-cols-2 gap-4">
                    {data.competitors.map((comp) => (
                      <div key={comp.id} className="bg-gray-900/50 rounded-xl p-6 border border-gray-700">
                        <div className="flex items-start justify-between mb-2">
                          <h3 className="font-medium text-white">
                            <a
                              href={comp.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-purple-400 hover:text-purple-300 transition-colors"
                            >
                              {comp.url}
                            </a>
                          </h3>
                          {comp.sourceType && (
                            <span className={`text-xs px-2 py-1 rounded ${comp.sourceType === 'serp'
                              ? 'bg-blue-900/50 text-blue-300'
                              : comp.sourceType === 'llm'
                                ? 'bg-purple-900/50 text-purple-300'
                                : 'bg-gray-700 text-gray-300'
                              }`}>
                              {comp.sourceType.toUpperCase()}
                            </span>
                          )}
                        </div>
                        {comp.title && <p className="text-sm text-gray-400 mb-2">{comp.title}</p>}
                        {comp.aiSummary && <p className="text-sm text-gray-300">{comp.aiSummary}</p>}
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* RAG Results */}
              {data.rag_results && data.rag_results.length > 0 && (
                <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl border border-gray-700 p-8 shadow-2xl">
                  <h2 className="text-3xl font-bold mb-6">RAG Simulation Results</h2>
                  <div className="space-y-4">
                    {data.rag_results.map((rag) => (
                      <div
                        key={rag.id}
                        className={`rounded-xl p-6 border ${rag.didSandboxAppear
                          ? 'bg-green-900/20 border-green-700'
                          : 'bg-gray-900/50 border-gray-700'
                          }`}
                      >
                        <div className="flex items-start justify-between mb-3">
                          <h3 className="font-medium text-white flex-1">{rag.question}</h3>
                          <div className="flex gap-2 ml-4">
                            <span
                              className={`text-xs px-3 py-1 rounded-full ${rag.didSandboxAppear
                                ? 'bg-green-500/20 text-green-300'
                                : 'bg-gray-700 text-gray-400'
                                }`}
                            >
                              {rag.didSandboxAppear ? '✓ Cited' : '✗ Not Cited'}
                            </span>
                            {rag.diagnosisType && (
                              <span
                                className={`text-xs px-3 py-1 rounded-full ${rag.diagnosisType === 'competitor_advantage'
                                  ? 'bg-orange-500/20 text-orange-300'
                                  : 'bg-purple-500/20 text-purple-300'
                                  }`}
                              >
                                {rag.diagnosisType === 'competitor_advantage' ? 'Competitor advantage' : 'Content gap'}
                              </span>
                            )}
                            <span className="text-xs px-3 py-1 rounded-full bg-blue-500/20 text-blue-300">
                              {rag.chunksUsed} chunks
                            </span>
                          </div>
                        </div>
                        <p className="text-gray-300 mb-3">{rag.answer}</p>
                        {rag.diagnosisDetail && (
                          <p className="text-sm text-gray-400 mb-3">
                            <strong>Why:</strong> {rag.diagnosisDetail}
                          </p>
                        )}
                        {(rag.sandboxCitations.length > 0 || rag.competitorCitations.length > 0) && (
                          <div className="text-xs text-gray-400 space-y-1">
                            <div>
                              <strong>Sandbox:</strong> {rag.sandboxCitations.length > 0 ? rag.sandboxCitations.join(', ') : 'None'}
                            </div>
                            <div>
                              <strong>Competitors:</strong>{' '}
                              {rag.competitorCitations.length > 0
                                ? rag.competitorCitations.slice(0, 3).join(', ')
                                : 'None'}
                              {rag.competitorCitations.length > 3 && ` (+${rag.competitorCitations.length - 3} more)`}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* Recommendations */}
              {data.scores && data.scores.recommendations && data.scores.recommendations.length > 0 && (
                <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl border border-yellow-700/50 p-8 shadow-2xl">
                  <h2 className="text-3xl font-bold mb-6">Recommendations</h2>
                  <ul className="space-y-3">
                    {data.scores.recommendations.map((rec, index) => (
                      <li key={index} className="flex items-start gap-3">
                        <span className="text-yellow-400 mt-1">•</span>
                        <span className="text-gray-300">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
            </div>
          )}

          {/* Error State */}
          {currentStep === 'error' && (
            <section className="bg-red-900/20 backdrop-blur-lg rounded-2xl border border-red-700 p-8 shadow-2xl">
              <h2 className="text-2xl font-bold mb-4 text-red-400">Analysis Failed</h2>
              <p className="text-gray-300">The analysis encountered an error. Please try again.</p>
            </section>
          )}
        </div>
      </main>
    </>
  );
};

export default RunDetails;

