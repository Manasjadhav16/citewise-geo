import { FormEvent, useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import React from 'react';
import { apiFetch } from '../lib/api';

const Home = () => {
  const [url, setUrl] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  // Two-stage GEO: Stage 1 real-world observation
  const [obsUrl, setObsUrl] = useState('');
  const [runsPerQuery, setRunsPerQuery] = useState('');
  const [queryCount, setQueryCount] = useState('');
  const [isObserving, setIsObserving] = useState(false);
  const [obsError, setObsError] = useState<string | null>(null);

  const handleObserve = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setObsError(null);
    setIsObserving(true);
    try {
      const { observation_id } = await apiFetch<{ observation_id: string }>('/api/observation/run', {
        method: 'POST',
        body: JSON.stringify({
          url: obsUrl.trim(),
          // Blank fields fall back to the server's STAGE1_* defaults
          runs_per_query: runsPerQuery ? Number(runsPerQuery) : undefined,
          query_count: queryCount ? Number(queryCount) : undefined,
        }),
      });
      router.push(`/observation/${observation_id}`);
    } catch (err) {
      setObsError(err instanceof Error ? err.message : 'Unexpected error occurred.');
      setIsObserving(false);
    }
  };

  const handleAnalyze = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    if (!url.trim()) {
      setError('Please enter a URL to analyze.');
      return;
    }

    setIsAnalyzing(true);

    try {
      // Start job
      const startResponse = await fetch('http://localhost:4000/api/sandbox/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url.trim() }),
      });

      const startData = await startResponse.json();

      if (!startResponse.ok || !startData.run_id) {
        setError(startData.error || 'Failed to start analysis job.');
        setIsAnalyzing(false);
        return;
      }

      // Redirect to run details page
      router.push(`/${startData.run_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error occurred.');
      setIsAnalyzing(false);
    }
  };

  return (
    <>
      <Head>
        <title>CiteLabs - AEO/GEO Evaluation Platform</title>
        <meta name="description" content="Analyze your page's performance in AI-powered search engines" />
      </Head>
      <main className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
        {/* Animated Background */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
        </div>

        <div className="relative z-10 flex min-h-screen flex-col items-center justify-center px-4 py-16">
          <div className="w-full max-w-4xl space-y-12">
            {/* Header */}
            <header className="text-center space-y-6">
              <h1 className="text-6xl md:text-7xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent animate-fade-in">
                CiteLabs
              </h1>
              <p className="text-2xl md:text-3xl font-light text-gray-300 animate-fade-in-delay">
                AEO/GEO Evaluation Platform
              </p>
              <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto animate-fade-in-delay-2">
                Analyze your page's performance in AI-powered search engines. 
                Get comprehensive insights into Answer Engine Optimization (AEO) 
                and Generative Engine Optimization (GEO) scores.
              </p>
            </header>

            {/* Input Form */}
            <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl border border-gray-700 p-8 md:p-12 shadow-2xl animate-fade-in-delay-3">
              <h2 className="text-2xl md:text-3xl font-semibold mb-6 text-center">Start Analysis</h2>
              <form className="space-y-6" onSubmit={handleAnalyze}>
                <label className="flex flex-col gap-3">
                  <span className="text-sm font-medium text-gray-300 uppercase tracking-wide">Sandbox Page URL</span>
                  <input
                    className="w-full rounded-lg border border-gray-600 bg-gray-900/50 px-4 py-4 text-lg text-white placeholder-gray-500 focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition-all"
                    placeholder="https://example.com"
                    value={url}
                    onChange={(event) => setUrl(event.target.value)}
                    type="url"
                    required
                    disabled={isAnalyzing}
                  />
                </label>
                <div className="flex flex-col items-center gap-4">
                  <button
                    className="w-full md:w-auto px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold rounded-lg shadow-lg transform hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                    type="submit"
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <span className="flex items-center gap-2">
                        <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Starting Analysis...
                      </span>
                    ) : (
                      'Start Analysis'
                    )}
                  </button>
                  {error && (
                    <div className="w-full bg-red-900/50 border border-red-700 text-red-200 px-4 py-3 rounded-lg">
                      {error}
                    </div>
                  )}
                </div>
              </form>
            </section>

            {/* Two-Stage GEO: Stage 1 */}
            <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl border border-gray-700 p-8 md:p-12 shadow-2xl">
              <h2 className="text-2xl md:text-3xl font-semibold mb-2 text-center">Two-Stage GEO: Real-World Observation</h2>
              <p className="text-gray-400 text-sm text-center mb-6 max-w-2xl mx-auto">
                Stage 1 asks representative queries repeatedly through Gemini with Google Search grounding and records
                which sources it cites. That source pool then seeds the controlled GEO score. This is an approximate
                proxy for Google AI Overview citations, not real AI Overview data.
              </p>
              <form className="space-y-4" onSubmit={handleObserve}>
                <input
                  className="w-full rounded-lg border border-gray-600 bg-gray-900/50 px-4 py-3 text-white placeholder-gray-500 focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                  placeholder="https://example.com/page"
                  value={obsUrl}
                  onChange={(event) => setObsUrl(event.target.value)}
                  type="url"
                  required
                  disabled={isObserving}
                />
                <div className="grid grid-cols-2 gap-4">
                  <label className="flex flex-col gap-2 text-sm text-gray-300">
                    Runs per query
                    <input
                      className="rounded-lg border border-gray-600 bg-gray-900/50 px-3 py-2 text-white placeholder-gray-500"
                      type="number" min={1} placeholder="Server default (15)"
                      value={runsPerQuery} onChange={(e) => setRunsPerQuery(e.target.value)} disabled={isObserving}
                    />
                  </label>
                  <label className="flex flex-col gap-2 text-sm text-gray-300">
                    Queries
                    <input
                      className="rounded-lg border border-gray-600 bg-gray-900/50 px-3 py-2 text-white placeholder-gray-500"
                      type="number" min={1} placeholder="Server default (10)"
                      value={queryCount} onChange={(e) => setQueryCount(e.target.value)} disabled={isObserving}
                    />
                  </label>
                </div>
                <p className="text-xs text-gray-500">Each run is one rate-limited grounded model call; the defaults (10 × 15) take a long time on a free-tier key.</p>
                <div className="flex justify-center">
                  <button
                    className="px-8 py-3 bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-700 hover:to-purple-700 rounded-lg font-semibold disabled:opacity-50"
                    type="submit"
                    disabled={isObserving}
                  >
                    {isObserving ? 'Starting…' : 'Start Real-World Observation'}
                  </button>
                </div>
                {obsError && <div className="bg-red-900/50 border border-red-700 text-red-200 px-4 py-3 rounded-lg">{obsError}</div>}
              </form>
            </section>

            {/* Features */}
            <section className="grid md:grid-cols-3 gap-6 mt-16">
              {[
                {
                  title: '7-Step Analysis',
                  description: 'Comprehensive evaluation pipeline covering crawl, intent, questions, competitors, RAG, and scoring.',
                  icon: '📊',
                },
                {
                  title: 'Real-Time Progress',
                  description: 'Track your analysis in real-time with detailed step-by-step progress updates.',
                  icon: '⚡',
                },
                {
                  title: 'Actionable Insights',
                  description: 'Get detailed recommendations to improve your AEO and GEO performance scores.',
                  icon: '🎯',
                },
              ].map((feature, index) => (
                <div
                  key={index}
                  className="bg-gray-800/30 backdrop-blur-sm rounded-xl border border-gray-700 p-6 hover:border-purple-500/50 transition-all duration-300"
                >
                  <div className="text-4xl mb-4">{feature.icon}</div>
                  <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                  <p className="text-gray-400 text-sm">{feature.description}</p>
                </div>
              ))}
            </section>
          </div>
        </div>
      </main>
    </>
  );
};

export default Home;
