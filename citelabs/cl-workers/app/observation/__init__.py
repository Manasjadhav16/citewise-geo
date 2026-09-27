"""
Stage 1: Real-World Observation Layer.

Repeatedly asks a search-grounded model the same representative queries and
records which web sources it cites, to approximate which sources real
AI-generated search answers draw on for those queries.

LIMITATION: the default fetcher (Gemini with Google Search grounding) is an
approximate proxy for Google AI Overview citation behaviour. It uses the same
underlying Google Search index that AI Overviews draw from, but it is Gemini's
own search-and-cite process, not AI Overview output. There is no official public
AI Overview API. Nothing produced here should be presented as real AI Overview data.
"""
