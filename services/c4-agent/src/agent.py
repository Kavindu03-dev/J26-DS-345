"""LLM orchestrator for the tourism agent.

Receives a natural-language tourist request, decides which tools to call
(C1 aspect scores, C2 hotel ranking, C3 itinerary), and combines the results
into one answer. The tool callers live in src/tools/ and the fake endpoints
used for testing live in src/mocks/.
"""
