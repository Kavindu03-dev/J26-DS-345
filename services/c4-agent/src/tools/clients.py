"""HTTP clients the agent uses to call Components 1, 2 and 3.

Each method returns the component's JSON response, or raises ToolError with
one of a small set of codes. The agent turns a ToolError into a message the
language model can react to (for example, ask the user for another hotel name
after NOT_FOUND), instead of crashing.

The same clients work against the mocks (ports 900x) and the real services
(ports 800x); only the URLs in config / .env change.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import httpx

from .. import config


class ErrorCode(str, Enum):
    NOT_FOUND = "NOT_FOUND"  # unknown hotel / destination
    INVALID_ARGUMENT = "INVALID_ARGUMENT"  # request rejected by the component
    UNAVAILABLE = "UNAVAILABLE"  # component down or returned a server error
    TIMEOUT = "TIMEOUT"  # component too slow


class ToolError(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int | None = None):
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message
        self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code.value, "message": self.message}}


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    detail = body.get("detail", body) if isinstance(body, dict) else body
    return str(detail)[:300]


def _call(client: httpx.Client, method: str, path: str, component: str, **kwargs) -> dict:
    try:
        response = client.request(method, path, **kwargs)
    except httpx.TimeoutException:
        raise ToolError(ErrorCode.TIMEOUT, f"{component} did not answer in time.")
    except httpx.TransportError as e:
        raise ToolError(ErrorCode.UNAVAILABLE, f"{component} is not reachable ({e.__class__.__name__}).")

    if response.status_code == 404:
        raise ToolError(ErrorCode.NOT_FOUND, _detail(response), 404)
    if response.status_code in (400, 422):
        raise ToolError(ErrorCode.INVALID_ARGUMENT, _detail(response), response.status_code)
    if response.status_code >= 400:
        raise ToolError(ErrorCode.UNAVAILABLE, f"{component} error: {_detail(response)}", response.status_code)
    return response.json()


class ToolClients:
    """Calls C1, C2 and C3. Pass ready-made httpx.Client objects in tests."""

    def __init__(self, c1: httpx.Client, c2: httpx.Client, c3: httpx.Client):
        self.c1, self.c2, self.c3 = c1, c2, c3

    @classmethod
    def from_config(cls) -> "ToolClients":
        return cls(
            c1=httpx.Client(base_url=config.C1_URL, timeout=config.C1_TIMEOUT_S),
            c2=httpx.Client(base_url=config.C2_URL, timeout=config.C2_TIMEOUT_S),
            c3=httpx.Client(base_url=config.C3_URL, timeout=config.C3_TIMEOUT_S),
        )

    def close(self) -> None:
        for c in (self.c1, self.c2, self.c3):
            c.close()

    # C1 - contracts/c1_aspect_scores.json
    def get_aspect_scores(self, hotel_id: str) -> dict:
        return _call(self.c1, "GET", f"/hotels/{hotel_id}/aspect-scores", "C1 review engine")

    # C2 - contracts/c2_hotel_ranking.json
    def recommend_hotels(self, request: dict) -> dict:
        return _call(self.c2, "POST", "/recommendations", "C2 hotel recommender", json=request)

    # C3 - real API POST /plan-itinerary (external C2 payload)
    def plan_itinerary(self, request: dict) -> dict:
        return _call(self.c3, "POST", "/plan-itinerary", "C3 itinerary planner", json=request)
