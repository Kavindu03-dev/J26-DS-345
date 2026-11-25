"""Random identifiers. They carry no user information."""

import uuid


def new_request_id() -> str:
    return f"req-{uuid.uuid4().hex}"


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex}"
