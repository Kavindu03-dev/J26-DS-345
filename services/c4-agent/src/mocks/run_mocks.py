"""Start all three mock services with one command.

    python -m src.mocks.run_mocks

C1 mock -> http://localhost:9001/docs
C2 mock -> http://localhost:9002/docs
C3 mock -> http://localhost:9003/docs

Press Ctrl+C to stop. Mock ports (900x) never clash with the real services (800x).
"""

from __future__ import annotations

import threading
import time

import uvicorn

from ..config import MOCK_PORTS
from .c1_mock import app as c1_app
from .c2_mock import app as c2_app
from .c3_mock import app as c3_app

APPS = {"c1": c1_app, "c2": c2_app, "c3": c3_app}


def main() -> None:
    servers = []
    for name, app in APPS.items():
        port = MOCK_PORTS[name]
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        # Signal handlers can only be installed on the main thread.
        server.install_signal_handlers = lambda: None
        threading.Thread(target=server.run, name=f"{name}-mock", daemon=True).start()
        servers.append((name, port, server))

    while not all(s.started for _, _, s in servers):
        time.sleep(0.1)
    for name, port, _ in servers:
        print(f"{name.upper()} mock running -> http://localhost:{port}/docs")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for _, _, s in servers:
            s.should_exit = True
        print("Stopped.")


if __name__ == "__main__":
    main()
