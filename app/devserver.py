"""Local development server entrypoint.

Why this exists
---------------
On Windows, uvicorn 0.36+ defaults to ``asyncio.ProactorEventLoop`` (see
``uvicorn/loops/asyncio.py``: it only returns the selector loop when a
subprocess is needed). The Proactor IOCP accept path can fail with::

    OSError: [WinError 64] The specified network name is no longer available
    Accept failed on a socket
    socket: <asyncio.TransportSocket fd=... laddr=('127.0.0.1', 8000)>

which tears down the whole server process. The selector loop never touches
that code path, so local development pins it explicitly.

Usage (from ``backend/``)::

    .\\venv\\Scripts\\python.exe -m app.devserver
    .\\venv\\Scripts\\python.exe -m app.devserver --port 8001
    .\\venv\\Scripts\\python.exe -m app.devserver --reload
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import uvicorn

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def create_event_loop(use_subprocess: bool = False) -> asyncio.AbstractEventLoop:
    """Return a fresh event loop that avoids the Windows IOCP accept bug.

    uvicorn passes a custom ``--loop`` target straight to ``asyncio.Runner``,
    which calls it with no arguments and expects a loop instance back. The
    ``use_subprocess`` parameter is accepted so the signature also satisfies
    uvicorn's built-in loop-factory contract.
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KM Search API locally.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart on file changes. Note the ETL job writes into app/db, "
        "which will trigger restarts while it runs.",
    )
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        loop="app.devserver:create_event_loop",
        http="h11",
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
