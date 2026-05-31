# MIT License
#
# Copyright (c) 2026
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""FastAPI web server for the Deep Research Agent.

Run with:
    uv run uvicorn server:app --reload --port 8000
"""

import asyncio
import builtins
import contextvars
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_log = logging.getLogger(__name__)

_TIMEOUT = float(os.environ.get("RESEARCH_TIMEOUT", "600"))

# Patch builtins.print before importing main. main.py snapshots print via
# functools.partial at import time, so this must come first or node progress
# calls bypass the SSE queue entirely.
_req_queue: contextvars.ContextVar[tuple | None] = contextvars.ContextVar(
    "_req_queue", default=None
)

_orig_print = builtins.print


def _web_print(*args, **kwargs) -> None:
    entry = _req_queue.get()
    if entry is not None:
        loop, q = entry
        msg = " ".join(str(a) for a in args)
        if not loop.is_closed():
            try:
                loop.call_soon_threadsafe(q.put_nowait, msg)
            except RuntimeError:
                pass
    _orig_print(*args, **kwargs)


builtins.print = _web_print

from main import ResearchState, _provider, build_research_graph, llm  # noqa: E402


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    _log.info("startup provider=%s timeout=%ss", _provider or "none", int(_TIMEOUT))
    yield
    _log.info("shutdown")


app = FastAPI(title="Deep Research Agent", docs_url=None, redoc_url=None, lifespan=_lifespan)


@app.middleware("http")
async def _security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "0"  # CSP supersedes legacy header
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "base-uri 'self'; "
        "form-action 'none'"
    )
    return response
app.mount("/static", StaticFiles(directory="static"), name="static")

_INDEX = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_INDEX.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "provider": _provider or "none — no API key set"})


@app.get("/stream")
async def stream(
    query: str = Query(..., min_length=1, max_length=2000),
) -> StreamingResponse:
    if llm is None:

        async def _no_key() -> AsyncGenerator[str, None]:
            yield (
                'data: {"type":"error",'
                '"message":"No LLM configured. Set OPENAI_API_KEY or MISTRAL_API_KEY."}\n\n'
            )

        return StreamingResponse(_no_key(), media_type="text/event-stream")

    async def generate() -> AsyncGenerator[str, None]:
        _log.info("stream start query_len=%d", len(query))
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[str | None] = asyncio.Queue()

        _req_queue.set((loop, q))

        initial_state: ResearchState = {
            "original_query": query,
            "search_plan": "",
            "subtopics": [],
            "search_queries": [],
            "parallel_results": [],
            "sources": [],
            "evaluation_feedback": "",
            "loop_count": 0,
            "is_complete": False,
            "final_report": "",
        }

        agent = build_research_graph()
        result: dict = {}

        async def _run() -> None:
            try:
                state = await asyncio.to_thread(agent.invoke, initial_state)
                result["report"] = state.get("final_report", "")
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        task = asyncio.create_task(_run())

        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=_TIMEOUT)
            except asyncio.TimeoutError:
                _log.warning("stream timeout after %ss", int(_TIMEOUT))
                yield f'data: {{"type":"error","message":"Request timed out ({int(_TIMEOUT)} s)."}}\n\n'
                task.cancel()
                return

            if msg is None:
                break

            yield f"data: {json.dumps({'type': 'progress', 'message': msg})}\n\n"

        await task

        if "error" in result:
            _log.error("stream error: %s", result["error"])
            yield f"data: {json.dumps({'type': 'error', 'message': result['error']})}\n\n"
        elif "report" in result:
            _log.info("stream complete report_len=%d", len(result["report"]))
            yield f"data: {json.dumps({'type': 'result', 'report': result['report']})}\n\n"
            yield 'data: {"type":"done"}\n\n'
        else:
            _log.error("stream: agent returned no report")
            yield 'data: {"type":"error","message":"Agent returned no report."}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
