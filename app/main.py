import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from sse_starlette.sse import EventSourceResponse

from app.config import GEMINI_API_KEY, GEMINI_CHAT_MODEL
from app.db.search_client import ensure_index, get_search_client
from app.models import (
    FILTERABLE_FIELDS,
    ChatRequest,
    FilterOptions,
    PaginatedSearchResponse,
    RankedDocument,
    SearchFilters,
    SearchMoreRequest,
    SearchRequest,
    active_filter_dict,
)
from app.services.embeddings import embed_single
from app.services.search import execute_search_paginated, get_search_page

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_index()
    yield


app = FastAPI(title="KM Search API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    client = get_search_client()
    count = client.get_document_count()
    return {"status": "ok", "document_count": count}


async def _fetch_filter_options(active_filters: SearchFilters | None = None) -> FilterOptions:
    """Return faceted filter options, optionally narrowed by active selections."""
    client = get_search_client()
    facets = client.get_facets(active_filter_dict(active_filters) or None)

    options: dict[str, list[str]] = {}
    for field in FILTERABLE_FIELDS:
        facet_values = facets.get(field, [])
        vals = sorted(
            v["value"] for v in facet_values
            if v.get("value") and str(v["value"]).strip()
        )
        options[field] = vals

    return FilterOptions(**options)


@app.get("/api/filters", response_model=FilterOptions)
async def get_filters():
    """Return all available filter options (no narrowing)."""
    return await _fetch_filter_options()


@app.post("/api/filters", response_model=FilterOptions)
async def get_filters_cascading(active_filters: SearchFilters):
    """Return filter options narrowed by the caller's current selections."""
    return await _fetch_filter_options(active_filters)


async def _search_event_generator(request: SearchRequest):
    """Generator that yields first page of search results as SSE events."""
    start = time.time()

    yield {
        "event": "start",
        "data": json.dumps({"query": request.query}),
    }

    loop = asyncio.get_event_loop()
    first_page, session_id, has_more, total_relevant = await loop.run_in_executor(
        None, execute_search_paginated, request,
    )

    for result in first_page:
        yield {
            "event": "result",
            "data": result.model_dump_json(),
        }
        await asyncio.sleep(0.01)

    elapsed = round(time.time() - start, 3)
    yield {
        "event": "done",
        "data": json.dumps({
            "total": len(first_page),
            "total_relevant": total_relevant,
            "elapsed_seconds": elapsed,
            "session_id": session_id,
            "has_more": has_more,
        }),
    }


@app.post("/api/search")
async def search(request: SearchRequest):
    """Main search endpoint returning first page of results via SSE stream."""
    return EventSourceResponse(_search_event_generator(request))


@app.post("/api/search/more", response_model=PaginatedSearchResponse)
async def search_more(request: SearchMoreRequest):
    """Return the next page of results from a cached search session."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_search_page, request)


@app.post("/api/search/sync")
async def search_sync(request: SearchRequest):
    """Non-streaming search endpoint for simpler clients."""
    loop = asyncio.get_event_loop()
    first_page, session_id, has_more, total_relevant = await loop.run_in_executor(
        None, execute_search_paginated, request,
    )
    return {
        "results": [r.model_dump() for r in first_page],
        "total": len(first_page),
        "total_relevant": total_relevant,
        "session_id": session_id,
        "has_more": has_more,
    }


RAG_SELECT_FIELDS = [
    "document_name", "summary", "file_text",
    "document_type", "designation", "knowledge_type", "knowledge_sub_type",
    "industry", "sub_industry", "solution",
    "client_name", "client_type", "year", "link", "taggings",
]

RAG_TOP_K = 10
MAX_FILE_TEXT_CHARS = 6000


def _retrieve_for_chat(question: str, document_names: list[str]) -> list[dict]:
    """RAG retrieval: embed the question and search FAISS, optionally scoped to document names."""
    client = get_search_client()
    query_embedding = embed_single(question)
    filters = {"document_name": document_names} if document_names else None

    results = client.search(
        query_vector=query_embedding,
        filters=filters,
        select=RAG_SELECT_FIELDS,
        top_k=RAG_TOP_K,
    )

    return [{field: hit.get(field, "") for field in RAG_SELECT_FIELDS} for hit in results]


def _build_rag_context(docs: list[dict]) -> str:
    """Format retrieved documents into a text block for the system prompt."""
    if not docs:
        return "No documents are currently loaded."

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        meta_pairs = []
        for key, label in [
            ("document_type", "Type"), ("designation", "Designation"),
            ("industry", "Industry"), ("sub_industry", "Sub-Industry"),
            ("knowledge_type", "Knowledge Type"), ("knowledge_sub_type", "Knowledge Sub-Type"),
            ("solution", "Solution"), ("client_name", "Client"),
            ("client_type", "Client Type"), ("year", "Year"),
        ]:
            val = doc.get(key)
            if val:
                meta_pairs.append(f"{label}: {val}")

        meta_line = " | ".join(meta_pairs) if meta_pairs else "No metadata"
        file_text = (doc.get("file_text") or "")[:MAX_FILE_TEXT_CHARS]
        summary = doc.get("summary") or "No summary available"

        section = (
            f"=== Document {i}: {doc.get('document_name', 'Unknown')} ===\n"
            f"{meta_line}\n"
            f"Summary: {summary}"
        )
        if file_text:
            section += f"\nContent:\n{file_text}"

        parts.append(section)
    return "\n\n".join(parts)


CHAT_SYSTEM_PROMPT = (
    "You are a knowledgeable document analysis assistant for BTS Knowledge Search. "
    "Answer questions ONLY using the provided documents below. "
    "If the answer cannot be found in the documents, clearly state that the information "
    "is not available in the current document set. "
    "Be concise, accurate, and cite document names when referencing specific information.\n\n"
    "The user is viewing search results in the order shown below. "
    "When they refer to documents by position (e.g. '1st document', '3rd result'), "
    "use this display order:\n"
    "--- SEARCH RESULTS (display order) ---\n{ranked_list}\n--- END SEARCH RESULTS ---\n\n"
    "The following documents contain detailed content retrieved for the user's question:\n"
    "--- DOCUMENT DETAILS ---\n{documents}\n--- END DOCUMENT DETAILS ---"
)


def _build_ranked_list_text(ranked_list: list[RankedDocument]) -> str:
    """Format the ranked display list into a numbered text block."""
    if not ranked_list:
        return "No search results displayed."
    lines = []
    for doc in sorted(ranked_list, key=lambda d: d.rank):
        line = f"{doc.rank}. {doc.document_name}"
        if doc.document_type:
            line += f" [{doc.document_type}]"
        if doc.summary:
            short = doc.summary[:150].rstrip()
            if len(doc.summary) > 150:
                short += "..."
            line += f" - {short}"
        lines.append(line)
    return "\n".join(lines)


def _get_gemini_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def _to_gemini_contents(messages: list) -> list[types.Content]:
    contents: list[types.Content] = []

    for message in messages:
        if message.role not in ("user", "assistant"):
            continue

        role = "user" if message.role == "user" else "model"

        contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part.from_text(
                        text=message.content
                    )
                ],
            )
        )

    # Gemini requires the conversation to end with a user message
    if contents and contents[-1].role == "model":
        contents.pop()

    return contents


async def _chat_stream_generator(request: ChatRequest):
    """RAG-based chat: retrieve relevant docs, then stream LLM response."""
    user_messages = [m for m in request.messages if m.role == "user"]
    latest_question = user_messages[-1].content if user_messages else ""

    loop = asyncio.get_event_loop()
    retrieved_docs = await loop.run_in_executor(
        None, _retrieve_for_chat, latest_question, request.document_names,
    )

    ranked_text = _build_ranked_list_text(request.ranked_list)
    context = _build_rag_context(retrieved_docs)
    system_text = CHAT_SYSTEM_PROMPT.format(ranked_list=ranked_text, documents=context)
    contents = _to_gemini_contents(request.messages)

    client = _get_gemini_client()

    def _call_gemini():
        return client.models.generate_content_stream(
            model=GEMINI_CHAT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_text),
        )

    try:
        stream = await loop.run_in_executor(None, _call_gemini)
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield f"data: {json.dumps({'token': text})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        logging.error("Chat stream error: %s", e)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """RAG-powered streaming chat endpoint grounded in search result documents."""
    return StreamingResponse(
        _chat_stream_generator(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
