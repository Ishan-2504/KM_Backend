"""Unified search pipeline using Gemini embeddings, FAISS retrieval, and score fusion."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

from app.config import RANKING_RELEVANCE_FLOOR, RANKING_TIER_THRESHOLDS
from app.indexing.factory import get_search_backend
from app.models import (
    PaginatedSearchResponse,
    SearchFilters,
    SearchMoreRequest,
    SearchRequest,
    SearchResult,
    active_filter_dict,
)
from app.services.filter_extractor import ExtractionResult, extract_filters
from app.services.query_expander import expand_query
from app.services.search_logger import new_search_log, step_ms, timed_step

TIER_THRESHOLDS = RANKING_TIER_THRESHOLDS or [63.5, 60.0, 55.5]
RELEVANCE_FLOOR = RANKING_RELEVANCE_FLOOR
SESSION_TTL = 600

ALL_SELECT_FIELDS = [
    "id", "document_name", "summary", "document_type", "designation",
    "knowledge_type", "knowledge_sub_type", "industry", "sub_industry",
    "solution", "client_name", "client_type", "year", "link", "taggings",
]


def _has_active_filters(filters: SearchFilters | None) -> bool:
    return bool(active_filter_dict(filters))


def _resolve_mode(query: str, has_manual_filters: bool, has_extracted_filters: bool) -> str:
    has_filters = has_manual_filters or has_extracted_filters
    if query.strip() and has_filters:
        return "hybrid"
    if query.strip():
        return "open_search"
    if has_filters:
        return "filter_only"
    return "none"


def _result_from_hit(hit: dict, rank: int = 0, fused_score: float | None = None) -> SearchResult:
    similarity = fused_score
    if similarity is None and hit.get("_score") is not None:
        similarity = round(float(hit["_score"]), 4)

    return SearchResult(
        document_name=hit.get("document_name", ""),
        summary=hit.get("summary", ""),
        document_type=hit.get("document_type", ""),
        designation=hit.get("designation") or None,
        knowledge_type=hit.get("knowledge_type") or None,
        knowledge_sub_type=hit.get("knowledge_sub_type") or None,
        industry=hit.get("industry") or None,
        sub_industry=hit.get("sub_industry") or None,
        solution=hit.get("solution") or None,
        client_name=hit.get("client_name") or None,
        client_type=hit.get("client_type") or None,
        year=hit.get("year") or None,
        link=hit.get("link") or None,
        taggings=hit.get("taggings") or None,
        similarity_score=similarity,
        rank=rank,
    )


def _compute_fused_score(
    base_score: float,
    hit: dict,
    extraction: ExtractionResult,
) -> float:
    """Combine cosine similarity with metadata-alignment boost.

    finalScore = baseScore * 0.7 + metadataBoost * 0.3
    """
    meta_boost = 0.0
    max_signals = 0

    if extraction.inferred_doc_type:
        max_signals += 1
        if extraction.inferred_doc_type == hit.get("document_type"):
            meta_boost += 1.0

    if extraction.inferred_industry:
        max_signals += 1
        if extraction.inferred_industry == hit.get("industry"):
            meta_boost += 1.0

    if extraction.inferred_solution:
        max_signals += 1
        if extraction.inferred_solution == hit.get("solution"):
            meta_boost += 1.0

    normalised = (meta_boost / max(max_signals, 1))
    return base_score * 0.7 + normalised * 0.3


def _filter_only_search(
    filters: SearchFilters,
    top_k: int,
) -> list[SearchResult]:
    backend = get_search_backend()
    hits = backend.search(
        query="",
        filters=active_filter_dict(filters),
        select=ALL_SELECT_FIELDS,
        top_k=top_k,
    )

    output = [_result_from_hit(hit) for hit in hits]
    for i, result in enumerate(output):
        result.rank = i + 1
    return output


def _fused_to_relevance_pct(fused: float) -> float:
    return round(max(0.0, min(fused * 100.0, 100.0)), 1)


def _vector_search(
    query: str,
    filters: SearchFilters | None,
    extraction: ExtractionResult,
    top_k: int,
    similarity_threshold: float,
) -> list[SearchResult]:
    backend = get_search_backend()
    hits = backend.search(
        query=query,
        filters=active_filter_dict(filters) or None,
        select=ALL_SELECT_FIELDS,
        top_k=top_k,
    )

    score_floor = 1.0 - similarity_threshold
    output: list[SearchResult] = []
    for hit in hits:
        base_score = float(hit.get("_score") or 0.0)
        if base_score < score_floor:
            continue
        fused = _compute_fused_score(base_score, hit, extraction)
        result = _result_from_hit(hit, fused_score=fused)
        result.relevance_pct = _fused_to_relevance_pct(fused)
        output.append(result)

    output.sort(key=lambda r: r.similarity_score or 0, reverse=True)
    for i, result in enumerate(output):
        result.rank = i + 1
    return output


def execute_search(request: SearchRequest) -> list[SearchResult]:
    """Unified search pipeline: expand -> extract -> route -> search -> fuse -> return."""
    total_start = time.perf_counter()
    raw_query = request.query or ""

    manual_filters_dict = active_filter_dict(request.filters) or None
    slog = new_search_log(raw_query, manual_filters_dict)

    with timed_step("QUERY EXPANSION") as s1:
        expanded_query = expand_query(raw_query)
        s1["input"] = raw_query
        s1["output"] = expanded_query
    slog.add_step("QUERY EXPANSION", step_ms(s1), input=raw_query, output=expanded_query)

    with timed_step("FILTER EXTRACTION") as s2:
        extraction = extract_filters(expanded_query, request.filters)
        extracted_dict = active_filter_dict(extraction.filters)
        s2["extracted_filters"] = extracted_dict
        s2["semantic_query"] = extraction.semantic_query
    slog.add_step(
        "FILTER EXTRACTION", step_ms(s2),
        extracted_filters=extracted_dict,
        semantic_query=extraction.semantic_query,
    )

    with timed_step("MODE RESOLUTION") as s3:
        has_manual = _has_active_filters(request.filters)
        has_extracted = _has_active_filters(extraction.filters)
        mode = _resolve_mode(expanded_query, has_manual, has_extracted)
        if has_manual:
            reason = "query + manual filters" if expanded_query.strip() else "manual filters only"
        elif has_extracted:
            reason = "query + extracted filters" if expanded_query.strip() else "extracted filters only"
        elif expanded_query.strip():
            reason = "query present, no filters"
        else:
            reason = "no query and no filters"
        s3["mode"] = mode
        s3["reason"] = reason
    slog.add_step("MODE RESOLUTION", step_ms(s3), mode=mode, reason=reason)

    if mode == "none":
        slog.add_step("SEARCH", 0.0, result="skipped (no inputs)")
        slog.result_count = 0
        slog.total_ms = (time.perf_counter() - total_start) * 1000
        slog.emit()
        return []

    with timed_step("SEARCH") as s4:
        if mode == "filter_only":
            results = _filter_only_search(extraction.filters, request.top_k)
            s4["type"] = "metadata-only (FAISS metadata filter)"
            s4["candidate_pool"] = len(results)
            s4["threshold"] = "N/A"
            s4["passing"] = len(results)
        else:
            merged_filters = extraction.filters
            semantic_q = extraction.semantic_query or expanded_query
            results = _vector_search(
                query=semantic_q,
                filters=merged_filters if _has_active_filters(merged_filters) else None,
                extraction=extraction,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
            )
            s4["type"] = f"hybrid+vector ({mode})"
            s4["semantic_query"] = semantic_q
            s4["filter_clause"] = extracted_dict or "(none)"
            s4["threshold"] = f"score >= {round(1 - request.similarity_threshold, 2)}"
            s4["passing"] = len(results)

    slog.add_step("SEARCH", step_ms(s4), **{k: v for k, v in s4.items()})

    if results and results[0].similarity_score is not None:
        with timed_step("SCORE FUSION SUMMARY") as s5:
            scores = [r.similarity_score for r in results if r.similarity_score is not None]
            s5["final_count"] = len(results)
            s5["top_score"] = round(max(scores), 4) if scores else 0
            s5["bottom_score"] = round(min(scores), 4) if scores else 0
        slog.add_step("SCORE FUSION SUMMARY", step_ms(s5), **{k: v for k, v in s5.items()})

    slog.result_count = len(results)
    slog.total_ms = (time.perf_counter() - total_start) * 1000
    slog.emit()

    return results


def _split_into_tiers(results: list[SearchResult]) -> list[list[SearchResult]]:
    """Partition results into relevance-based tiers for progressive disclosure."""
    tier_count = len(TIER_THRESHOLDS)
    buckets: list[list[SearchResult]] = [[] for _ in range(tier_count)]

    for r in results:
        if r.relevance_pct is None:
            buckets[0].append(r)
            continue
        if r.relevance_pct < RELEVANCE_FLOOR:
            continue
        placed = False
        for i, threshold in enumerate(TIER_THRESHOLDS):
            if r.relevance_pct >= threshold:
                buckets[i].append(r)
                placed = True
                break
        if not placed:
            buckets[-1].append(r)

    if not buckets[0]:
        collapsed = [r for bucket in buckets for r in bucket]
        return [collapsed] if collapsed else [[]]

    return [b for b in buckets if b]


@dataclass
class _SearchSession:
    tiers: list[list[SearchResult]]
    created_at: float
    total_relevant: int = 0


_sessions: dict[str, _SearchSession] = {}
_sessions_lock = threading.Lock()


def _cleanup_sessions() -> None:
    now = time.time()
    expired = [k for k, v in _sessions.items() if now - v.created_at > SESSION_TTL]
    for k in expired:
        del _sessions[k]


def _store_session(results: list[SearchResult]) -> tuple[str, int]:
    tiers = _split_into_tiers(results)
    total_relevant = sum(len(t) for t in tiers)
    session_id = uuid.uuid4().hex[:16]
    with _sessions_lock:
        _cleanup_sessions()
        _sessions[session_id] = _SearchSession(
            tiers=tiers,
            created_at=time.time(),
            total_relevant=total_relevant,
        )
    return session_id, total_relevant


def execute_search_paginated(request: SearchRequest) -> tuple[list[SearchResult], str, bool, int]:
    all_results = execute_search(request)
    session_id, total_relevant = _store_session(all_results)

    with _sessions_lock:
        session = _sessions.get(session_id)
    if not session:
        return [], session_id, False, 0

    first_page = session.tiers[0] if session.tiers else []
    has_more = len(session.tiers) > 1
    return first_page, session_id, has_more, total_relevant


def get_search_page(request: SearchMoreRequest) -> PaginatedSearchResponse:
    with _sessions_lock:
        session = _sessions.get(request.session_id)

    if not session:
        return PaginatedSearchResponse(
            page=request.page,
            has_more=False,
            session_id=request.session_id,
            message="Session expired. Please search again.",
        )

    tier_index = request.page - 1
    if tier_index >= len(session.tiers):
        return PaginatedSearchResponse(
            page=request.page,
            has_more=False,
            session_id=request.session_id,
            total_relevant=session.total_relevant,
            message="No more relevant documents found.",
        )

    page_results = session.tiers[tier_index]

    rank_offset = sum(len(session.tiers[i]) for i in range(tier_index))
    for i, r in enumerate(page_results):
        r.rank = rank_offset + i + 1

    has_more = tier_index + 1 < len(session.tiers)

    return PaginatedSearchResponse(
        results=page_results,
        page=request.page,
        has_more=has_more,
        session_id=request.session_id,
        total_relevant=session.total_relevant,
    )
