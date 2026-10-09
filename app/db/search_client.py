"""Local FAISS index with JSON metadata for KM document search."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import faiss
import numpy as np

from app.config import (
    BACKEND_ROOT,
    FAISS_DIR,
    VECTOR_DIMENSIONS,
    VECTOR_INDEX_NAME,
)
from app.models import FILTERABLE_FIELDS

_client: "FaissSearchClient | None" = None
_client_lock = threading.Lock()


def _resolve_index_dir() -> Path:
    path = Path(FAISS_DIR)
    if not path.is_absolute():
        path = (BACKEND_ROOT / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _filters_dict(filters: dict | None) -> dict[str, str | list[str]]:
    if not filters:
        return {}
    return {k: v for k, v in filters.items() if v not in (None, "", [])}


def _doc_matches(doc: dict, filters: dict[str, str | list[str]]) -> bool:
    for field, expected in filters.items():
        actual = doc.get(field) or ""
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _select_fields(doc: dict, select: list[str] | None, score: float | None) -> dict:
    if select:
        hit = {field: doc.get(field, "") for field in select}
    else:
        hit = dict(doc)
    hit["_score"] = score
    return hit


class FaissSearchClient:
    def __init__(self, index_name: str = VECTOR_INDEX_NAME):
        self.index_name = index_name
        self._dir = _resolve_index_dir()
        self._index_path = self._dir / f"{index_name}.index"
        self._meta_path = self._dir / f"{index_name}_meta.json"
        self._lock = threading.Lock()
        self.index = faiss.IndexFlatIP(VECTOR_DIMENSIONS)
        self._metadata: list[dict] = []
        self._embeddings = np.zeros((0, VECTOR_DIMENSIONS), dtype=np.float32)
        self._load()

    def _init_empty(self) -> None:
        self.index = faiss.IndexFlatIP(VECTOR_DIMENSIONS)
        self._metadata = []
        self._embeddings = np.zeros((0, VECTOR_DIMENSIONS), dtype=np.float32)

    def _load(self) -> None:
        if not self._index_path.exists() or not self._meta_path.exists():
            self._init_empty()
            return

        index = faiss.read_index(str(self._index_path))
        with self._meta_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

        if index.d != VECTOR_DIMENSIONS:
            raise ValueError(
                f"FAISS index dimension {index.d} does not match VECTOR_DIMENSIONS={VECTOR_DIMENSIONS}"
            )
        if index.ntotal != len(metadata):
            print(
                f"WARNING: index size ({index.ntotal}) != metadata rows ({len(metadata)}). "
                "Rebuilding an empty index."
            )
            self._init_empty()
            return

        self.index = index
        self._metadata = metadata
        if index.ntotal == 0:
            self._embeddings = np.zeros((0, VECTOR_DIMENSIONS), dtype=np.float32)
        else:
            self._embeddings = np.ascontiguousarray(
                index.reconstruct_n(0, index.ntotal),
                dtype=np.float32,
            )

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self._index_path))
        with self._meta_path.open("w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)

    def reset(self) -> None:
        with self._lock:
            self._init_empty()
            self._persist()

    def get_document_count(self) -> int:
        with self._lock:
            return len(self._metadata)

    def upload_documents(self, documents: list[dict]) -> None:
        if not documents:
            return

        vectors: list[np.ndarray] = []
        metas: list[dict] = []
        for doc in documents:
            emb = doc.get("embedding")
            if emb is None:
                raise ValueError(f"Document '{doc.get('document_name', '')}' is missing an embedding")
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            if vec.shape[0] != VECTOR_DIMENSIONS:
                raise ValueError(
                    f"Embedding dim {vec.shape[0]} != {VECTOR_DIMENSIONS} for '{doc.get('document_name', '')}'"
                )
            vectors.append(vec)
            metas.append({k: ("" if v is None else v) for k, v in doc.items() if k != "embedding"})

        mat = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
        faiss.normalize_L2(mat)

        with self._lock:
            self.index.add(mat)
            self._embeddings = mat if self._embeddings.shape[0] == 0 else np.vstack(
                [self._embeddings, mat]
            )
            self._metadata.extend(metas)
            self._persist()

    def get_facets(
        self,
        filters: dict[str, str | list[str]] | None = None,
        fields: list[str] | None = None,
    ) -> dict[str, list[dict]]:
        active = _filters_dict(filters)
        facet_fields = fields or FILTERABLE_FIELDS
        with self._lock:
            docs = [d for d in self._metadata if _doc_matches(d, active)]

        facets: dict[str, list[dict]] = {}
        for field in facet_fields:
            counts: dict[str, int] = {}
            for doc in docs:
                value = doc.get(field)
                if value is None:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                counts[text] = counts.get(text, 0) + 1
            facets[field] = [{"value": value, "count": count} for value, count in counts.items()]
        return facets

    def search(
        self,
        *,
        query_vector: list[float] | None = None,
        filters: dict[str, str | list[str]] | None = None,
        top_k: int = 50,
        select: list[str] | None = None,
    ) -> list[dict]:
        active = _filters_dict(filters)
        k = max(1, top_k)

        with self._lock:
            candidate_ids = [i for i, doc in enumerate(self._metadata) if _doc_matches(doc, active)]
            if not candidate_ids:
                return []

            if query_vector is None:
                ranked = sorted(
                    candidate_ids,
                    key=lambda i: self._metadata[i].get("year") or "0000",
                    reverse=True,
                )[:k]
                return [_select_fields(self._metadata[i], select, None) for i in ranked]

            query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
            if query.shape[1] != VECTOR_DIMENSIONS:
                raise ValueError(
                    f"Query embedding dim {query.shape[1]} != {VECTOR_DIMENSIONS}"
                )
            faiss.normalize_L2(query)

            if len(candidate_ids) == self.index.ntotal:
                scores, indices = self.index.search(query, min(k, self.index.ntotal))
                hits: list[dict] = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0:
                        continue
                    hits.append(_select_fields(self._metadata[int(idx)], select, float(score)))
                return hits

            subset = self._embeddings[candidate_ids]
            scores = subset @ query.reshape(-1)
            order = np.argsort(-scores)[:k]
            return [
                _select_fields(
                    self._metadata[candidate_ids[int(i)]],
                    select,
                    float(scores[int(i)]),
                )
                for i in order
            ]


def get_search_client() -> FaissSearchClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = FaissSearchClient()
    return _client


def ensure_index() -> None:
    """Load the on-disk FAISS index, or create an empty one."""
    client = get_search_client()
    print(f"Index '{client.index_name}' is ready ({client.get_document_count()} documents).")
