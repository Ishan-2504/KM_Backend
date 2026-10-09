from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    BIO = "bio"
    KNOWLEDGE_LIBRARY = "knowledge_library"


class SearchMode(str, Enum):
    FILTER_ONLY = "filter_only"
    OPEN_SEARCH = "open_search"
    HYBRID = "hybrid"


COLUMN_MAP: dict[str, str] = {
    "documentName": "document_name",
    "File_Text": "file_text",
    "Document type": "document_type",
    "designation": "designation",
    "Knowledge Type": "knowledge_type",
    "Knowledge Sub-type": "knowledge_sub_type",
    "industry": "industry",
    "subIndustry": "sub_industry",
    "Solution": "solution",
    "Client Name": "client_name",
    "Client Type": "client_type",
    "Year": "year",
    "link": "link",
    "summary": "summary",
    "Taggings": "taggings",
}

DOC_TYPE_MAP: dict[str, str] = {
    "Bios": "bio",
    "Knowledge Library": "knowledge_library",
}

NULL_MARKERS = {"Not Specified", "NA", "N/A", "", "None"}

FILTERABLE_FIELDS = [
    "document_type",
    "designation",
    "knowledge_type",
    "knowledge_sub_type",
    "industry",
    "sub_industry",
    "solution",
    "client_name",
    "client_type",
    "year",
]

METADATA_FIELDS = [
    "document_name",
    "document_type",
    "designation",
    "knowledge_type",
    "knowledge_sub_type",
    "industry",
    "sub_industry",
    "solution",
    "client_name",
    "client_type",
    "year",
    "link",
    "taggings",
]


class DocumentRecord(BaseModel):
    document_name: str
    file_text: str = ""
    document_type: str
    designation: Optional[str] = None
    knowledge_type: Optional[str] = None
    knowledge_sub_type: Optional[str] = None
    industry: Optional[str] = None
    sub_industry: Optional[str] = None
    solution: Optional[str] = None
    client_name: Optional[str] = None
    client_type: Optional[str] = None
    year: Optional[str] = None
    link: Optional[str] = None
    summary: str = ""
    taggings: Optional[str] = None


class SearchFilters(BaseModel):
    document_type: Optional[str] = None
    designation: Optional[str] = None
    knowledge_type: Optional[str] = None
    knowledge_sub_type: Optional[str] = None
    industry: Optional[str] = None
    sub_industry: Optional[str] = None
    solution: Optional[str] = None
    client_name: Optional[str] = None
    client_type: Optional[str] = None
    year: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = ""
    filters: Optional[SearchFilters] = None
    top_k: int = Field(default=500, ge=1, le=5000)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=2.0)


class SearchResult(BaseModel):
    document_name: str
    summary: str
    document_type: str
    designation: Optional[str] = None
    knowledge_type: Optional[str] = None
    knowledge_sub_type: Optional[str] = None
    industry: Optional[str] = None
    sub_industry: Optional[str] = None
    solution: Optional[str] = None
    client_name: Optional[str] = None
    client_type: Optional[str] = None
    year: Optional[str] = None
    link: Optional[str] = None
    taggings: Optional[str] = None
    similarity_score: Optional[float] = None
    relevance_pct: Optional[float] = None
    rank: int = 0


class SearchMoreRequest(BaseModel):
    session_id: str
    page: int = Field(default=2, ge=2)


class PaginatedSearchResponse(BaseModel):
    results: list[SearchResult] = []
    page: int = 1
    has_more: bool = False
    session_id: str = ""
    total_relevant: int = 0
    message: Optional[str] = None


class FilterOptions(BaseModel):
    document_type: list[str] = []
    designation: list[str] = []
    knowledge_type: list[str] = []
    knowledge_sub_type: list[str] = []
    industry: list[str] = []
    sub_industry: list[str] = []
    solution: list[str] = []
    client_name: list[str] = []
    client_type: list[str] = []
    year: list[str] = []


class QueryDecomposition(BaseModel):
    filters: SearchFilters = Field(default_factory=SearchFilters)
    semantic_query: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class RankedDocument(BaseModel):
    rank: int
    document_name: str
    document_type: str = ""
    summary: str = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    document_names: list[str] = []
    ranked_list: list[RankedDocument] = []
