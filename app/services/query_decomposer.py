"""LLM-based query decomposition using gpt-4o-mini.

Parses a natural language query into structured metadata filters and a residual
semantic query, using the known canonical values for each field.
"""

import json

from google import genai

from app.config import (
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
)
from app.models import QueryDecomposition, SearchFilters

_chat_client: genai.Client | None = None

CANONICAL_VALUES = {
    "document_type": ["bio", "knowledge_library"],
    "designation": [
        "Managing Director",
        "Senior Director",
        "Director",
        "Senior Manager",
        "Manager",
        "Senior Associate",
        "Associate",
        "Analyst",
    ],
    "knowledge_type": [
        "Case Studies and Credentials",
        "Pitches and Proposals",
        "Deliverables",
        "Tools and Templates",
        "Points of View",
        "Research and Benchmarks",
        "Approaches and Methodologies",
        "Trainings",
        "Service Offerings",
    ],
    "knowledge_sub_type": ["Case Study", "Knowledge Nugget", "Credential"],
    "industry": [
        "Healthcare",
        "Industrial / Capital Goods",
        "Consumer Goods",
        "Financial Services",
        "Materials",
        "Pharma and Life Sciences",
        "Retail",
        "Diversified Financials",
        "Automobiles and Components",
        "Transportation",
        "Commercial and Professional Services",
        "Consumer Services",
        "Clean Energy",
        "Infrastructure",
        "Software and Services",
        "Chemicals",
        "Power and Utilities",
        "Telecommunication Services",
        "Real Estate",
        "Media",
        "Public Sector",
        "Travel Hospitality and Leisure",
        "Technology Hardware and Equipment",
        "Insurance",
        "Oil and Gas",
    ],
    "client_type": [
        "Corporate",
        "Investor - PE",
        "Bank",
        "Others",
        "Investor - Distressed fund",
        "Investor - VC",
    ],
    "year": ["2022", "2023", "2024", "2025", "2026"],
    "solution": [
        "Commercial Due Diligence",
        "Manufacturing and Supply Chain Transformation",
        "Overall Business Transformation",
        "Merger Integration and Carve-outs",
        "Operational Due Diligence and Assessments",
        "Integrated Due Diligence",
        "Finance and Business Process Optimization",
        "Business Plan Assessment/Development",
        "Growth Strategy",
        "Cash and Working Capital Management",
        "Market/Category Entry",
        "Interim Management",
        "Commercial Excellence",
        "Restructuring Advisory (Creditor - Debtor - Investor)",
        "Insolvency Resolution Process",
        "Digital",
    ],
}

SYSTEM_PROMPT = """You are a query decomposition engine for a Knowledge Management search system.
Your job is to parse a user's natural language query into two parts:
1. **filters**: structured metadata filters that map to known database fields
2. **semantic_query**: the remaining natural language intent for vector similarity search

The available filter fields and their canonical values are:
{canonical_json}

IMPORTANT DOCUMENT TYPE RULES:
- There are two document types: "bio" and "knowledge_library"
- Bio documents ONLY have these filterable fields: document_type, designation, year
- Knowledge Library documents have: document_type, knowledge_type, knowledge_sub_type, industry, sub_industry, solution, client_name, client_type, year
- If a query is about bios/people, do NOT set industry/solution/knowledge_type as filters.
  Instead, put domain/topic terms (like "healthcare", "supply chain") into semantic_query.
- Example: "healthcare MDs" -> filters: {{document_type: "bio", designation: "Managing Director"}}, semantic_query: "healthcare"
- Example: "CDD experts" -> filters: {{document_type: "bio"}}, semantic_query: "commercial due diligence experts"
- Example: "supply chain case studies" -> filters: {{document_type: "knowledge_library", knowledge_type: "Case Studies and Credentials"}}, semantic_query: "supply chain"

Shorthand mappings:
- "MDs" or "MD" -> designation = "Managing Director"
- "bios" or "bio" -> document_type = "bio"
- "case studies" -> knowledge_type = "Case Studies and Credentials"
- "CDD" -> solution = "Commercial Due Diligence"
- "ODD" -> solution = "Operational Due Diligence and Assessments"
- "PE" -> client_type = "Investor - PE"

Rules:
- Only include a filter if you are confident the query refers to that field.
- If a query part is ambiguous between a filter and semantic intent, prefer semantic.
- The semantic_query should contain whatever is left after extracting filters. It can be empty.
- Return ONLY valid JSON matching the schema below. No markdown fences, no explanation.

Output JSON schema:
{{"filters": {{"document_type": "...", "designation": "...", ...}}, "semantic_query": "..."}}
Only include filter keys that have a value. Omit keys with null/empty values."""


def _get_chat_client() -> genai.Client:
    global _chat_client
    if _chat_client is None:
        _chat_client = genai.Client(
            api_key=GEMINI_API_KEY
        )
    return _chat_client


def decompose_query(query: str) -> QueryDecomposition:
    """Parse a user query into structured filters + residual semantic query."""
    if not query.strip():
        return QueryDecomposition()

    client = _get_chat_client()
    system_msg = SYSTEM_PROMPT.format(canonical_json=json.dumps(CANONICAL_VALUES, indent=2))

    try:
        response = client.models.generate_content_stream(
            model=GEMINI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=500,
        )

        content = response.choices[0].message.content or "{}"
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]

        parsed = json.loads(content)
        filters_dict = parsed.get("filters", {})
        semantic = parsed.get("semantic_query", "")

        filters = SearchFilters(**{k: v for k, v in filters_dict.items() if v})
        return QueryDecomposition(filters=filters, semantic_query=semantic)

    except Exception:
        return QueryDecomposition(semantic_query=query)
