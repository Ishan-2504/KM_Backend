"""Rule-based filter extraction from an expanded query string.

Runs in ~5ms (pure regex / dictionary lookups). Extracts implicit metadata
filters and returns the residual semantic query for vector search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models import SearchFilters

# ── Bio-signal keywords (query likely targets bios / people) ────────────────
BIO_SIGNALS = re.compile(
    r"\b(?:bio|bios|expert|experts|consultant|consultants|people|team|"
    r"managing director|senior director|director|senior manager|manager|"
    r"senior associate|associate|analyst|professionals?|staff|personnel|"
    r"specialists?)\b",
    re.IGNORECASE,
)

# ── KL-signal keywords (query likely targets knowledge library) ─────────────
KL_SIGNALS = re.compile(
    r"\b(?:case stud(?:y|ies)|credential|credentials|pitch|pitches|"
    r"proposal|proposals|deliverable|deliverables|template|templates|"
    r"tool|tools|point of view|research|benchmark|benchmarks|"
    r"methodology|methodologies|approach|approaches|training|trainings|"
    r"service offering|service offerings)\b",
    re.IGNORECASE,
)

# ── Designation patterns ────────────────────────────────────────────────────
DESIGNATION_MAP: dict[str, str] = {
    "managing director": "Managing Director",
    "managing directors": "Managing Director",
    "senior director": "Senior Director",
    "senior directors": "Senior Director",
    "director": "Director",
    "directors": "Director",
    "senior manager": "Senior Manager",
    "senior managers": "Senior Manager",
    "manager": "Manager",
    "managers": "Manager",
    "senior associate": "Senior Associate",
    "senior associates": "Senior Associate",
    "associate": "Associate",
    "associates": "Associate",
    "analyst": "Analyst",
    "analysts": "Analyst",
}

_DESIGNATION_RE = re.compile(
    r"\b({keys})\b".format(
        keys="|".join(
            re.escape(k) for k in sorted(DESIGNATION_MAP, key=len, reverse=True)
        )
    ),
    re.IGNORECASE,
)

# ── Knowledge type patterns ─────────────────────────────────────────────────
KNOWLEDGE_TYPE_MAP: dict[str, str] = {
    "case studies": "Case Studies and Credentials",
    "case study": "Case Studies and Credentials",
    "credentials": "Case Studies and Credentials",
    "credential": "Case Studies and Credentials",
    "pitches": "Pitches and Proposals",
    "pitch": "Pitches and Proposals",
    "proposals": "Pitches and Proposals",
    "proposal": "Pitches and Proposals",
    "deliverables": "Deliverables",
    "deliverable": "Deliverables",
    "tools": "Tools and Templates",
    "templates": "Tools and Templates",
    "tools and templates": "Tools and Templates",
    "point of view": "Points of View",
    "points of view": "Points of View",
    "research": "Research and Benchmarks",
    "benchmarks": "Research and Benchmarks",
    "research and benchmarks": "Research and Benchmarks",
    "methodologies": "Approaches and Methodologies",
    "methodology": "Approaches and Methodologies",
    "approaches": "Approaches and Methodologies",
    "approaches and methodologies": "Approaches and Methodologies",
    "trainings": "Trainings",
    "training": "Trainings",
    "service offerings": "Service Offerings",
    "service offering": "Service Offerings",
}

_KT_RE = re.compile(
    r"\b({keys})\b".format(
        keys="|".join(
            re.escape(k) for k in sorted(KNOWLEDGE_TYPE_MAP, key=len, reverse=True)
        )
    ),
    re.IGNORECASE,
)

# ── Industry patterns ───────────────────────────────────────────────────────
INDUSTRY_CANONICAL: list[str] = [
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
]

INDUSTRY_ALIASES: dict[str, str] = {}
for _ind in INDUSTRY_CANONICAL:
    INDUSTRY_ALIASES[_ind.lower()] = _ind
INDUSTRY_ALIASES.update({
    "pharma": "Pharma and Life Sciences",
    "life sciences": "Pharma and Life Sciences",
    "telecom": "Telecommunication Services",
    "telecommunications": "Telecommunication Services",
    "auto": "Automobiles and Components",
    "automotive": "Automobiles and Components",
    "automobiles": "Automobiles and Components",
    "fmcg": "Consumer Goods",
    "industrial": "Industrial / Capital Goods",
    "capital goods": "Industrial / Capital Goods",
    "travel": "Travel Hospitality and Leisure",
    "hospitality": "Travel Hospitality and Leisure",
    "leisure": "Travel Hospitality and Leisure",
    "energy": "Clean Energy",
    "utilities": "Power and Utilities",
    "power": "Power and Utilities",
    "real estate": "Real Estate",
    "software": "Software and Services",
    "tech": "Software and Services",
    "technology": "Software and Services",
    "it services": "Software and Services",
    "banking": "Financial Services",
    "bfsi": "Financial Services",
})

_INDUSTRY_RE = re.compile(
    r"\b({keys})\b".format(
        keys="|".join(
            re.escape(k) for k in sorted(INDUSTRY_ALIASES, key=len, reverse=True)
        )
    ),
    re.IGNORECASE,
)

# ── Solution patterns ───────────────────────────────────────────────────────
SOLUTION_ALIASES: dict[str, str] = {
    "commercial due diligence": "Commercial Due Diligence",
    "operational due diligence": "Operational Due Diligence and Assessments",
    "operational due diligence and assessments": "Operational Due Diligence and Assessments",
    "integrated due diligence": "Integrated Due Diligence",
    "merger integration": "Merger Integration and Carve-outs",
    "carve-outs": "Merger Integration and Carve-outs",
    "carve out": "Merger Integration and Carve-outs",
    "mergers and acquisitions": "Merger Integration and Carve-outs",
    "growth strategy": "Growth Strategy",
    "manufacturing and supply chain": "Manufacturing and Supply Chain Transformation",
    "supply chain transformation": "Manufacturing and Supply Chain Transformation",
    "overall business transformation": "Overall Business Transformation",
    "business transformation": "Overall Business Transformation",
    "business plan assessment": "Business Plan Assessment/Development",
    "business plan development": "Business Plan Assessment/Development",
    "finance and business process optimization": "Finance and Business Process Optimization",
    "business process optimization": "Finance and Business Process Optimization",
    "cash and working capital": "Cash and Working Capital Management",
    "working capital management": "Cash and Working Capital Management",
    "market entry": "Market/Category Entry",
    "category entry": "Market/Category Entry",
    "interim management": "Interim Management",
    "commercial excellence": "Commercial Excellence",
    "restructuring": "Restructuring Advisory (Creditor - Debtor - Investor)",
    "restructuring advisory": "Restructuring Advisory (Creditor - Debtor - Investor)",
    "insolvency": "Insolvency Resolution Process",
    "insolvency resolution": "Insolvency Resolution Process",
    "digital": "Digital",
}

_SOLUTION_RE = re.compile(
    r"\b({keys})\b".format(
        keys="|".join(
            re.escape(k) for k in sorted(SOLUTION_ALIASES, key=len, reverse=True)
        )
    ),
    re.IGNORECASE,
)

# ── Client type patterns ────────────────────────────────────────────────────
CLIENT_TYPE_ALIASES: dict[str, str] = {
    "corporate": "Corporate",
    "corporates": "Corporate",
    "private equity": "Investor - PE",
    "pe fund": "Investor - PE",
    "pe investor": "Investor - PE",
    "bank": "Bank",
    "banks": "Bank",
    "distressed fund": "Investor - Distressed fund",
    "venture capital": "Investor - VC",
    "vc fund": "Investor - VC",
    "vc investor": "Investor - VC",
}

_CLIENT_TYPE_RE = re.compile(
    r"\b({keys})\b".format(
        keys="|".join(
            re.escape(k) for k in sorted(CLIENT_TYPE_ALIASES, key=len, reverse=True)
        )
    ),
    re.IGNORECASE,
)

# ── Year pattern ────────────────────────────────────────────────────────────
_YEAR_RE = re.compile(r"\b(202[2-9]|203[0-9])\b")


@dataclass
class ExtractionResult:
    filters: SearchFilters = field(default_factory=SearchFilters)
    semantic_query: str = ""
    inferred_doc_type: str | None = None
    inferred_industry: str | None = None
    inferred_solution: str | None = None


def extract_filters(query: str, manual_filters: SearchFilters | None = None) -> ExtractionResult:
    """Extract implicit filters from the expanded query via pattern matching.

    Returns an ExtractionResult with extracted filters and the residual
    semantic query (parts not consumed by filter extraction).
    """
    if not query or not query.strip():
        return ExtractionResult(
            filters=manual_filters or SearchFilters(),
            semantic_query="",
        )

    result = ExtractionResult(semantic_query=query)
    remaining = query

    bio_score = len(BIO_SIGNALS.findall(remaining))
    kl_score = len(KL_SIGNALS.findall(remaining))

    inferred_doc_type: str | None = None
    if bio_score > 0 and kl_score == 0:
        inferred_doc_type = "bio"
    elif kl_score > 0 and bio_score == 0:
        inferred_doc_type = "knowledge_library"

    designation_match = _DESIGNATION_RE.search(remaining)
    designation_val: str | None = None
    if designation_match:
        designation_val = DESIGNATION_MAP.get(designation_match.group(0).lower())
        if designation_val:
            inferred_doc_type = inferred_doc_type or "bio"

    kt_match = _KT_RE.search(remaining)
    kt_val: str | None = None
    if kt_match:
        kt_val = KNOWLEDGE_TYPE_MAP.get(kt_match.group(0).lower())
        if kt_val:
            inferred_doc_type = inferred_doc_type or "knowledge_library"

    if manual_filters and manual_filters.document_type:
        inferred_doc_type = manual_filters.document_type

    industry_match = _INDUSTRY_RE.search(remaining)
    industry_val: str | None = None
    if industry_match:
        industry_val = INDUSTRY_ALIASES.get(industry_match.group(0).lower())

    solution_match = _SOLUTION_RE.search(remaining)
    solution_val: str | None = None
    if solution_match:
        solution_val = SOLUTION_ALIASES.get(solution_match.group(0).lower())

    client_type_match = _CLIENT_TYPE_RE.search(remaining)
    client_type_val: str | None = None
    if client_type_match:
        client_type_val = CLIENT_TYPE_ALIASES.get(client_type_match.group(0).lower())

    year_match = _YEAR_RE.search(remaining)
    year_val: str | None = None
    if year_match:
        year_val = year_match.group(0)

    extracted = SearchFilters()

    if inferred_doc_type:
        extracted.document_type = inferred_doc_type
    if designation_val and inferred_doc_type == "bio":
        extracted.designation = designation_val
    if kt_val and inferred_doc_type != "bio":
        extracted.knowledge_type = kt_val
    if industry_val and inferred_doc_type != "bio":
        extracted.industry = industry_val
    if solution_val and inferred_doc_type != "bio":
        extracted.solution = solution_val
    if client_type_val and inferred_doc_type != "bio":
        extracted.client_type = client_type_val
    if year_val:
        extracted.year = year_val

    if manual_filters:
        for fld in [
            "document_type", "designation", "knowledge_type",
            "knowledge_sub_type", "industry", "sub_industry",
            "solution", "client_name", "client_type", "year",
        ]:
            manual_val = getattr(manual_filters, fld, None)
            if manual_val:
                setattr(extracted, fld, manual_val)

    result.filters = extracted
    result.inferred_doc_type = inferred_doc_type
    result.inferred_industry = industry_val
    result.inferred_solution = solution_val
    result.semantic_query = remaining

    return result
