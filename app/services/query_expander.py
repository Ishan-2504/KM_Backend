"""Zero-latency query expansion via acronym dictionary and context enrichment.

Resolves known abbreviations and appends inferred context so the query
embedding lands closer to relevant document vectors in the embedding space.
"""

import re

ACRONYM_MAP: dict[str, str] = {
    "CDD": "commercial due diligence",
    "ODD": "operational due diligence and assessments",
    "FDD": "financial due diligence",
    "IDD": "integrated due diligence",
    "MD": "managing director",
    "MDs": "managing directors",
    "BD": "business development",
    "PE": "private equity",
    "VC": "venture capital",
    "M&A": "mergers and acquisitions",
    "PMI": "post merger integration",
    "BPA": "business plan assessment",
    "SCM": "supply chain management",
    "WCM": "working capital management",
    "FP&A": "financial planning and analysis",
    "BPO": "business process optimization",
    "IRP": "insolvency resolution process",
    "POV": "point of view",
    "R&B": "research and benchmarks",
    "SOP": "standard operating procedure",
    "KPI": "key performance indicator",
    "EBITDA": "earnings before interest taxes depreciation amortization",
    "P&L": "profit and loss",
    "ROI": "return on investment",
    "NPS": "net promoter score",
    "TAM": "total addressable market",
    "GTM": "go to market",
}

_WORD_BOUNDARY = re.compile(r"(?<!\w)({keys})(?!\w)".format(
    keys="|".join(re.escape(k) for k in sorted(ACRONYM_MAP, key=len, reverse=True))
))


def expand_query(query: str) -> str:
    """Replace known acronyms with their full forms.

    Case-insensitive match but preserves surrounding text.
    Returns the expanded query string.
    """
    if not query:
        return query

    def _replace(match: re.Match) -> str:
        token = match.group(0)
        return ACRONYM_MAP.get(token) or ACRONYM_MAP.get(token.upper(), token)

    expanded = _WORD_BOUNDARY.sub(_replace, query)

    ci_pattern = re.compile(
        r"(?<!\w)({keys})(?!\w)".format(
            keys="|".join(re.escape(k) for k in sorted(ACRONYM_MAP, key=len, reverse=True))
        ),
        re.IGNORECASE,
    )

    def _replace_ci(match: re.Match) -> str:
        token = match.group(0)
        for key, val in ACRONYM_MAP.items():
            if token.upper() == key.upper():
                return val
        return token

    expanded = ci_pattern.sub(_replace_ci, query)

    return expanded
