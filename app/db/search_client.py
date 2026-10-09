"""Azure AI Search client and index management."""

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from app.config import (
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX_NAME,
)

VECTOR_DIMENSIONS = 3072  # text-embedding-3-large default

_credential: AzureKeyCredential | None = None
_search_client: SearchClient | None = None
_index_client: SearchIndexClient | None = None


def _get_credential() -> AzureKeyCredential:
    global _credential
    if _credential is None:
        _credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    return _credential


def get_index_client() -> SearchIndexClient:
    global _index_client
    if _index_client is None:
        _index_client = SearchIndexClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=_get_credential(),
        )
    return _index_client


def get_search_client() -> SearchClient:
    global _search_client
    if _search_client is None:
        _search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=_get_credential(),
        )
    return _search_client


def _build_index_schema() -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="document_name", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="summary", type=SearchFieldDataType.String),
        SearchableField(name="composite_text", type=SearchFieldDataType.String),
        SearchableField(name="file_text", type=SearchFieldDataType.String),
        # Filterable + facetable metadata
        SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="designation", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="knowledge_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="knowledge_sub_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="industry", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="sub_industry", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="solution", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="client_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="client_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="year", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="link", type=SearchFieldDataType.String),
        SimpleField(name="taggings", type=SearchFieldDataType.String),
        # Vector field
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
    )

    semantic_config = SemanticConfiguration(
        name="semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="document_name"),
            content_fields=[SemanticField(field_name="summary")],
        ),
    )

    semantic_search = SemanticSearch(
        default_configuration_name="semantic-config",
        configurations=[semantic_config],
    )

    return SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def ensure_index() -> None:
    """Create or update the search index."""
    index_client = get_index_client()
    index = _build_index_schema()
    index_client.create_or_update_index(index)
    print(f"Index '{AZURE_SEARCH_INDEX_NAME}' is ready.")
