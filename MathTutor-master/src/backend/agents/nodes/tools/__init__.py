from backend.agents.utils.helper import _get_secret


# BGE Chinese embedding model
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"

EMBED_INPUT_TYPE_DOC   = "document"
EMBED_INPUT_TYPE_QUERY = "query"
EMBED_DIM              = 1024

# Minimum cosine similarity to include a chunk in results.
MIN_SCORE = 0.30

# Number of chunks to retrieve per query (final after fusion)
TOP_K = 5


__all__ = [
    "_get_secret",
    "EMBED_MODEL",
    "EMBED_INPUT_TYPE_DOC",
    "EMBED_INPUT_TYPE_QUERY",
    "EMBED_DIM",
    "MIN_SCORE",
    "TOP_K",
]
