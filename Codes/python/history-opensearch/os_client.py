"""OpenSearch client — 3개 인덱스(chat-messages, chat-sessions, user-long-memory) 관리."""

import logging
from datetime import datetime

from opensearchpy import OpenSearch

import config
from models import Message, Session, UserFact

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Index mappings
# ──────────────────────────────────────────────

MESSAGES_MAPPING = {
    "settings": {"index.knn": True},
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
            "session_id": {"type": "keyword"},
            "role": {"type": "keyword"},
            "content": {"type": "text", "analyzer": "standard"},
            "embedding": {
                "type": "knn_vector",
                "dimension": config.EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                },
            },
            "timestamp": {"type": "date"},
        }
    },
}

SESSIONS_MAPPING = {
    "settings": {"index.knn": True},
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
            "session_id": {"type": "keyword"},
            "summary": {"type": "text", "analyzer": "standard"},
            "topics": {"type": "keyword"},
            "message_count": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": config.EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                },
            },
            "start_time": {"type": "date"},
            "end_time": {"type": "date"},
        }
    },
}

LONG_MEMORY_MAPPING = {
    "settings": {"index.knn": True},
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
            "fact": {"type": "text", "analyzer": "standard"},
            "category": {"type": "keyword"},
            "importance": {"type": "float"},
            "embedding": {
                "type": "knn_vector",
                "dimension": config.EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                },
            },
            "created_at": {"type": "date"},
            "last_accessed": {"type": "date"},
        }
    },
}

INDEX_CONFIGS = {
    config.IDX_MESSAGES: MESSAGES_MAPPING,
    config.IDX_SESSIONS: SESSIONS_MAPPING,
    config.IDX_LONG_MEMORY: LONG_MEMORY_MAPPING,
}


# ──────────────────────────────────────────────
# Client helpers
# ──────────────────────────────────────────────


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[config.OS_HOST],
        http_auth=(config.OS_USER, config.OS_PASSWORD),
        verify_certs=config.OS_VERIFY_CERTS,
        ssl_show_warn=False,
    )


def ensure_indices(client: OpenSearch) -> None:
    for index_name, mapping in INDEX_CONFIGS.items():
        if not client.indices.exists(index=index_name):
            client.indices.create(index=index_name, body=mapping)
            logger.info("Created index: %s", index_name)
        else:
            logger.info("Index already exists: %s", index_name)


# ──────────────────────────────────────────────
# CRUD — Messages (단기 메모리)
# ──────────────────────────────────────────────


def index_message(client: OpenSearch, msg: Message) -> str:
    resp = client.index(
        index=config.IDX_MESSAGES,
        body=msg.model_dump(mode="json"),
        refresh="wait_for",
    )
    return resp["_id"]


def get_recent_messages(
    client: OpenSearch, user_id: str, session_id: str, limit: int = 20
) -> list[dict]:
    body = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"user_id": user_id}},
                    {"term": {"session_id": session_id}},
                ]
            }
        },
        "sort": [{"timestamp": {"order": "asc"}}],
        "size": limit,
    }
    resp = client.search(index=config.IDX_MESSAGES, body=body)
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def delete_old_messages(
    client: OpenSearch, user_id: str, session_id: str, before: datetime
) -> int:
    body = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"user_id": user_id}},
                    {"term": {"session_id": session_id}},
                    {"range": {"timestamp": {"lt": before.isoformat()}}},
                ]
            }
        }
    }
    resp = client.delete_by_query(index=config.IDX_MESSAGES, body=body)
    return resp.get("deleted", 0)


# ──────────────────────────────────────────────
# CRUD — Sessions (중기 메모리)
# ──────────────────────────────────────────────


def index_session(client: OpenSearch, session: Session) -> str:
    resp = client.index(
        index=config.IDX_SESSIONS,
        id=session.session_id,
        body=session.model_dump(mode="json"),
        refresh="wait_for",
    )
    return resp["_id"]


def get_recent_sessions(
    client: OpenSearch, user_id: str, limit: int = 3
) -> list[dict]:
    body = {
        "query": {"term": {"user_id": user_id}},
        "sort": [{"end_time": {"order": "desc"}}],
        "size": limit,
    }
    resp = client.search(index=config.IDX_SESSIONS, body=body)
    return [hit["_source"] for hit in resp["hits"]["hits"]]


# ──────────────────────────────────────────────
# CRUD — Long Memory (장기 메모리)
# ──────────────────────────────────────────────


def index_fact(client: OpenSearch, fact: UserFact) -> str:
    resp = client.index(
        index=config.IDX_LONG_MEMORY,
        body=fact.model_dump(mode="json"),
        refresh="wait_for",
    )
    return resp["_id"]


def search_facts_by_vector(
    client: OpenSearch,
    user_id: str,
    query_vector: list[float],
    top_k: int = 5,
) -> list[dict]:
    body = {
        "size": top_k,
        "query": {
            "bool": {
                "filter": [{"term": {"user_id": user_id}}],
                "must": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": top_k,
                            }
                        }
                    }
                ],
            }
        },
    }
    resp = client.search(index=config.IDX_LONG_MEMORY, body=body)
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def get_all_facts(client: OpenSearch, user_id: str) -> list[dict]:
    body = {
        "query": {"term": {"user_id": user_id}},
        "sort": [{"importance": {"order": "desc"}}],
        "size": 100,
    }
    resp = client.search(index=config.IDX_LONG_MEMORY, body=body)
    return [hit["_source"] for hit in resp["hits"]["hits"]]
