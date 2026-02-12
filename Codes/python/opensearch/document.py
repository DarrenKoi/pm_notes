"""Document CRUD and bulk operations."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from opensearchpy import OpenSearch, helpers


def index_document(
    client: OpenSearch,
    index: str,
    doc: dict[str, Any],
    doc_id: Optional[str] = None,
) -> dict:
    """Index (insert or replace) a single document."""
    kwargs: dict[str, Any] = {"index": index, "body": doc}
    if doc_id is not None:
        kwargs["id"] = doc_id
    return client.index(**kwargs)


def get_document(client: OpenSearch, index: str, doc_id: str) -> dict:
    """Retrieve a document by ID."""
    return client.get(index=index, id=doc_id)


def update_document(
    client: OpenSearch,
    index: str,
    doc_id: str,
    doc: dict[str, Any],
) -> dict:
    """Partially update an existing document."""
    return client.update(index=index, id=doc_id, body={"doc": doc})


def delete_document(client: OpenSearch, index: str, doc_id: str) -> dict:
    """Delete a document by ID."""
    return client.delete(index=index, id=doc_id)


def bulk_index(
    client: OpenSearch,
    index: str,
    docs: Sequence[dict[str, Any]],
    id_field: Optional[str] = None,
    chunk_size: int = 500,
) -> tuple[int, list]:
    """Bulk-index a sequence of documents.

    Args:
        client: OpenSearch client.
        index: Target index name.
        docs: Documents to index.
        id_field: If set, uses ``doc[id_field]`` as the document ``_id``.
        chunk_size: Number of docs per bulk request (default 500,
            matching ``EnvProfile.bulk_chunk``).

    Returns:
        A tuple of ``(success_count, error_list)``.
    """
    actions = []
    for doc in docs:
        action: dict[str, Any] = {
            "_index": index,
            "_source": doc,
        }
        if id_field and id_field in doc:
            action["_id"] = doc[id_field]
        actions.append(action)

    return helpers.bulk(client, actions, chunk_size=chunk_size)
