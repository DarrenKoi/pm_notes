"""Document CRUD and bulk operations."""

from __future__ import annotations

from typing import Any, Optional, Sequence

try:
    from opensearchpy import helpers
except ModuleNotFoundError:  # pragma: no cover
    from elasticsearch import helpers


def index_document(
    client: Any,
    index: str,
    doc: dict[str, Any],
    doc_id: Optional[str] = None,
    refresh: Optional[str] = None,
) -> dict:
    """Index (insert or replace) a single document."""
    kwargs: dict[str, Any] = {"index": index, "body": doc}
    if doc_id is not None:
        kwargs["id"] = doc_id
    if refresh is not None:
        kwargs["refresh"] = refresh
    return client.index(**kwargs)


def get_document(client: Any, index: str, doc_id: str) -> dict:
    """Retrieve a document by ID."""
    return client.get(index=index, id=doc_id)


def update_document(
    client: Any,
    index: str,
    doc_id: str,
    doc: dict[str, Any],
    refresh: Optional[str] = None,
) -> dict:
    """Partially update an existing document."""
    kwargs: dict[str, Any] = {"index": index, "id": doc_id, "body": {"doc": doc}}
    if refresh is not None:
        kwargs["refresh"] = refresh
    return client.update(**kwargs)


def upsert_document(
    client: Any,
    index: str,
    doc_id: str,
    doc: dict[str, Any],
    refresh: Optional[str] = None,
) -> dict:
    """Create or update a document via doc_as_upsert."""
    kwargs: dict[str, Any] = {
        "index": index,
        "id": doc_id,
        "body": {"doc": doc, "doc_as_upsert": True},
    }
    if refresh is not None:
        kwargs["refresh"] = refresh
    return client.update(**kwargs)


def delete_document(
    client: Any,
    index: str,
    doc_id: str,
    refresh: Optional[str] = None,
) -> dict:
    """Delete a document by ID."""
    kwargs: dict[str, Any] = {"index": index, "id": doc_id}
    if refresh is not None:
        kwargs["refresh"] = refresh
    return client.delete(**kwargs)


def bulk_index(
    client: Any,
    index: str,
    docs: Sequence[dict[str, Any]],
    id_field: Optional[str] = None,
    chunk_size: int = 500,
    refresh: bool = False,
    raise_on_error: bool = False,
) -> tuple[int, list]:
    """Bulk-index a sequence of documents.

    Args:
        client: OpenSearch client.
        index: Target index name.
        docs: Documents to index.
        id_field: If set, uses ``doc[id_field]`` as the document ``_id``.
        chunk_size: Number of docs per bulk request (default 500).
        refresh: If True, refreshes index after indexing.
        raise_on_error: Whether to raise exceptions for item-level errors.

    Returns:
        A tuple of ``(success_count, error_list)``.
    """
    def iter_actions():
        for doc in docs:
            action: dict[str, Any] = {"_index": index, "_source": doc}
            if id_field and id_field in doc:
                action["_id"] = doc[id_field]
            yield action

    return helpers.bulk(
        client,
        iter_actions(),
        chunk_size=chunk_size,
        refresh=refresh,
        raise_on_error=raise_on_error,
    )
