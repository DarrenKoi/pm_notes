"""Topic 03: document CRUD and bulk indexing."""

import _path_setup  # noqa: F401

from opensearch_handler import (
    bulk_index,
    count_documents,
    create_client,
    create_index,
    delete_document,
    delete_index,
    get_document,
    index_document,
    index_exists,
    load_config,
    refresh_index,
    update_document,
    upsert_document,
)

INDEX_NAME = "example-topic-docs"


def main() -> None:
    client = create_client(config=load_config())

    if index_exists(client, INDEX_NAME):
        delete_index(client, INDEX_NAME)

    create_index(
        client,
        INDEX_NAME,
        mappings={
            "properties": {
                "title": {"type": "text"},
                "category": {"type": "keyword"},
                "views": {"type": "integer"},
            }
        },
    )

    index_document(
        client,
        INDEX_NAME,
        doc={"title": "OpenSearch Basics", "category": "tutorial", "views": 10},
        doc_id="1",
        refresh="wait_for",
    )

    success, errors = bulk_index(
        client,
        INDEX_NAME,
        docs=[
            {"id": "2", "title": "Bulk Indexing", "category": "guide", "views": 5},
            {"id": "3", "title": "Index Tuning", "category": "operations", "views": 8},
        ],
        id_field="id",
        refresh=True,
    )
    print(f"Bulk indexed: success={success}, errors={len(errors)}")

    doc = get_document(client, INDEX_NAME, "1")
    print("Fetched doc 1:", doc["_source"])

    update_document(client, INDEX_NAME, "1", {"views": 11}, refresh="wait_for")
    upsert_document(
        client,
        INDEX_NAME,
        "4",
        {"title": "Upserted Doc", "category": "tutorial", "views": 1},
        refresh="wait_for",
    )

    delete_document(client, INDEX_NAME, "2", refresh="wait_for")
    refresh_index(client, INDEX_NAME)

    total = count_documents(client, INDEX_NAME)["count"]
    print(f"Document count after CRUD operations: {total}")

    delete_index(client, INDEX_NAME)
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
