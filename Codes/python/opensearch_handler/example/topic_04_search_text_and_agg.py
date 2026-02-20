"""Topic 04: text search patterns and aggregations."""

import _path_setup  # noqa: F401

from opensearch_handler import (
    aggregate,
    bool_search,
    bulk_index,
    count_documents,
    create_client,
    create_index,
    delete_index,
    index_exists,
    load_config,
    match_search,
    multi_match_search,
    term_search,
)

INDEX_NAME = "example-topic-search"


def _print_hits(label: str, response: dict) -> None:
    titles = [hit["_source"]["title"] for hit in response["hits"]["hits"]]
    print(f"{label}: {titles}")


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
                "content": {"type": "text"},
                "category": {"type": "keyword"},
            }
        },
    )

    bulk_index(
        client,
        INDEX_NAME,
        docs=[
            {
                "id": "1",
                "title": "OpenSearch Intro",
                "content": "Learn search fundamentals and mappings.",
                "category": "tutorial",
            },
            {
                "id": "2",
                "title": "Vector Search Guide",
                "content": "k-NN search with embedding vectors.",
                "category": "guide",
            },
            {
                "id": "3",
                "title": "Hybrid Query Design",
                "content": "Combine lexical and vector retrieval.",
                "category": "tutorial",
            },
        ],
        id_field="id",
        refresh=True,
    )

    _print_hits("match_search(content: vector)", match_search(client, INDEX_NAME, "content", "vector"))
    _print_hits("term_search(category: tutorial)", term_search(client, INDEX_NAME, "category", "tutorial"))
    _print_hits(
        "multi_match_search(title/content: search)",
        multi_match_search(client, INDEX_NAME, "search", ["title", "content"]),
    )
    _print_hits(
        "bool_search(must+filter)",
        bool_search(
            client,
            INDEX_NAME,
            must=[{"match": {"content": "vector"}}],
            filter=[{"term": {"category": "tutorial"}}],
        ),
    )

    agg = aggregate(
        client,
        INDEX_NAME,
        agg_body={"by_category": {"terms": {"field": "category"}}},
    )
    print("Aggregation buckets:", agg["aggregations"]["by_category"]["buckets"])
    print("Total docs:", count_documents(client, INDEX_NAME)["count"])

    delete_index(client, INDEX_NAME)
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
