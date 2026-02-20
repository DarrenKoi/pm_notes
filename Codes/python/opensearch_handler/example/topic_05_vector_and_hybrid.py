"""Topic 05: k-NN vector search and hybrid retrieval."""

import _path_setup  # noqa: F401

from opensearch_handler import (
    bulk_index,
    create_client,
    create_index,
    delete_index,
    hybrid_search,
    index_exists,
    knn_search,
    load_config,
)

INDEX_NAME = "example-topic-vector"


def _print_hits(label: str, response: dict) -> None:
    titles = [hit["_source"]["title"] for hit in response["hits"]["hits"]]
    print(f"{label}: {titles}")


def main() -> None:
    client = create_client(config=load_config())

    try:
        if index_exists(client, INDEX_NAME):
            delete_index(client, INDEX_NAME)

        create_index(
            client,
            INDEX_NAME,
            mappings={
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 3,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                }
            },
            settings={"index.knn": True},
        )

        bulk_index(
            client,
            INDEX_NAME,
            docs=[
                {
                    "id": "1",
                    "title": "OpenSearch Basics",
                    "content": "Search and indexing fundamentals.",
                    "embedding": [0.1, 0.2, 0.3],
                },
                {
                    "id": "2",
                    "title": "Vector Retrieval",
                    "content": "Semantic retrieval with embeddings.",
                    "embedding": [0.4, 0.5, 0.6],
                },
                {
                    "id": "3",
                    "title": "Hybrid Search",
                    "content": "Blend lexical relevance with vectors.",
                    "embedding": [0.39, 0.49, 0.58],
                },
            ],
            id_field="id",
            refresh=True,
        )

        _print_hits(
            "knn_search",
            knn_search(client, INDEX_NAME, "embedding", [0.4, 0.5, 0.6], k=2),
        )
        _print_hits(
            "hybrid_search",
            hybrid_search(
                client,
                INDEX_NAME,
                query="vector retrieval",
                text_field="content",
                vector_field="embedding",
                vector=[0.4, 0.5, 0.6],
                k=2,
            ),
        )
    except Exception as exc:
        print("Vector and hybrid search example failed.")
        print("Make sure your OpenSearch cluster has k-NN enabled.")
        print(f"Error: {exc}")
    finally:
        if index_exists(client, INDEX_NAME):
            delete_index(client, INDEX_NAME)
            print("Cleanup complete.")


if __name__ == "__main__":
    main()
