"""Topic 02: index creation, inspection, update, refresh, and cleanup."""

import _path_setup  # noqa: F401

from opensearch_handler import (
    create_client,
    create_index,
    delete_index,
    get_index_mapping,
    get_index_settings,
    index_exists,
    load_config,
    refresh_index,
    update_index_settings,
)

INDEX_NAME = "example-topic-index"


def main() -> None:
    client = create_client(config=load_config())

    if index_exists(client, INDEX_NAME):
        delete_index(client, INDEX_NAME)

    create_index(
        client,
        INDEX_NAME,
        shards=1,
        replicas=0,
        mappings={
            "properties": {
                "title": {"type": "text"},
                "category": {"type": "keyword"},
                "published_at": {"type": "date"},
            }
        },
    )

    settings = get_index_settings(client, INDEX_NAME)
    mapping = get_index_mapping(client, INDEX_NAME)
    print(f"Index created: {INDEX_NAME}")
    print("Settings keys:", list(settings[INDEX_NAME]["settings"]["index"].keys())[:6])
    print("Mapped fields:", list(mapping[INDEX_NAME]["mappings"]["properties"].keys()))

    update_index_settings(client, INDEX_NAME, {"refresh_interval": "1s"})
    refresh_index(client, INDEX_NAME)
    print("Updated refresh_interval to 1s and refreshed index.")

    delete_index(client, INDEX_NAME)
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
