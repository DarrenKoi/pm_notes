"""Topic 06: template, alias, and rollover conventions without retention policy."""

from opensearch_handler import (
    bootstrap_aliases,
    create_client,
    delete_index,
    format_rollover_index,
    index_exists,
    load_config,
    put_index_template,
    rollover_alias,
)

TEMPLATE_NAME = "example-topic-template"
INDEX_PREFIX = "example-topic-logs"
FIRST_INDEX = format_rollover_index(INDEX_PREFIX, 1)
WRITE_ALIAS = "example-topic-logs-write"
READ_ALIAS = "example-topic-logs-read"


def main() -> None:
    client = create_client(config=load_config())

    put_index_template(
        client,
        TEMPLATE_NAME,
        {
            "index_patterns": [f"{INDEX_PREFIX}-*"],
            "template": {
                "settings": {
                    "index.number_of_shards": 1,
                    "index.number_of_replicas": 0,
                    "index.refresh_interval": "30s",
                },
                "mappings": {
                    "properties": {
                        "message": {"type": "text"},
                        "level": {"type": "keyword"},
                        "service": {"type": "keyword"},
                        "ts": {"type": "date"},
                    }
                },
            },
            "priority": 100,
        },
    )
    print(f"Template upserted: {TEMPLATE_NAME}")

    if index_exists(client, FIRST_INDEX):
        delete_index(client, FIRST_INDEX)

    bootstrap_aliases(client, FIRST_INDEX, write_alias=WRITE_ALIAS, read_alias=READ_ALIAS)
    print(f"Bootstrap complete: {FIRST_INDEX} with aliases {WRITE_ALIAS}, {READ_ALIAS}")

    rollover = rollover_alias(
        client,
        write_alias=WRITE_ALIAS,
        conditions={"max_docs": 1000000, "max_age": "1d"},
        dry_run=True,
    )
    print("Rollover dry-run:", rollover)

    delete_index(client, FIRST_INDEX)
    print("Cleanup complete (template is intentionally left in place).")


if __name__ == "__main__":
    main()
