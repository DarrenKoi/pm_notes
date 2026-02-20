"""Topic 06: templates, aliases, rollover, and lifecycle policy."""

import _path_setup  # noqa: F401

from opensearch_handler import (
    bootstrap_aliases,
    create_client,
    delete_index,
    detect_cluster_flavor,
    index_exists,
    load_config,
    put_index_template,
    put_lifecycle_policy,
    rollover_alias,
)

TEMPLATE_NAME = "example-topic-template"
FIRST_INDEX = "example-topic-logs-000001"
WRITE_ALIAS = "example-topic-logs-write"
READ_ALIAS = "example-topic-logs-read"
POLICY_NAME = "example-topic-retention"


def main() -> None:
    client = create_client(config=load_config())

    flavor = detect_cluster_flavor(client)
    print(f"Detected cluster flavor: {flavor}")

    put_index_template(
        client,
        TEMPLATE_NAME,
        {
            "index_patterns": ["example-topic-logs-*"],
            "template": {
                "settings": {
                    "index.number_of_shards": 1,
                    "index.number_of_replicas": 0,
                },
                "mappings": {
                    "properties": {
                        "message": {"type": "text"},
                        "level": {"type": "keyword"},
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
        conditions={"max_docs": 1, "max_age": "1d"},
        dry_run=True,
    )
    print("Rollover dry-run:", rollover)

    try:
        put_lifecycle_policy(
            client,
            POLICY_NAME,
            {
                "policy": {
                    "description": "Example retention policy",
                    "default_state": "hot",
                    "states": [
                        {
                            "name": "hot",
                            "actions": [],
                            "transitions": [{"state_name": "delete", "conditions": {"min_index_age": "7d"}}],
                        },
                        {"name": "delete", "actions": [{"delete": {}}], "transitions": []},
                    ],
                    "ism_template": [{"index_patterns": ["example-topic-logs-*"], "priority": 1}],
                }
            },
            flavor="auto",
        )
        print(f"Lifecycle policy upserted: {POLICY_NAME}")
    except Exception as exc:
        print("Lifecycle policy step skipped.")
        print(f"Error: {exc}")

    delete_index(client, FIRST_INDEX)
    print("Cleanup complete (template/policy are intentionally left in place).")


if __name__ == "__main__":
    main()
