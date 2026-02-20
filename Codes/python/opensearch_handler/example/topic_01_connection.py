"""Topic 01: create a client and verify cluster connection."""

import _path_setup  # noqa: F401

from opensearch_handler import create_client, load_config


def main() -> None:
    config = load_config()
    client = create_client(config=config)

    print("Connection config")
    print(f"  host={config.host}")
    print(f"  port={config.port}")
    print(f"  use_ssl={config.use_ssl}")
    print(f"  verify_certs={config.verify_certs}")

    print(f"Ping: {client.ping()}")
    info = client.info()
    version = info.get("version", {}).get("number", "unknown")
    distribution = info.get("version", {}).get("distribution", "unknown")
    print(f"Version: {version}")
    print(f"Distribution: {distribution}")


if __name__ == "__main__":
    main()
