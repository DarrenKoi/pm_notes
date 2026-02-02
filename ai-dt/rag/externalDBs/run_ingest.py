#!/usr/bin/env python3
"""CLI: Read snippets → LLM extract glossary terms → Index to Elasticsearch."""

import argparse
import sys

from ingest.reader import read_snippets
from ingest.extractor import extract_glossary_batch
from ingest.indexer import create_index, ingest_glossary


def main():
    parser = argparse.ArgumentParser(description="Ingest glossary terms into Elasticsearch")
    parser.add_argument("--input", required=True, help="Path to input file or directory")
    parser.add_argument("--format", choices=["csv", "json", "text"], default=None,
                        help="Force input format (auto-detected if omitted)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="LLM batch size (overrides config)")
    parser.add_argument("--recreate-index", action="store_true",
                        help="Delete and recreate ES index before ingestion")
    args = parser.parse_args()

    # 1. Read snippets
    print("[1/3] Reading snippets...")
    snippets = read_snippets(args.input, fmt=args.format)
    if not snippets:
        print("No snippets found. Exiting.")
        sys.exit(1)

    # 2. Extract glossary terms via LLM
    print("[2/3] Extracting glossary terms via LLM...")
    kwargs = {}
    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
    entries = extract_glossary_batch(snippets, **kwargs)
    print(f"Extracted {len(entries)} unique terms.")

    # 3. Index to Elasticsearch
    print("[3/3] Indexing to Elasticsearch...")
    create_index(recreate=args.recreate_index)
    ingest_glossary(entries)
    print("Done.")


if __name__ == "__main__":
    main()
