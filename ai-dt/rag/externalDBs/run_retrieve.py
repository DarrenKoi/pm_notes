#!/usr/bin/env python3
"""CLI: Query glossary from Elasticsearch and output JSON."""

import argparse
import json

from retrieve.es_client import lookup_terms
from retrieve.glossary_retriever import (
    retrieve_glossary,
    expand_query,
    build_glossary_context,
)


def main():
    parser = argparse.ArgumentParser(description="Query glossary from Elasticsearch")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum ES score threshold")
    parser.add_argument("--output", choices=["json", "context", "expanded"], default="json",
                        help="Output format: json (raw), context (LLM prompt block), expanded (query expansion)")
    args = parser.parse_args()

    if args.output == "json":
        results = retrieve_glossary(args.query, top_k=args.top_k, min_score=args.min_score)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    elif args.output == "context":
        matches = lookup_terms(args.query, top_k=args.top_k, min_score=args.min_score)
        ctx = build_glossary_context(matches)
        print(ctx if ctx else "(no matches)")

    elif args.output == "expanded":
        matches = lookup_terms(args.query, top_k=args.top_k, min_score=args.min_score)
        expanded = expand_query(args.query, matches)
        print(expanded)


if __name__ == "__main__":
    main()
