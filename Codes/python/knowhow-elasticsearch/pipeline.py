"""Legacy entrypoint. Use extract.py and index.py directly instead."""
import sys


def main():
    print("This file is deprecated.")
    print("  python extract.py   — run/resume LLM enrichment")
    print("  python index.py     — index to Elasticsearch")
    sys.exit(1)


if __name__ == "__main__":
    main()
