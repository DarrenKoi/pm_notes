"""Glossary pipeline configuration."""

import os

# Elasticsearch
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "glossary")

# Local LLM (OpenAI-compatible API)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "default")

# Ingestion
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
