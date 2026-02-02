LLM_URL = "http://common.llm.skhynix.com/v1"
LLM_MODEL = "gpt-oss-20b"

ES_HOST = "http://localhost:9200"
ES_INDEX = "knowhow"

LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 2  # seconds

BATCH_SIZE = 500  # items per batch for LLM processing + ES indexing
ES_BULK_CHUNK = 500  # docs per ES bulk request
