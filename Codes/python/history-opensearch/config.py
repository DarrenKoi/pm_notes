# --- OpenSearch ---
OS_HOST = "https://localhost:9200"
OS_USER = "admin"
OS_PASSWORD = "admin"
OS_VERIFY_CERTS = False

# Index names
IDX_MESSAGES = "chat-messages"
IDX_SESSIONS = "chat-sessions"
IDX_LONG_MEMORY = "user-long-memory"

# --- Embedding (BGE-M3, OpenAI-compatible local API) ---
EMBEDDING_URL = "http://localhost:8000/v1"
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024

# --- LLM (Qwen3 / Kimi2, OpenAI-compatible local API) ---
LLM_URL = "http://localhost:8001/v1"
LLM_MODEL = "qwen3"

# --- Memory settings ---
SHORT_TERM_LIMIT = 20          # max messages kept in short-term
SESSION_SUMMARY_THRESHOLD = 10  # summarize after N messages
LONG_MEMORY_TOP_K = 5          # top-K facts retrieved per query
SESSION_SUMMARY_TOP_K = 3      # recent session summaries to inject
