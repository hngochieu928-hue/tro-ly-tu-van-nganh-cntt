# ============================================================
# CONFIG.PY — TỐI ƯU CHÍNH XÁC + TỐC ĐỘ + CHI TIẾT
# ============================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "vector_db")


# ============================================================
# MODEL EMBEDDING (LOCAL)
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================
# CHROMADB
# ============================================================

COLLECTION_NAME = "kien_thuc_cntt"


# ============================================================
# CHUNK
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# TÌM KIẾM — HYBRID + RERANK
# ============================================================

USE_HYBRID = True
USE_RERANKER = False

CANDIDATES_K = 20
FINAL_TOP_K = 6

TOP_K = FINAL_TOP_K

RRF_K = 60
MAX_DISTANCE = 1.3


# ============================================================
# RERANKER
# ============================================================

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LENGTH = 512


# ============================================================
# CONTEXT
# ============================================================

MAX_CONTEXT_CHARS = 5000


# ============================================================
# CACHE
# ============================================================

EMBEDDING_CACHE_SIZE = 512


# ============================================================
# OLLAMA LOCAL
# ============================================================

LLM_MODEL = "qwen2.5:3b"
OLLAMA_KEEP_ALIVE = "60m"
OLLAMA_NUM_THREAD = None


# ============================================================
# THAM SỐ SINH CÂU TRẢ LỜI
# ============================================================

LLM_TEMPERATURE = 0.05
LLM_NUM_PREDICT = 900
LLM_NUM_CTX = 6144


# ============================================================
# LLM PROVIDER
# ============================================================

DEFAULT_PROVIDER = "gemini"

# ⚡ Gemini: chỉ dùng model hỗ trợ SDK tương ứng
#   - gemini-1.5-*  → SDK cũ (google-generativeai)
#   - gemini-2.0-*  → SDK mới (google-genai)
PROVIDER_MODELS = {
    "ollama":   ["qwen2.5:3b", "qwen2.5:7b", "llama3.2:3b"],
    "openai":   ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
    "gemini":   ["gemini-3.0-flash",        # mới nhất, khuyến nghị
                 "gemini-2.5-flash",
                 "gemini-2.0-flash",
                 "gemini-1.5-flash"],       # cũ, có thể đã bị Google ngừng hỗ trợ
    "claude":   ["claude-3-5-sonnet-latest",
                 "claude-3-5-haiku-latest",
                 "claude-3-opus-latest"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "groq":     ["llama-3.3-70b-versatile",
                 "llama-3.1-8b-instant",
                 "mixtral-8x7b-32768"],
}

PROVIDER_LABELS = {
    "ollama":   "🖥️  Ollama local",
    "openai":   "🟢 OpenAI · ChatGPT",
    "gemini":   "🔷 Google Gemini",
    "claude":   "🟣 Anthropic Claude",
    "deepseek": "🐋 DeepSeek",
    "groq":     "⚡ Groq (Llama 3.3)",
}

PROVIDER_KEY_LABELS = {
    "openai":   "OPENAI_API_KEY",
    "gemini":   "GEMINI_API_KEY",
    "claude":   "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq":     "GROQ_API_KEY",
}

OPENAI_BASE_URL   = None
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GROQ_BASE_URL     = "https://api.groq.com/openai/v1"