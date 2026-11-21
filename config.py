"""Central settings loaded from .env (all optional; defaults run fully locally)."""

import os

from dotenv import load_dotenv

load_dotenv()

# Translation / agent LLM: local Ollama via its OpenAI-compatible endpoint (no API key needed)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))  # upper bound for one translation

# Speech-to-text: local faster-whisper
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # tiny | base | small | medium | large-v3
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # cpu | cuda
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 on CPU, float16 on GPU

# Input guardrails
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".mp4", ".mpeg", ".mpga", ".ogg", ".flac"}
SUPPORTED_LANGUAGES = {"ar": "Arabic", "zh": "Chinese"}  # ISO-639-1 codes used by Whisper
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_MIC_SECONDS = 60
MAX_TEXT_CHARS = 5000

# Document guardrails (PDF / DOCX / images)
MAX_DOC_BYTES = int(os.getenv("MAX_DOC_MB", "10")) * 1024 * 1024
MAX_DOC_PAGES = int(os.getenv("MAX_DOC_PAGES", "10"))
MAX_DOC_CHARS = int(os.getenv("MAX_DOC_CHARS", "20000"))  # ~10 pages of text
# Text per agent call. Smaller = results appear sooner (a 380-char Chinese chunk took ~2.5 min
# on CPU, ~250 chars gives an update about every minute); the cached prompt keeps overhead low.
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "250"))
OCR_DPI = 200  # render resolution for scanned PDF pages
# History budget (source + English characters) for one document's session. Ollama's default
# context is 4096 tokens and Chinese costs ~1 token per character, so: instructions (~650) +
# history (<=2000) + current chunk (<=400) + reply stays under the limit. Beyond the budget a
# fresh session starts (the cached instructions prefix is reused, so this costs nothing).
DOC_SESSION_CHAR_BUDGET = int(os.getenv("DOC_SESSION_CHAR_BUDGET", "2000"))
