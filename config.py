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
