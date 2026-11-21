"""Local web frontend for the SpeechTranslator MAF agent (Arabic / Chinese -> English).

    .venv\\Scripts\\python -m frontend.app            (then open http://127.0.0.1:8000)

One agent and one conversation session (short-term memory) are shared by the page; requests are
processed one at a time so turns don't interleave in the session. "Clear memory" starts a new session.

Privacy: uploaded audio is written to a temporary folder and deleted when the request ends; input and
output text are not logged or stored.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import config  # noqa: E402
from agent import SCRIPT_RE, audio_message, detect_text_language, text_message  # noqa: E402

log = logging.getLogger("frontend")
STATIC = Path(__file__).resolve().parent / "static"

MAX_TEXT_CHARS = config.MAX_TEXT_CHARS
MAX_UPLOAD_MB = config.MAX_AUDIO_BYTES // (1024 * 1024)

_FIELD_RE = re.compile(
    r"Detected language:\s*(?P<lang>.+?)\s*\n\s*Original:\s*(?P<orig>.*?)\s*\n\s*English:\s*(?P<en>.*)",
    re.S | re.I,
)

Lang = Literal["auto", "ar", "zh"]


def parse_reply(reply: str) -> dict:
    """Split the agent's fixed-format reply into fields. Raises ValueError for refusals/errors."""
    reply = reply.strip()
    if reply.lower().startswith(("unsupported language", "error:")):
        raise ValueError(reply)
    m = _FIELD_RE.search(reply)
    if not m:  # model drifted from the format: show the raw reply rather than failing
        return {"language": None, "original": "", "english": reply}
    return {"language": m["lang"].strip(), "original": m["orig"].strip(), "english": m["en"].strip()}


class Service:
    """Owns the agent and the conversation session."""

    def __init__(self) -> None:
        from agent import create_agent

        self.agent = create_agent()
        self.session = self.agent.create_session()
        self.lock = asyncio.Lock()

    def reset(self) -> None:
        self.session = self.agent.create_session()

    async def _ask(self, message: str) -> dict:
        from agent import translate

        async with self.lock:
            reply = await translate(self.agent, self.session, message)
        return parse_reply(reply)

    async def translate_text(self, text: str, lang: str) -> dict:
        t0 = time.perf_counter()
        out = await self._ask(text_message(text, lang))
        out["original"] = text  # the code already knows the source text exactly
        return {"mode": "text", **out, "elapsed_s": round(time.perf_counter() - t0, 1), "audio_s": None}

    async def translate_audio(self, src: Path, lang: str) -> dict:
        t0 = time.perf_counter()
        out = await self._ask(audio_message(src, lang))
        return {"mode": "audio", **out, "elapsed_s": round(time.perf_counter() - t0, 1),
                "audio_s": audio_duration(src)}


def audio_duration(path: Path) -> float | None:
    try:
        from faster_whisper.audio import decode_audio

        return round(len(decode_audio(str(path))) / 16000, 1)
    except Exception:  # duration is informational only
        return None


# ------------------------------------------------------------------ app
class TextIn(BaseModel):
    text: str
    language: Lang = "auto"


def create_app(service_factory=None) -> FastAPI:
    """service_factory() -> Service; tests can pass a fake so no API calls are made."""
    state: dict[str, Any] = {"service": None, "error": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            if service_factory is None:
                from audio import load_whisper

                await asyncio.to_thread(load_whisper)  # first run downloads the Whisper model
            state["service"] = (service_factory or Service)()
        except Exception as e:  # keep serving so the page can show the problem
            log.exception("Agent failed to load")
            state["error"] = f"{type(e).__name__}: {e}"
        print("Frontend ready: http://127.0.0.1:8000", flush=True)
        yield

    app = FastAPI(title="Arabic & Chinese to English Translator", lifespan=lifespan, docs_url=None, redoc_url=None)

    def service() -> Service:
        if state["service"] is None:
            raise HTTPException(503, f"Agent not loaded: {state['error'] or 'still starting'}")
        return state["service"]

    async def guarded(coro):
        try:
            return await coro
        except ValueError as e:  # agent refusal / tool error, shown to the user as-is
            raise HTTPException(400, str(e))
        except Exception as e:
            log.exception("Translation failed")
            msg = str(e)
            if "connect" in msg.lower():
                msg = f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. Is Ollama running?"
            elif "not found" in msg.lower() and config.CHAT_MODEL in msg:
                msg = f"Ollama model '{config.CHAT_MODEL}' is missing. Run: ollama pull {config.CHAT_MODEL}"
            raise HTTPException(502, f"Translation failed: {msg}")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"ready": state["service"] is not None, "error": state["error"],
                "agent": "SpeechTranslator (Microsoft Agent Framework)",
                "asr_model": f"faster-whisper {config.WHISPER_MODEL}", "mt_model": f"{config.CHAT_MODEL} (Ollama)",
                "max_text_chars": MAX_TEXT_CHARS, "max_upload_mb": MAX_UPLOAD_MB}

    @app.post("/api/reset")
    def reset():
        service().reset()
        return {"ok": True}

    @app.post("/api/translate/text")
    async def translate_text(body: TextIn):
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "Please enter some Arabic or Chinese text.")
        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(400, f"Text is too long (max {MAX_TEXT_CHARS} characters).")
        if body.language == "auto":
            lang = detect_text_language(text)
            if lang is None:
                raise HTTPException(400, "No Arabic or Chinese characters found in the text.")
        elif SCRIPT_RE[body.language].search(text):
            lang = body.language
        else:
            name = config.SUPPORTED_LANGUAGES[body.language]
            raise HTTPException(400, f"No {name} characters found. Change the language or the text.")
        return await guarded(service().translate_text(text, lang))

    @app.post("/api/translate/audio")
    async def translate_audio(file: UploadFile = File(...), language: Lang = Form("auto")):
        svc = service()
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in config.ALLOWED_AUDIO_EXTENSIONS:
            allowed = ", ".join(sorted(e.lstrip(".") for e in config.ALLOWED_AUDIO_EXTENSIONS))
            raise HTTPException(400, f"Unsupported audio format. Use one of: {allowed}.")
        with tempfile.TemporaryDirectory(prefix="st_web_") as tmp:
            src = Path(tmp) / f"upload{suffix}"
            size, limit = 0, config.MAX_AUDIO_BYTES
            with src.open("wb") as f:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
                    f.write(chunk)
            if size == 0:
                raise HTTPException(400, "The uploaded file is empty.")
            return await guarded(svc.translate_audio(src, language))

    return app


def main() -> None:
    import uvicorn

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Local web frontend for the SpeechTranslator agent")
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 unless you add authentication")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
