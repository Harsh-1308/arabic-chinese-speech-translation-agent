"""Local web frontend for the SpeechTranslator MAF agent (Arabic / Chinese -> English).

    .venv\\Scripts\\python -m frontend.app            (then open http://127.0.0.1:8000)

One agent and one conversation session (short-term memory) are shared by the page; requests are
processed one at a time so turns don't interleave in the session. "Clear memory" starts a new session.

Start-up runs in the background (the page loads immediately and shows "Starting..."): load Whisper,
create the agent, warm up the LLM, then keep it loaded in Ollama so no request pays a cold start.

Privacy: uploaded audio is written to a temporary folder and deleted when the request ends; input and
output text are not logged or stored.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
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

import agent as translator  # noqa: E402
import config  # noqa: E402
from agent import SCRIPT_RE, detect_text_language  # noqa: E402

log = logging.getLogger("frontend")
STATIC = Path(__file__).resolve().parent / "static"

MAX_TEXT_CHARS = config.MAX_TEXT_CHARS
MAX_UPLOAD_MB = config.MAX_AUDIO_BYTES // (1024 * 1024)

Lang = Literal["auto", "ar", "zh"]


class Service:
    """Owns the agent and the conversation session (the agent's short-term memory)."""

    def __init__(self) -> None:
        self.agent = translator.create_agent()
        self.session = self.agent.create_session()
        self.lock = asyncio.Lock()  # one request at a time: turns must not interleave in the session

    async def warm_up(self) -> None:
        await translator.warm_up(self.agent)

    def reset(self) -> None:
        self.session = self.agent.create_session()

    async def translate_text(self, text: str, lang: str) -> dict:
        t0 = time.perf_counter()
        async with self.lock:
            out = await translator.translate_text(self.agent, self.session, text, lang)
        return {"mode": "text", **out, "elapsed_s": round(time.perf_counter() - t0, 1), "audio_s": None}

    async def translate_audio(self, src: Path, lang: str) -> dict:
        t0 = time.perf_counter()
        async with self.lock:
            out = await translator.translate_audio(self.agent, self.session, src, lang)
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

    async def start_up() -> None:
        """Load models and warm up; the service is published only when it's fast to use."""
        try:
            if service_factory is None:
                from audio import load_whisper

                await asyncio.to_thread(load_whisper)  # first run downloads the Whisper model
            svc = (service_factory or Service)()
            if service_factory is None:
                t0 = time.perf_counter()
                await svc.warm_up()  # load the LLM + fill Ollama's prompt cache
                print(f"LLM warmed up in {time.perf_counter() - t0:.0f}s", flush=True)
            state["service"] = svc
        except Exception as e:  # keep serving so the page can show the problem
            log.exception("Agent failed to load")
            state["error"] = f"{type(e).__name__}: {e}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = [asyncio.create_task(start_up())]
        if service_factory is None:
            tasks.append(asyncio.create_task(translator.keep_model_loaded()))
        print("Frontend ready: http://127.0.0.1:8000 (models are loading in the background)", flush=True)
        yield
        for t in tasks:
            t.cancel()

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
