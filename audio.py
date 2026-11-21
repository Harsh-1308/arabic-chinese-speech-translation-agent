"""Audio helpers: file validation, microphone recording, transcription."""

import tempfile
import threading
from pathlib import Path

import numpy as np

import config

SAMPLE_RATE = 16_000
SILENCE_RMS_THRESHOLD = 0.005  # below this the recording is treated as silent


class AudioInputError(ValueError):
    """Raised when audio input fails a guardrail check."""


def validate_audio_file(path: str) -> Path:
    """Check that the file exists, has a supported format and is within size limits."""
    p = Path(path.strip().strip('"')).expanduser().resolve()
    if not p.is_file():
        raise AudioInputError(f"File not found: {p}")
    if p.suffix.lower() not in config.ALLOWED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_AUDIO_EXTENSIONS))
        raise AudioInputError(f"Unsupported format '{p.suffix}'. Allowed: {allowed}")
    size = p.stat().st_size
    if size == 0:
        raise AudioInputError("Audio file is empty.")
    if size > config.MAX_AUDIO_BYTES:
        raise AudioInputError("Audio file is larger than 25 MB.")
    return p


def record_from_mic(max_seconds: int = config.MAX_MIC_SECONDS) -> Path:
    """Record from the default microphone until Enter is pressed (or max_seconds)."""
    import sounddevice as sd
    import soundfile as sf

    chunks: list[np.ndarray] = []
    stop = threading.Event()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        chunks.append(indata.copy())

    print(f"Recording... press Enter to stop (max {max_seconds}s).")
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        stop.wait(timeout=max_seconds)
    print("Recording stopped.")

    if not chunks:
        raise AudioInputError("No audio was captured from the microphone.")
    audio = np.concatenate(chunks, axis=0)
    if float(np.sqrt(np.mean(audio**2))) < SILENCE_RMS_THRESHOLD:
        raise AudioInputError("Recording appears to be silent. Please speak closer to the mic.")

    out = Path(tempfile.gettempdir()) / "speech_translator_mic.wav"
    sf.write(out, audio, SAMPLE_RATE)
    return out


_whisper = None
_whisper_lock = threading.Lock()  # faster-whisper model: load once, run one transcription at a time


def _get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        print(f"Loading faster-whisper '{config.WHISPER_MODEL}' on {config.WHISPER_DEVICE} (first run downloads it)...")
        _whisper = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                                compute_type=config.WHISPER_COMPUTE_TYPE)
    return _whisper


def _transcribe_sync(path: Path, language: str | None) -> tuple[str, str, float]:
    with _whisper_lock:
        model = _get_whisper()
        segments, info = model.transcribe(
            str(path),
            task="transcribe",  # keep the source language; the agent does the translation
            language=language if language in config.SUPPORTED_LANGUAGES else None,
            vad_filter=True,  # skip silence
            beam_size=5,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    return text, info.language, info.language_probability


async def transcribe(path: Path, language: str | None = None) -> tuple[str, str, float]:
    """Transcribe locally with faster-whisper.

    language: optional "ar" or "zh" hint; when omitted Whisper auto-detects the language.
    Returns (text, detected_language_code, probability).
    """
    import asyncio

    return await asyncio.to_thread(_transcribe_sync, path, language)


def load_whisper() -> None:
    """Pre-load the Whisper model (e.g. at server start) so the first request isn't slow."""
    with _whisper_lock:
        _get_whisper()
