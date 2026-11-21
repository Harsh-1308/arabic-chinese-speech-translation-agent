# SpeechTranslator (MAF, fully local)

A basic Microsoft Agent Framework agent that translates **Arabic** and **Chinese** speech, text and
documents (PDF, Word, images) into **English**.
No API keys: speech-to-text is **faster-whisper**, OCR is **RapidOCR** and translation is a local
**Ollama** model.

## Setup
1. Install [Ollama](https://ollama.com) and pull the model: `ollama pull qwen2.5:7b`
2. Python deps:
   ```powershell
   .venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env   # optional: change models here
   ```
   The Whisper and OCR models download automatically on first start.

## Run (web page)
```powershell
python -m frontend.app      # then open http://127.0.0.1:8000
```
Tabs for **Text**, **Audio** (upload or browser microphone) and **Document** (PDF, DOCX, images),
a source-language selector (Auto / Arabic / Chinese) and a "Clear memory" button. Documents are
translated part by part and results appear as they are ready (with a Stop button).

## Run (command line)
```powershell
python main.py
```
| Input | How |
|---|---|
| Audio file | `/file C:\path\clip.mp3` (wav, mp3, m4a, ogg, flac, webm, mp4; max 25 MB) |
| Document | `/doc C:\path\report.pdf` (pdf, docx, png, jpg, webp, bmp, tiff; max 10 MB / 10 pages) |
| Microphone | `/mic`, then speak and press Enter to stop (max 60 s) |
| Text | Type or paste Arabic or Chinese text |
| Clear memory | `/reset` |
| Exit | `/quit` |

## Documents (`documents.py`)
| Type | How the text is read |
|---|---|
| PDF with text | PyMuPDF text layer, in reading order; Unicode NFKC turns Arabic presentation forms back into normal letters |
| Scanned PDF | The page's embedded image at native resolution (else rendered at 200 dpi), then OCR |
| DOCX | python-docx: paragraphs **and tables** in document order (table cells joined with `\|`) |
| Image | RapidOCR; Chinese model PP-OCRv6, Arabic model PP-OCRv5 (with python-bidi for right-to-left) |

OCR details that were tested and fixed:
- **Numbers inside Arabic lines**: the Arabic OCR model silently drops Western digits/Latin text in an
  Arabic line ("عدد الموظفين 120 موظفاً" came out without "120"). Word boxes reveal the gap, which is
  cropped and read with the Latin/Chinese model, then put back in reading order.
- **Scale sensitivity**: greyscale/upscaling made Arabic OCR worse, so images are left as-is (only tiny
  ones are upscaled); if any Arabic line has low confidence, OCR retries at 0.75x and 1.5x and keeps
  the best result.
- With "Auto" language, both OCR models run and the one that reads more Arabic/Chinese wins.
- Parts with OCR confidence below 80% are tagged **check OCR** in the results.

Translation: text is split into ~250-character parts (paragraph boundaries first), parts with no
Arabic/Chinese are kept as-is, and each document uses its own agent session (consistent names and
terms, chat memory untouched) that restarts before it could overflow the 4096-token context.

## Models and speed (CPU)
- `WHISPER_MODEL=small` is the default; `medium` is noticeably better for Arabic but slower.
- `OLLAMA_CHAT_MODEL=qwen2.5:7b` is the default; `qwen2.5:3b` is ~1.5x faster here, with lower quality.
- Start-up takes ~1-2 min (Whisper, OCR and LLM warm-up); after that a short sentence takes ~8-15 s
  on a 4-core laptop CPU (was 160 s+ before the optimisations below).
- Documents: OCR takes ~5-20 s per page; translation runs at ~2 tokens/s, so roughly
  1 minute per 250 characters of Chinese (a dense page is several minutes).

### Speed optimisations (measured with Ollama's timings)
| Problem | Fix |
|---|---|
| Ollama unloads the model after 5 idle minutes; reload + re-reading the prompt took ~80 s | Warm-up request at start-up and a background keep-alive ping every 4 min |
| Generation is the slowest step (~2 tokens/s) and the model echoed the original text | Model replies with the English only; language and original come from code / the tool |
| A changed prompt prefix forces Ollama to re-read the whole prompt | Instructions and tool list are identical on every call; history is appended, never trimmed |
| Runaway generations | `max_tokens` cap (`MAX_OUTPUT_TOKENS`) and temperature 0 |

## Design
- **Agent** (`agent.py`): role, task, context, examples and guardrails are in the instructions;
  `transcribe_audio` is the agent's tool. Document extraction runs in code before the agent (text must
  be chunked before the model sees it, and a tool round trip would add ~20 s per document).
- **Memory**: one in-memory MAF `AgentSession` for chat; a separate one per document;
  `/reset` or "Clear memory" starts a new chat session.
- **Guardrails in code**: file type, size, page and character limits; silence checks; language
  detection by script (text/documents) and by Whisper (audio), so only Arabic and Chinese reach the
  model; one automatic retry if Arabic/Chinese characters are left in the English output; error
  messages never expose server paths.
