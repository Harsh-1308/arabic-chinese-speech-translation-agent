# SpeechTranslator (MAF, fully local)

A basic Microsoft Agent Framework agent that translates **Arabic** and **Chinese** speech or text into **English**.
No API keys: speech-to-text is **faster-whisper** and translation is a local **Ollama** model.

## Setup
1. Install [Ollama](https://ollama.com) and pull the model: `ollama pull qwen2.5:7b`
2. Python deps:
   ```powershell
   .venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env   # optional: change models here
   ```

## Run (web page)
```powershell
python -m frontend.app      # then open http://127.0.0.1:8000
```
Text, audio upload and browser microphone recording, with a source-language selector
(Auto / Arabic / Chinese) and a "Clear memory" button.

## Run (command line)
```powershell
python main.py
```
| Input | How |
|---|---|
| Audio file | `/file C:\path\clip.mp3` (wav, mp3, m4a, ogg, flac, webm, mp4; max 25 MB) |
| Microphone | `/mic`, then speak and press Enter to stop (max 60 s) |
| Text | Type or paste Arabic or Chinese text |
| Clear memory | `/reset` |
| Exit | `/quit` |

## Models and speed (CPU)
- `WHISPER_MODEL=small` is the default; `medium` is noticeably better for Arabic but slower.
- `OLLAMA_CHAT_MODEL=qwen2.5:7b` is the default; `qwen2.5:3b` is ~1.5x faster here, with lower quality.
- Start-up takes ~1-2 min (Whisper load + LLM warm-up); after that a short sentence takes ~8-15 s
  on a 4-core laptop CPU (was 160 s+ before the optimisations below).

### Speed optimisations (measured with Ollama's timings)
| Problem | Fix |
|---|---|
| Ollama unloads the model after 5 idle minutes; reload + re-reading the prompt took ~80 s | Warm-up request at start-up and a background keep-alive ping every 4 min |
| Generation is the slowest step (~2 tokens/s) and the model echoed the original text | Model replies with the English only; language and original come from code / the tool |
| A changed prompt prefix forces Ollama to re-read the whole prompt | Instructions and tool list are identical on every call; history is appended, never trimmed |
| Runaway generations | `max_tokens` cap (`MAX_OUTPUT_TOKENS`) and temperature 0 |

## Design
- **Agent** (`agent.py`): role, task, context, examples and guardrails are in the instructions;
  `transcribe_audio` is the agent's tool.
- **Memory**: one in-memory MAF `AgentSession`; `/reset` or "Clear memory" starts a new one.
- **Guardrails in code**: file format, size and silence checks; language detection by script (text)
  and by Whisper (audio), so only Arabic and Chinese reach the model; one automatic retry if
  Arabic/Chinese characters are left in the English output.
