"""Interactive CLI for the SpeechTranslator agent.

Commands:
  /file <path>   translate an audio file
  /doc <path>    translate a document (PDF, DOCX or image)
  /mic           record from microphone and translate
  /reset         clear conversation memory
  /quit          exit
  anything else  treated as Arabic/Chinese text to translate
"""

import asyncio
import time

import config
import agent as translator
import documents
from audio import AudioInputError, record_from_mic, validate_audio_file

HELP = __doc__.split("Commands:")[1]


def show(result: dict, t0: float) -> None:
    print(f"\nDetected language: {result['language']}\nOriginal: {result['original']}\n"
          f"English: {result['english']}\n({time.perf_counter() - t0:.1f}s)")


async def main() -> None:
    agent = translator.create_agent()
    session = agent.create_session()

    # Load the LLM and fill Ollama's prompt cache now, so the first real translation is fast,
    # and keep the model loaded while the CLI is open.
    print("Warming up the model (about 1-2 min on first start)...")
    await translator.warm_up(agent)
    keep_alive = asyncio.create_task(translator.keep_model_loaded())

    print("SpeechTranslator (Arabic/Chinese -> English, local: faster-whisper + Ollama)\nCommands:" + HELP)
    while True:
        try:
            # input() blocks, so run it in a thread to let the keep-alive task keep running
            line = (await asyncio.to_thread(input, "\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        try:
            t0 = time.perf_counter()
            if line == "/quit":
                break
            elif line == "/reset":
                session = agent.create_session()
                print("Memory cleared.")
            elif line == "/mic":
                path = await asyncio.to_thread(record_from_mic)
                print("Translating...")
                show(await translator.translate_audio(agent, session, path), t0)
            elif line.startswith("/doc"):
                print("Extracting text...")
                doc = await asyncio.to_thread(documents.extract_document, line[len("/doc"):])
                print(f"{doc.kind.upper()}, {doc.pages} page(s), {doc.ocr_pages} read with OCR, "
                      f"{len(doc.chunks)} part(s). Translating...")
                async for r in translator.translate_document(agent, doc):
                    flag = " [check OCR]" if r["check"] else " [kept as-is]" if r.get("skipped") else ""
                    english = r["english"] or f"({r.get('error')})"
                    print(f"\n--- part {r['index'] + 1}/{r['total']} ({r['where']}){flag}\n"
                          f"{r['original']}\n=> {english}")
                print(f"\n({time.perf_counter() - t0:.1f}s)")
            elif line.startswith("/file"):
                path = validate_audio_file(line[len("/file"):])
                print("Translating...")
                show(await translator.translate_audio(agent, session, path), t0)
            elif line.startswith("/"):
                print("Unknown command." + HELP)
            else:
                if len(line) > config.MAX_TEXT_CHARS:
                    print(f"Text too long (max {config.MAX_TEXT_CHARS} characters).")
                    continue
                lang = translator.detect_text_language(line)  # code decides the language, not the model
                if lang is None:
                    print("Unsupported input: please type Arabic or Chinese text.")
                    continue
                print("Translating...")
                show(await translator.translate_text(agent, session, line, lang), t0)

        except (AudioInputError, ValueError) as e:  # input checks and agent-reported errors
            print(e)
        except Exception as e:  # keep the CLI alive on model / device errors
            print(f"Error: {type(e).__name__}: {e}")

    keep_alive.cancel()


if __name__ == "__main__":
    asyncio.run(main())
