"""Interactive CLI for the SpeechTranslator agent.

Commands:
  /file <path>   translate an audio file
  /mic           record from microphone and translate
  /reset         clear conversation memory
  /quit          exit
  anything else  treated as Arabic/Chinese text to translate
"""

import asyncio

import config
from agent import audio_message, create_agent, detect_text_language, text_message, translate
from audio import AudioInputError, record_from_mic, validate_audio_file

HELP = __doc__.split("Commands:")[1]


async def main() -> None:
    agent = create_agent()
    session = agent.create_session()
    print("SpeechTranslator (Arabic/Chinese -> English, local: faster-whisper + Ollama)\nCommands:" + HELP)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        try:
            if line == "/quit":
                break
            elif line == "/reset":
                session = agent.create_session()
                print("Memory cleared.")
                continue
            elif line == "/mic":
                message = audio_message(record_from_mic())
            elif line.startswith("/file"):
                message = audio_message(validate_audio_file(line[len("/file"):]))
            elif line.startswith("/"):
                print("Unknown command." + HELP)
                continue
            else:
                if len(line) > config.MAX_TEXT_CHARS:
                    print(f"Text too long (max {config.MAX_TEXT_CHARS} characters).")
                    continue
                lang = detect_text_language(line)
                if lang is None:
                    print("Unsupported input: please type Arabic or Chinese text.")
                    continue
                message = text_message(line, lang)

            print("Translating...")
            print("\n" + await translate(agent, session, message))

        except AudioInputError as e:
            print(f"Input error: {e}")
        except Exception as e:  # keep the CLI alive on model / device errors
            print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
