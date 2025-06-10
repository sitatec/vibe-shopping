from typing import AsyncGenerator, Iterator

import numpy as np
from fastrtc import KokoroTTSOptions, get_tts_model
from stream2sentence import generate_sentences

from mcp_host.tts.utils import KOKORO_TO_STD_LANG, VOICES

__all__ = ["stream_text_to_speech"]


model = get_tts_model(model="kokoro")


async def stream_text_to_speech(
    text_stream: Iterator[str], voice: str | None = None
) -> AsyncGenerator[tuple[int, np.ndarray], None]:
    """
    Convert text to speech using the specified voice.

    Args:
        text_stream (Iterator[str]): An iterator that yields text strings to convert to speech.
        voice (str | None): The voice to use for the conversion. Default to af_heart.

    Yields:
        np.ndarray: The audio as a NumPy array.
    """
    voice = voice or "af_heart"
    if voice not in VOICES.values():
        raise ValueError(f"Voice '{voice}' is not available.")

    kokoro_lang = voice[0]
    standard_lang_code = KOKORO_TO_STD_LANG.get(kokoro_lang, "en")

    options = KokoroTTSOptions(voice=voice, lang=standard_lang_code)

    for text in generate_sentences(text_stream, language=standard_lang_code):
      async for audio in model.stream_tts(text, options):
          yield audio
        
