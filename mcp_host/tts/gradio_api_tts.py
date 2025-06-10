from typing import AsyncGenerator, Iterator

import numpy as np
from gradio_client import Client
from stream2sentence import generate_sentences

from mcp_host.tts.utils import KOKORO_TO_STD_LANG, VOICES


__all__ = ["stream_text_to_speech"]


async def stream_text_to_speech(
    text_stream: Iterator[str],
    client: Client,
    voice: str | None = None,
) -> AsyncGenerator[tuple[int, np.ndarray], None]:
    """
    Convert text to speech using the specified voice.

    Args:
        text (str): The text to convert to speech.
        voice (str): The voice to use for the conversion. Default to af_heart

    Returns:
        np.ndarray: The audio as a NumPy array.
    """
    voice = voice or "af_heart"
    if voice not in VOICES.values():
        raise ValueError(f"Voice '{voice}' is not available.")

    kokoro_lang = voice[0]
    standard_lang_code = KOKORO_TO_STD_LANG[kokoro_lang]

    for text in generate_sentences(text_stream, language=standard_lang_code):
        audio = client.submit(
            text=text, voice=voice, speed=1, use_gpu="true", api_name="/generate_all"
        )
        for audio_chunk in audio:
            yield audio_chunk
