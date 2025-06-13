import io
import time
from typing import Generator, Iterator
import base64
import wave

import numpy as np
from gradio_client import Client
from stream2sentence import generate_sentences

from mcp_host.tts.utils import KOKORO_TO_STD_LANG, VOICES


__all__ = ["stream_text_to_speech"]


def stream_text_to_speech(
    text_stream: Iterator[str],
    client: Client,
    voice: str | None = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
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
        print(f"Streaming audio for text: {text}")
        audio = client.submit(
            text=text, voice=voice, speed=1, use_gpu=True, api_name="/stream"
        )
        print("Job submitted, waiting for audio chunks...")
        t = time.time()
        for audio_chunk in audio:
            yield base64_to_audio_array(audio_chunk)
            print(f"Received audio chunk: {audio_chunk[:10]} in {time.time() - t:.2f} seconds")


def base64_to_audio_array(base64_string):
    # Decode base64 to raw WAV bytes
    audio_bytes = base64.b64decode(base64_string)
    buffer = io.BytesIO(audio_bytes)

    # Read WAV using wave module
    with wave.open(buffer, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()

        audio_data = wf.readframes(n_frames)

    # Convert bytes to NumPy array (assumes int16)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Reshape for stereo if needed
    if n_channels > 1:
        audio_array = audio_array.reshape(-1, n_channels)

    # Normalize to float32 [-1.0, 1.0]
    audio_array = audio_array.astype(np.float32) / 32767

    return sample_rate, audio_array