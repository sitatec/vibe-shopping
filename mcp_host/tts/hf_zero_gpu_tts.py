from typing import AsyncGenerator, Iterator

import numpy as np
import torch
import spaces
from kokoro import KPipeline, KModel
from stream2sentence import generate_sentences

from mcp_host.tts.utils import KOKORO_TO_STD_LANG, VOICES

__all__ = ["stream_text_to_speech"]


device = 0 if torch.cuda.is_available() else "cpu"
model = KModel().to(device).eval()

# Create a pipeline for each language. Kokoro language codes:
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
pipes = {
    lang_code: KPipeline(lang_code=lang_code, model=model, device=device)
    for lang_code in "abzefhip"
    # for lang_code in "abjzefhip"
}

# Preload voices into pipelines
for voice_code in VOICES.values():
    # First letter of the voice code is the language code (kokoro format)
    lang_code = voice_code[0]
    if lang_code in pipes:
        pipes[lang_code].load_voice(voice_code)


async def stream_text_to_speech(
    text_stream: Iterator[str], voice: str | None = None
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
        for audio in text_to_speech(text, pipe_key=kokoro_lang, voice=voice):
            yield 24000, audio


@spaces.GPU(duration=20)
def text_to_speech(
    text: str,
    pipe_key: str,
    voice: str | None = None,
):
    for _, __, audio in pipes[pipe_key](text, voice=voice):
        yield audio.numpy()