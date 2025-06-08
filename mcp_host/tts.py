from typing import Generator

import numpy as np
import torch
import spaces
from kokoro import KPipeline, KModel

__all__ = ["text_to_speech"]

VOICES = {
    # American English
    "EN🇺🇸 Heart 👩": "af_heart",
    "EN🇺🇸 Bella 👩": "af_bella",
    "EN🇺🇸 Michael 👨": "am_michael",
    "EN🇺🇸 Fenrir 👨": "am_fenrir",
    # British English
    "EN🇬🇧 Emma 👩": "bf_emma",
    "EN🇬🇧 George 👨": "bm_george",
    # Japanese
    "JA🇯🇵 Alpha 👩": "jf_alpha",
    "JA🇯🇵 Kumo 👨": "jm_kumo",
    # Mandarin Chinese
    "ZH🇨🇳 Xiaoni 👩": "zf_xiaoni",
    "ZH🇨🇳 Yunjian 👨": "zm_yunjian",
    # Spanish
    "ES🇪🇸 Dora 👩": "ef_dora",
    "ES🇪🇸 Alex 👨": "em_alex",
    # French
    "FR🇫🇷 Siwis 👩": "ff_siwis",
    # Hindi
    "HI🇮🇳 Beta 👩": "hf_beta",
    "HI🇮🇳 Omega 👨": "hm_omega",
    # Italian
    "IT🇮🇹 Sara 👩": "if_sara",
    "IT🇮🇹 Nicola 👨": "im_nicola",
    # Brazilian Portuguese
    "PT🇧🇷 Dora 👩": "pf_dora",
    "PT🇧🇷 Santa 👨": "pm_santa",
}

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
    for lang_code in "abjzefhip"
}

# Preload voices into pipelines
for voice_code in VOICES.values():
    # First letter of the voice code is the language code (kokoro format)
    lang_code = voice_code[0]
    if lang_code in pipes:
        pipes[lang_code].load_voice(voice_code)

@spaces.GPU(duration=10)
def text_to_speech(text: str, voice: str = "af_heart") -> Generator[np.ndarray, None, None]:
    """
    Convert text to speech using the specified voice.
    
    Args:
        text (str): The text to convert to speech.
        voice (str): The voice to use for the conversion. Default to af_heart
    
    Returns:
        np.ndarray: The audio as a NumPy array.
    """
    if voice not in VOICES:
        raise ValueError(f"Voice '{voice}' is not available.")
    
    pipe = pipes[voice[0]]

    for _, __, result in pipe(text, voice=voice):
        yield result.audio.numpy()
