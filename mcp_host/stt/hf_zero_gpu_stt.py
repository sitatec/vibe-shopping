import numpy as np
import spaces
import torch
from transformers import pipeline

__all__ = ["speech_to_text"]

MODEL_NAME = "openai/whisper-large-v3-turbo"
device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
)


@spaces.GPU(duration=10)
def speech_to_text(inputs: tuple[int, np.ndarray]) -> str:
    sampling_rate, audio = inputs
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = audio.squeeze()

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    text: str = pipe({"sampling_rate": sampling_rate, "raw": audio})["text"]  # type: ignore
    return text
