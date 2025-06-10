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
)


@spaces.GPU(duration=10)
def speech_to_text(inputs: tuple[int, np.ndarray]) -> str:
    sampling_rate, audio = inputs
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 1e-9:  # small epsilon to guard against floating-point imprecision
        audio = audio / peak

    text: str = pipe({"sampling_rate": sampling_rate, "raw": audio})["text"]  # type: ignore
    return text
