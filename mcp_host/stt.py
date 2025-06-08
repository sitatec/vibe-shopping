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
    text: str = pipe(inputs)["text"]  # type: ignore
    return text
