import io
import os
import wave
import numpy as np
from openai import OpenAI
from numpy.typing import NDArray


__all__ = ["speech_to_text"]


class OpenAISTT:
    def __init__(
        self,
        api_key: str = os.getenv("STT_OPENAI_API_KEY", ""),
        api_base: str = os.getenv("STT_OPENAI_API_BASE_URL", "https://api.sambanova.ai/v1"),
        model: str = "Whisper-Large-v3",
    ):
        self.openai_client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        self.model = model

    def _numpy_to_wav_bytes(self, sample_rate: int, audio: NDArray[np.int16]) -> bytes:
        """
        Convert numpy int16 audio array to a WAV bytes buffer.
        """
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)  # mono audio
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())
            return buf.getvalue()

    def transcribe_from_tuple(
        self, audio_input: tuple[int, NDArray[np.int16]], language: str | None = None
    ) -> str:
        """
        Transcribe audio from a tuple (sample_rate, audio_array).
        :param audio_input: Tuple of (sample_rate, np.int16 numpy array).
        :param language: Optional language code.
        :return: Transcription string.
        """
        sample_rate, audio_array = audio_input

        wav_bytes = self._numpy_to_wav_bytes(sample_rate, audio_array)

        # Prepare the file-like object to pass to openai.Audio.transcribe
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"  # some APIs require the filename attribute

        try:
            if language:
                response = self.openai_client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language,
                )
            else:
                response = self.openai_client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")


model = OpenAISTT()

## Need to have the same interface as the other STT (like zero_gpu_stt.py)
speech_to_text = model.transcribe_from_tuple
