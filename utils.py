import os
from io import BytesIO
from PIL import Image
from fastrtc import WebRTCError
import requests


class ImageUploader:
    """
    A simple image uploader that can be used to upload images to a server and return their URLs.
    By default, this class assumes that it is deployed on Huggingface Spaces, you can extend it to
    upload images to any other server by overriding the `upload_image` method.
    """

    def upload_image(self, image: bytes | Image.Image, filename: str) -> str:
        """
        Upload an image to the server and return its URL.
        Args:
            image: The image to upload.

        Returns:
            str: The URL of the uploaded image.
        """
        if isinstance(image, Image.Image):
            image = pil_to_bytes(image)

        unique_filename = f"{os.urandom(8).hex()}_{filename}"
        file_path = f"/tmp/{unique_filename}"
        with open(file_path, "wb") as f:
            f.write(image)

        return f"{get_hf_space_file_url_prefix()}{file_path}"


def pil_to_bytes(image, format: str = "PNG") -> bytes:
    """
    Convert a PIL image to bytes.
    Args:
        image: The PIL image to convert.
        format: The format to use for the conversion (default is PNG).

    Returns:
        bytes: The image bytes.
    """

    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def get_hf_space_file_url_prefix() -> str:
    space_host = os.getenv("SPACE_HOST")
    return f"https://{space_host}/gradio_api/file="


def health_check_virtual_try_model():
    try:
        virtual_try_hc_response = requests.get(
            "https://sita-berete-3-vibe-shopping--health-check.modal.run"
        )
        virtual_try_hc_response.raise_for_status()
    except Exception as e:
        print(f"Virtual try-on model health check failed: {e}")
        raise WebRTCError("Error: Virtual try-on server failed to start")