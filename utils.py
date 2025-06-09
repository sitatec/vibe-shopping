import os
from io import BytesIO
from PIL import Image


class ImageUploader:
    """
    A simple image uploader that can be used to upload images to a server and return their URLs.
    By default, this class assumes that it is deployed on Huggingface Spaces, you can extend it to
    upload images to any other server by overriding the `upload_image` method.
    """

    def _get_space_url(self):
        space_host = os.getenv("SPACE_HOST")
        return f"https://{space_host}"

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

        return f"{self._get_space_url()}/gradio_api/file={file_path}"


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
