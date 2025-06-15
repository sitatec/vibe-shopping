import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import requests
from typing import List, Optional

from fastrtc import WebRTCError


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
    

def make_image_grid_with_index_labels(img_bytes: List[Optional[bytes]]) -> Image.Image:
    """
    Combine 10 images (byte strings or None) into a single 1000×1000 grid image (5 per row),
    with 2px padding between images. Each cell is 200×200 including padding.
    Original image aspect ratio is preserved. Each image is labeled with its index (1–10).
    If an image is None, display "Invalid Image" centered in the cell.

    :param img_bytes: List of 10 items, each either image bytes or None.
    :return: A new PIL Image of size 1000×1000 pixels.
    """
    GRID_SIZE = (5, 2)
    CELL_SIZE = (200, 200)
    INNER_PADDING = 2
    IMAGE_AREA = (CELL_SIZE[0] - INNER_PADDING, CELL_SIZE[1] - INNER_PADDING)
    CANVAS_SIZE = (GRID_SIZE[0] * CELL_SIZE[0], GRID_SIZE[1] * CELL_SIZE[1])
    LABEL_PADDING = 4

    canvas = Image.new('RGB', CANVAS_SIZE, color='white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, b in enumerate(img_bytes):
        col, row = idx % GRID_SIZE[0], idx // GRID_SIZE[0]
        cell_x = col * CELL_SIZE[0]
        cell_y = row * CELL_SIZE[1]

        if b is not None:
            try:
                with Image.open(BytesIO(b)) as im:
                    im.thumbnail(IMAGE_AREA, Image.Resampling.BICUBIC)
                    x = cell_x + (CELL_SIZE[0] - im.width) // 2
                    y = cell_y + (CELL_SIZE[1] - im.height) // 2
                    canvas.paste(im, (x, y))

                    label = str(idx)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    bg_rect = [x, y, x + text_w + 2 * LABEL_PADDING, y + text_h + 2 * LABEL_PADDING]
                    draw.rectangle(bg_rect, fill='black')
                    draw.text((x + LABEL_PADDING, y + LABEL_PADDING), label, fill='white', font=font, stroke_width=0.1)
            except Exception:
                b = None  # treat as invalid if loading fails

        if b is None:
            # draw "Invalid Image" centered in the cell
            msg = f"Image {idx} Invalid"
            
            font = ImageFont.load_default(size=14)
            bbox = draw.textbbox((0, 0), msg, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = cell_x + (CELL_SIZE[0] - text_w) // 2
            y = cell_y + (CELL_SIZE[1] - text_h) // 2
            draw.text((x, y), msg, fill='black', font=font)

    return canvas