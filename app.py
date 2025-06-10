from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
import numpy as np
from PIL import Image
from fastrtc import (
    AdditionalOutputs,
    WebRTC,
    ReplyOnPause,
    get_cloudflare_turn_credentials_async,
    get_cloudflare_turn_credentials,
)

from mcp_host.agent import VibeShoppingAgent
from mcp_host.tts import VOICES
from mcp_host.ui import UI

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

gr.set_static_paths("static/")


vibe_shopping_agent = VibeShoppingAgent()


def handle_image_upload(
    image_with_mask: dict | None,
) -> tuple[Image.Image | None, Image.Image | None]:
    if not image_with_mask:
        return None, None

    # Extract image and mask from ImageEditor data
    image = image_with_mask["background"]
    mask = None
    if "layers" in image_with_mask and len(image_with_mask["layers"]) > 0:
        mask = image_with_mask["layers"][0]  # First layer contains the mask

        # Convert mask to a binary mask (white for masked area, black for unmasked)
        mask_array = np.array(mask)
        is_black = np.all(mask_array < 10, axis=2)
        mask = Image.fromarray(((~is_black) * 255).astype(np.uint8))

    return image, mask


async def handle_audio_stream(
    audio: tuple[int, np.ndarray],
    chat_history: list[ChatCompletionMessageParam],
    voice: str | None = None,
    displayed_products: list[dict] | None = None,
    displayed_image: str | None = None,
    image_with_mask: dict | None = None,
):
    image, mask = handle_image_upload(image_with_mask)

    def update_ui(products, image, clear_ui):
        nonlocal displayed_products, displayed_image
        if clear_ui:
            displayed_products = None
            displayed_image = None
        else:
            displayed_products = products
            displayed_image = image

    async for ai_speech in vibe_shopping_agent.chat(
        user_speech=audio,
        chat_history=chat_history,
        voice=voice,
        update_ui=update_ui,
        input_image=image,
        input_mask=mask,
    ):
        # Yield the audio chunk to the WebRTC stream
        yield ai_speech

    yield AdditionalOutputs(
        chat_history, displayed_products, displayed_image, None
    )  # None for resetting the input_image state


with gr.Blocks(theme=gr.themes.Ocean()) as vibe_shopping_app:
    vibe_shopping_app.load(vibe_shopping_agent.connect_clients)

    chat_history = gr.State(value=[])
    displayed_products = gr.State(value=[])
    displayed_image = gr.State(value=None)
    with gr.Column():
        voice = gr.Dropdown(
            label="Language & Voice",
            choices=list(VOICES.items()),
            value=list(VOICES.values())[0],  # Default to the first voice
            info="The AI will always respond in the language you spoke to it. So make sure to speak in the language of the selected voice.",
            scale=0,
        )
        shopping_ui = UI(
            products_state=displayed_products,
            image_state=displayed_image,
        )
        audio_stream = WebRTC(
            label="Stream",
            mode="send-receive",
            modality="audio",
            button_labels={"start": "Start Vibe Shopping"},
            rtc_configuration=get_cloudflare_turn_credentials_async,
            server_rtc_configuration=get_cloudflare_turn_credentials(ttl=360_000),
            scale=0,
        )
        with gr.Accordion(open=False, label="Input Image"):
            gr.Markdown(
                "Select an image to show to the AI assistant. You can click on the edit (🖌) icon and draw a mask. "
                "Once you select the image you need to let the assistant know that you have done so, and tell it what you want to do with the image if it doesn't already know from the context of the conversation.\n\n"
                "The mask is optional, but it can improve the quality of the results. If you notice that an object you virtually tried on is miss-placed or not fitting well, try adding a mask. "
                "For example if you want to try trying on a shirt, draw a mask over the upper body of the person in the image. "
                "If you want to see how a furniture looks in a room, draw a mask over the area where you want it to be placed... "
            )
            input_image = gr.ImageMask(type="pil")

    audio_stream.stream(
        ReplyOnPause(handle_audio_stream),  # type: ignore
        inputs=[
            audio_stream,
            chat_history,
            voice,
            displayed_products,
            displayed_image,
            input_image,
        ],
        outputs=[audio_stream],
    )
    audio_stream.on_additional_outputs(
        lambda s, a: (s, a),
        outputs=[chat_history, displayed_products, displayed_image, input_image],
        queue=False,
        show_progress="hidden",
    )

    vibe_shopping_app.launch()
