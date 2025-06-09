import asyncio

import gradio as gr
import numpy as np
from fastrtc import AdditionalOutputs, WebRTC, ReplyOnPause
from openai.types.chat import ChatCompletionMessageParam

from mcp_host.agent import VibeShoppingAgent
from mcp_host.tts import VOICES
from mcp_host.ui import UI


vibe_shopping_agent = VibeShoppingAgent()
asyncio.run(vibe_shopping_agent.connect_clients())


async def handle_audio_stream(
    audio: tuple[int, np.ndarray],
    chat_history: list[ChatCompletionMessageParam],
    voice: str | None = None,
    displayed_products: list[dict] | None = None,
    displayed_image: str | None = None,
):
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
    ):
        # Yield the audio chunk to the WebRTC stream
        yield ai_speech

    yield AdditionalOutputs(chat_history, displayed_products, displayed_image)


with gr.Blocks(theme=gr.themes.Ocean()) as vibe_shopping_app:
    chat_history = gr.State(value=[])
    displayed_products = gr.State(value=[])
    displayed_image = gr.State(value=None)
    with gr.Column():
        voice = gr.Dropdown(
            label="AI Voice",
            choices=list(VOICES.items()),
            value=list(VOICES.items())[0],
            info="Select a voice for the AI assistant.",
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
            scale=0,
        )

    audio_stream.stream(
        ReplyOnPause(handle_audio_stream),  # type: ignore
        inputs=[
            audio_stream,
            chat_history,
            voice,
            displayed_products,
            displayed_image,
        ],
        outputs=[audio_stream],
    )
    audio_stream.on_additional_outputs(
        lambda s, a: (s, a),
        outputs=[chat_history, displayed_products, displayed_image],
        queue=False,
        show_progress="hidden",
    )

    vibe_shopping_app.launch()
