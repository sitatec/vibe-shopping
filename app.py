from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from utils import health_check_virtual_try_model


IS_HF_ZERO_GPU = os.getenv("SPACE_ID", "").startswith("sitatech/")
IS_LOCAL = os.getenv("LOCALE_RUN") is not None
print("IS_LOCAL:", IS_LOCAL)
if IS_LOCAL:
    import dotenv

    dotenv.load_dotenv(override=True)

    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY env var must be set"
    assert os.getenv("OPENAI_API_BASE_URL") is not None, (
        "OPENAI_API_BASE_URL env var must be set"
    )
    assert os.getenv("STT_OPENAI_API_KEY") is not None, (
        "STT_OPENAI_API_KEY env var must be set"
    )

    print("OPENAI_API_BASE_URL: ", os.getenv("OPENAI_API_BASE_URL"))

import gradio as gr
from gradio_modal import Modal
import requests
from gradio_client import Client
import numpy as np
from PIL import Image
from fastrtc import (
    AdditionalOutputs,
    AlgoOptions,
    WebRTC,
    ReplyOnPause,
    get_cloudflare_turn_credentials_async,
    get_cloudflare_turn_credentials,
    get_twilio_turn_credentials,
    WebRTCError,
)

from mcp_host.agent import VibeShoppingAgent
from mcp_host.tts.utils import VOICES
from mcp_host.ui import WelcomeUI, ColdBootUI, ImageDisplay, ProductList

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
        # If the whole mask is black, we can assume no mask was drawn
        if np.all(mask_array < 10):
            mask = None
        else:
            is_black = np.all(mask_array < 10, axis=2)
            mask = Image.fromarray(((~is_black) * 255).astype(np.uint8))

    return image, mask


def handle_audio_stream(
    audio: tuple[int, np.ndarray],
    chat_history: list[ChatCompletionMessageParam],
    voice: str | None = None,
    ui_html: str | None = None,
    displayed_image: str | None = None,
    image_with_mask: dict | None = None,
    gradio_client: Client | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    system_prompt: str | None = None,
):
    try:
        image_modal_visible = False
        image, mask = handle_image_upload(image_with_mask)
        chat_history = chat_history.copy()
        for ai_speech_or_ui_update in vibe_shopping_agent.chat(
            user_speech=audio,
            chat_history=chat_history,
            voice=voice,
            input_image=image,
            input_mask=mask,
            gradio_client=gradio_client,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
        ):
            is_ui_update = len(ai_speech_or_ui_update) == 3
            if is_ui_update:
                product_list, image_url, clear_ui = ai_speech_or_ui_update

                if product_list:
                    ui_html = ProductList(product_list)
                    image_modal_visible = False
                elif image_url:
                    displayed_image = ImageDisplay(image_url)
                    image_modal_visible = True
                elif clear_ui:
                    ui_html = WelcomeUI()
                    displayed_image = ""
                    image_modal_visible = False

                yield AdditionalOutputs(
                    chat_history,
                    ui_html,
                    displayed_image,
                    Modal(visible=image_modal_visible),
                    None,  # None for resetting the input_image state
                )
                continue

            # Yield the audio chunk to the WebRTC stream
            yield ai_speech_or_ui_update

        yield AdditionalOutputs(
            chat_history,
            ui_html,
            displayed_image,
            Modal(visible=image_modal_visible),
            # The last None for resetting the input_image state
            None,
        )
    except Exception as e:
        print(f"Error in handle_audio_stream: {e}")
        raise WebRTCError(f"An error occurred: {e}")


def set_client_for_session(request: gr.Request):
    health_check_response = requests.get(
        os.environ["OPENAI_API_BASE_URL"].replace("/v1", "/health")
    )
    if health_check_response.status_code != 200:
        raise WebRTCError(
            f"Inference server is not available. Status code: {health_check_response.status_code}"
        )

    threading.Thread(target=health_check_virtual_try_model).start()

    if not vibe_shopping_agent.clients_connected:
        vibe_shopping_agent.connect_clients()

    if IS_HF_ZERO_GPU:
        # No need to set client for HF Zero GPU, we will run tts & stt inference on the gpu
        return None, Modal(visible=False)

    if "x-ip-token" not in request.headers:
        # Probably running in a local environment
        return Client("sitatech/Kokoro-TTS"), Modal(visible=False)

    x_ip_token = request.headers["x-ip-token"]

    return Client("sitatech/Kokoro-TTS", headers={"X-IP-Token": x_ip_token}), Modal(
        visible=False
    )


with gr.Blocks(
    theme=gr.themes.Ocean(),
    css="#main-container { max-width: 1200px; margin: 0 auto; }",
) as vibe_shopping_app:
    gradio_client = gr.State()

    debuging_options = {
        "Echo user speech (for debugging)": "debug_echo_user_speech",
    }

    chat_history = gr.State(value=[])
    with gr.Column(elem_id="main-container"):
        with gr.Accordion(open=False, label="Important Notes"):
            gr.Markdown(
                "1. This is a demo hacked together in ~10 days, so it may not handle well all the edge cases.\n"
                "2. The demo is powered by a quantized **Qwen-2.5VL 72 Billion parameters** model and is running on a **single H100**, so **expect some delays** especially when displaying many items like product search results.\n"
                "3. The virtual try-on model is powered by a fine-tuned flux-fill combined with auto-masking. The model was exclusively trained on clothing with white background, but I made some adjustments that should generalize but the results may not be perfect for all items.\n"
                "4. Unless you are in very quiet place, I recommend you muting yourself by disabling the mic when you are done speaking (when waiting for or listening to the AI), otherwise the AI may pickup some background voices or even sometimes background noises that may break the flow of your conversation. Also, avoid making long pauses when speaking, otherwise the AI will think you are done speaking and start answering (This can be adjusted if you fork the repo).\n"
                "5. This demo was made possible thanks to the generous **250$ in compute credits** provided by [Modal](modal.com) and 1 month of Pro subscription from [HuggingFace](huggingface.co).\n"
                "6. If by the time you see this demo the credits are exhausted, you can still clone the repo and run it yourself, let me know in the [community tab of the demo](https://huggingface.co/spaces/sitatech/vibe-shopping/discussions) if you need help with that, I will be happy to assist you.\n"
            )
        voice = gr.Dropdown(
            label="Language & Voice",
            choices=list(VOICES.items()) + list(debuging_options.items()),
            value=list(VOICES.values())[0],  # Default to the first voice
            info="The AI will always respond in the language you spoke to it. So make sure to speak in the language of the selected voice.",
            scale=0,
        )
        shopping_ui = gr.HTML(
            WelcomeUI(),
            container=True,
            max_height=600,
            min_height=450,
            padding=False,
        )
        with Modal(visible=False) as displayed_image_modal:
            displayed_image = gr.HTML("")

        audio_stream = WebRTC(
            label="Audio Chat",
            mode="send-receive",
            modality="audio",
            button_labels={"start": "Start Vibe Shopping"},
            rtc_configuration=(get_twilio_turn_credentials() if not IS_LOCAL else None),
            # rtc_configuration=(
            #     get_cloudflare_turn_credentials_async if not IS_LOCAL else None
            # ),
            # server_rtc_configuration=(
            #     get_cloudflare_turn_credentials(ttl=360_000) if not IS_LOCAL else None
            # ),
            scale=0,
            time_limit=3600,
        )
        with gr.Accordion(open=False, label="Input Image"):
            gr.Markdown(
                "Select an image to show to the AI assistant. You can click on the edit (🖌) icon and draw a mask. "
                "Once you select the image you need to let the assistant know that you have done so, and tell it what you want to do with the image if it doesn't already know from the context of the conversation.\n\n"
                "The mask is optional, but it can improve the quality of the results. If you notice that an object you virtually tried on is miss-placed or not fitting well, try adding a mask. "
                "For example if you want to try trying on a shirt, draw a mask over the upper body of the person in the image. "
                "If you want to see how a furniture looks in a room, draw a mask over the area where you want it to be placed... "
            )
            input_image = gr.ImageMask(value=None, type="pil")

        with gr.Accordion(open=False, label="LLM Parameters"):
            gr.Markdown(
                "You can change the LLM parameters to control the behavior of the AI assistant. "
                "For example, you can make it more creative or more focused on specific tasks."
            )

            with gr.Row():
                with gr.Column():
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                    )

                with gr.Column():
                    top_p = gr.Slider(
                        label="Top P",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                    )

                with gr.Column():
                    clear_history_btn = gr.Button("Clear Chat History", variant="primary")
                    clear_history_btn.click(
                        lambda: [],
                        inputs=[],
                        outputs=chat_history,
                    )

            system_prompt = gr.Textbox(
                label="System Prompt",
                value=VibeShoppingAgent.SYSTEM_PROMPT,
                lines=20,
            )

    audio_stream.stream(
        ReplyOnPause(
            handle_audio_stream,
            algo_options=AlgoOptions(
                speech_threshold=0.3
            )
        ),
        inputs=[
            audio_stream,
            chat_history,
            voice,
            shopping_ui,
            displayed_image,
            input_image,
            gradio_client,
            temperature,
            top_p,
            system_prompt,
        ],
        outputs=[audio_stream],
    )
    audio_stream.on_additional_outputs(
        lambda *args: (
            args[-5],  # chat_history
            args[-4],  # shopping_ui
            args[-3],  # displayed_image
            args[-2],  # displayed_image_modal
            args[-1],  # input_image
        ),  # Last four outputs
        outputs=[
            chat_history,
            shopping_ui,
            displayed_image,
            displayed_image_modal,
            input_image,
        ],
        show_progress="hidden",
    )

    with Modal(visible=True, allow_user_close=False) as modal:
        ColdBootUI()

    vibe_shopping_app.load(set_client_for_session, None, [gradio_client, modal])
    vibe_shopping_app.queue().launch(allowed_paths=["/tmp/vibe-shopping-public/"])
