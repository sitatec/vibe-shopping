from typing import Callable, cast

import modal
from configs import (
    image,
    hf_cache_vol,
    API_KEY,
    MINUTE,
    PORT,
)

with image.imports():
    import torch
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    from diffusers import FluxFillPipeline
    from nunchaku import NunchakuFluxTransformer2dModel
    from nunchaku.utils import get_precision
    from nunchaku.lora.flux.compose import compose_lora

    from virtual_try.auto_masker import AutoInpaintMaskGenerator

    TransformType = Callable[[Image.Image | np.ndarray], torch.Tensor]

app = modal.App("vibe-shopping")


@app.cls(
    image=image,
    gpu="A100-40GB",
    cpu=4,  # 8vCPUs
    memory=16,  # 16 GB RAM
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[API_KEY],
    scaledown_window=(
        1 * MINUTE
        # how long should we stay up with no requests? Keep it low to minimize credit usage for now.
    ),
    timeout=10 * MINUTE,  # how long should we wait for container start?
)
class VirtualTryModel:
    @modal.enter()
    def enter(self):
        precision = get_precision()  # auto-detect precision 'int4' or 'fp4' based GPU
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/nunchaku-flux.1-fill-dev/svdq-{precision}_r32-flux.1-fill-dev.safetensors"
        )
        transformer.set_attention_impl("nunchaku-fp16")
        composed_lora = compose_lora(
            [
                ("xiaozaa/catvton-flux-lora-alpha/pytorch_lora_weights.safetensors", 1),
                ("ByteDance/Hyper-SD/Hyper-FLUX.1-dev-8steps-lora.safetensors", 0.125),
            ]
        )
        transformer.update_lora_params(composed_lora)

        self.pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        self.auto_masker = AutoInpaintMaskGenerator()

    def get_preprocessors(
        self, input_size: tuple[int, int], target_megapixels: float = 1.0
    ) -> tuple[TransformType, TransformType, tuple[int, int]]:
        num_pixels = int(target_megapixels * 1024 * 1024)

        input_width, input_height = input_size

        # Resizes the input dimensions to the target number of megapixels while maintaining
        # the aspect ratio and ensuring the new dimensions are multiples of 64.
        scale_by = np.sqrt(num_pixels / (input_height * input_width))
        new_height = int(np.ceil((input_height * scale_by) / 64)) * 64
        new_width = int(np.ceil((input_width * scale_by) / 64)) * 64

        transform = cast(
            TransformType,
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((new_height, new_width)),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )
        mask_transform = cast(
            TransformType,
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((new_height, new_width)),
                ]
            ),
        )
        return transform, mask_transform, (new_width, new_height)

    @modal.method()
    def try_it(
        self,
        item_to_try: Image.Image,
        image: Image.Image,
        mask: Image.Image | np.ndarray | None = None,
        prompt: str | None = None,
        masking_prompt: str | None = None,
    ) -> Image.Image:
        assert mask or masking_prompt, "Either mask or masking_prompt must be provided."

        preprocessor, mask_preprocessor, output_size = self.get_preprocessors(
            input_size=image.size,
            target_megapixels=0.7,  # The image will be stacked which will double the pixel count
        )

        if mask is None:
            # Generate mask using the auto-masker
            mask = self.auto_masker.generate_mask(
                image,
                prompt=masking_prompt,  # type: ignore
            )

        image_tensor = preprocessor(image.convert("RGB"))
        item_to_try_tensor = preprocessor(item_to_try.convert("RGB"))
        mask_tensor = mask_preprocessor(mask)

        # Create concatenated images
        inpaint_image = torch.cat(
            [item_to_try_tensor, image_tensor], dim=2
        )  # Concatenate along width
        extended_mask = torch.cat([torch.zeros_like(mask_tensor), mask_tensor], dim=2)

        prompt = prompt or (
            "The pair of images highlights a product and its use in context, high resolution, 4K, 8K;"
            "[IMAGE1] Detailed product shot of the item."
            "[IMAGE2] The same item shown in a realistic lifestyle or usage setting."
        )

        width, height = output_size
        result = self.pipe(
            height=height,
            width=width * 2,
            image=inpaint_image,
            mask_image=extended_mask,
            num_inference_steps=10,
            generator=torch.Generator("cuda").manual_seed(11),
            max_sequence_length=512,
            guidance_scale=30,
            prompt=prompt,
        ).images[0]

        return result.crop((width, 0, width * 2, height))
