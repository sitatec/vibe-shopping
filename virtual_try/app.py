from io import BytesIO
from typing import Callable, cast

import modal
from configs import image, modal_class_config

with image.imports():
    import torch
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    from diffusers import FluxFillPipeline
    from nunchaku import NunchakuFluxTransformer2dModel
    from nunchaku.utils import get_precision
    from nunchaku.lora.flux.compose import compose_lora

    from auto_masker import AutoInpaintMaskGenerator

TransformType = Callable[[Image.Image | np.ndarray], torch.Tensor]

app = modal.App("vibe-shopping-virtual-try")


@app.cls(**modal_class_config, max_containers=1)
class VirtualTryModel:
    @modal.web_endpoint(method="GET")
    def health_check(self) -> str:
        return "Virtual Try Model is healthy!"

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

    def _get_preprocessors(
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

    def _bytes_to_image(self, byte_stream: bytes, mode: str = "RGB") -> Image.Image:
        """Convert bytes to PIL Image."""
        return Image.open(BytesIO(byte_stream)).convert(mode)

    @modal.method()
    def try_it(
        self,
        image_bytes: bytes,
        item_to_try_bytes: bytes,
        mask_bytes: bytes | None = None,
        prompt: str | None = None,
        masking_prompt: str | None = None,
    ) -> bytes:
        # We are using bytes for images for serialization/deserialization
        # during Modal function calls.
        assert mask_bytes or masking_prompt, (
            "Either mask or masking_prompt must be provided."
        )

        image = self._bytes_to_image(image_bytes)
        item_to_try = self._bytes_to_image(item_to_try_bytes)

        if mask_bytes:
            mask = self._bytes_to_image(mask_bytes, mode="L")
        else:
            mask = self.auto_masker.generate_mask(
                image,
                prompt=masking_prompt,  # type: ignore
            )

        preprocessor, mask_preprocessor, output_size = self._get_preprocessors(
            input_size=image.size,
            target_megapixels=0.7,  # The image will be stacked which will double the pixel count
        )

        image_tensor = preprocessor(image.convert("RGB"))
        item_to_try_tensor = preprocessor(item_to_try.convert("RGB"))
        mask_tensor = mask_preprocessor(mask)

        # Create concatenated images along the width axis
        inpaint_image = torch.cat([item_to_try_tensor, image_tensor], dim=2)
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

        output_image = result.crop((width, 0, width * 2, height))
        byte_stream = BytesIO()
        output_image.save(byte_stream, format="WEBP", quality=90)
        return byte_stream.getvalue()


###### ------ FOR TESTING PURPOSES ONLY ------ ######
@app.local_entrypoint()
def main(twice: bool = True):
    import time
    from pathlib import Path

    test_data_dir = Path(__file__).parent / "test_data"
    with open(test_data_dir / "target_image.jpg", "rb") as f:
        target_image_bytes = f.read()
    with open(test_data_dir / "item_to_try.jpg", "rb") as f:
        item_to_try_bytes = f.read()
    with open(test_data_dir / "item_to_try2.png", "rb") as f:
        item_to_try_2_bytes = f.read()

    prompt = (
        "The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; "
        "[IMAGE1] Detailed product shot of a clothing"
        "[IMAGE2] The same cloth is worn by a model in a lifestyle setting."
    )

    t0 = time.time()
    image_bytes = VirtualTryModel().try_it.remote(
        prompt=prompt,
        image_bytes=target_image_bytes,
        item_to_try_bytes=item_to_try_bytes,
        masking_prompt="t-shirt, arms, neck",
    )
    output_path = test_data_dir / "output1.jpg"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_bytes(image_bytes)
    print(f"🎨 first inference latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image_bytes = VirtualTryModel().try_it.remote(
            prompt=prompt,
            image_bytes=target_image_bytes,
            item_to_try_bytes=item_to_try_2_bytes,
            masking_prompt="t-shirt, arms",
        )
        print(f"🎨 second inference latency: {time.time() - t0:.2f} seconds")

        output_path = test_data_dir / "output2.jpg"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        output_path.write_bytes(image_bytes)
