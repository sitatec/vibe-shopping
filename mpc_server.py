import requests
import base64

from mcp.server.fastmcp import FastMCP
import modal

mcp = FastMCP("Fewsats MCP Server")
_virtual_try_model = None


def virtual_try_model():
    """Get or create an VirtualTryModel instance.
    We want to create the class instance inside the tool,
    so the init errors will bubble up to the tool and hence the MCP client instead of silently failing
    during the server creation.
    """
    global _virtual_try_model
    if _virtual_try_model is None:
        virtual_try_model_class = modal.Cls.from_name(
            "vibe-shopping", "VirtualTryModel"
        )
        _virtual_try_model = virtual_try_model_class()

    return _virtual_try_model


def bytes_to_base64_image(result_image_bytes: bytes, format: str = "webp") -> str:
    base64_image = base64.b64encode(result_image_bytes).decode("utf-8")
    return f"data:image/{format};base64,{base64_image}"


@mcp.tool()
def try_item_with_masking(
    prompt: str,
    item_image_url: str,
    target_image_url: str,
    mask_image_url: str,
) -> str:
    """
    Try an item on a target image by inpainting (with a diffusion model) the item onto the target image using a provided mask.

    For example, if the item is a purple skirt, the target image is a woman standing, you could use this prompt:
    The pair of images highlights a skirt and its fit on a woman, high resolution, 4K, 8K;
    [IMAGE1] Detailed product shot of a purple skirt
    [IMAGE2] The same skirt is worn by a woman standing in a realistic lifestyle setting, the skirt fits naturally.

    Args:
        prompt: A prompt for the diffusion model to use for inpainting.
        item_image_url: URL of the item image to try.
        target_image_url: URL of the target image where the item will be tried.
        mask_image_url: Optional URL of a mask image to use.

    Returns:
        The image where the item is applied to the target image in the base64 data URL format.
    """
    item_image_bytes = requests.get(item_image_url).content
    target_image_bytes = requests.get(target_image_url).content
    mask_image_bytes = requests.get(mask_image_url).content

    result_image_bytes: bytes = virtual_try_model().try_it(
        prompt=prompt,
        image_bytes=target_image_bytes,
        item_to_try_bytes=item_image_bytes,
        mask_bytes=mask_image_bytes,
    )

    # The virtual_try_model return a webp image
    return bytes_to_base64_image(result_image_bytes, format="webp")


@mcp.tool()
def try_item_with_auto_masking(
    prompt: str,
    item_image_url: str,
    target_image_url: str,
    masking_prompt: str,
) -> str:
    """
    Try an item on a target image by inpainting the item onto the target image using an auto-generated mask based on the masking_prompt.
    For example, if the item is a wheel, and the target image a car, so the masking prompt would be "wheel" and the prompt should be something like:
    The pair of images highlights a wheel and its fit on a car, high resolution, 4K, 8K;
    [IMAGE1] Detailed product shot of a wheel
    [IMAGE2] The same wheel is shown on a car in a realistic lifestyle setting.
    
    If the item is a sofa and the target image is a living room containing a yellow sofa, the masking prompt could be "yellow sofa" and the prompt could be:
    The pair of images highlights a yellow sofa and how it fits in a living room, high resolution, 4K, 8K;
    [IMAGE1] Detailed product shot of a yellow sofa
    [IMAGE2] The same sofa is shown in a living room in a realistic lifestyle setting, the sofa fits in naturally with the room decor.

    This tool requires a similar item to be present in the target image, so it can generate a mask of the item using the masking_prompt.

    Args:
        prompt: A prompt for the diffusion model to use for inpainting.
        item_image_url: URL of the item image to try.
        target_image_url: URL of the target image where the item will be tried.
        masking_prompt: Prompt for generating a mask of the corresponding item in the target image. It need to be short and descriptive, e.g. "red dress", "blue sofa", "tire", etc.

    Returns:
        The image where the item is applied to the target image in the base64 data URL format.
    """
    item_image_bytes = requests.get(item_image_url).content
    target_image_bytes = requests.get(target_image_url).content
    virtual_try = virtual_try_model()

    result_image_bytes: bytes = virtual_try.try_it(
        prompt=prompt,
        image_bytes=target_image_bytes,
        item_to_try_bytes=item_image_bytes,
        masking_prompt=masking_prompt,
    )

    # The virtual_try_model return a webp image
    return bytes_to_base64_image(result_image_bytes, format="webp")


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")
