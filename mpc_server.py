from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Fewsats MCP Server")


@mcp.tool()
def try_item_with_masking(
    item_image_url: str,
    target_image_url: str,
    mask_image_url: str,
) -> str:
    """
    Try an item on a target image by inpainting the item onto the target image using a provided mask.

    Args:
        item_image_url: URL of the item image to try.
        target_image_url: URL of the target image where the item will be tried.
        mask_image_url: Optional URL of a mask image to use.

    Returns:
        The url of the image where the item is applied to the target image,
    """
    # Here you would implement the logic to process the images and apply the item
    # For now, we return a placeholder message
    return f"Trying item from {item_image_url} on target {target_image_url} with mask {mask_image_url}."


@mcp.tool()
def try_item_with_auto_masking(
    item_image_url: str,
    target_image_url: str,
    masking_prompt: str,
) -> str:
    """
    Try an item on a target image by inpainting the item onto the target image using an auto-generated mask based on the masking_prompt.
    For example, the item could be a yellow shirt, and the target image could be a person wearing a green shirt, so the masking prompt could be "shirt" or "green shirt".
    Or the item could be a tire, and the target image a car, so the masking prompt would be "tire".

    This tool requires a similar item to be present in the target image, so it can generate a mask of the item using the masking_prompt.

    Args:
        item_image_url: URL of the item image to try.
        target_image_url: URL of the target image where the item will be tried.
        masking_prompt: Prompt for generating a mask of the corresponding item in the target image. It need to be short and descriptive, e.g. "red dress", "blue sofa", "wheels", etc.

    Returns:
        The url of the image where the item is applied to the target image,
    """
    # Here you would implement the logic to process the images and apply the item
    # For now, we return a placeholder message
    return f"Trying item from {item_image_url} on target {target_image_url} with auto-generated mask from prompt '{masking_prompt}'."


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")
