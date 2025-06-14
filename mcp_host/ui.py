import gradio as gr

from utils import get_hf_space_file_url_prefix


def WelcomeUI():
    # Placeholder for initial UI Use the image as background with a dark overlay
    return f"""<div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 600px;
            width: 100%;
            background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)), url('{get_hf_space_file_url_prefix()}static/welcome-to-vibe-shopping-upscaled.webp');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
            text-align: center;
            padding: 32px;
        ">
            <h1 style="text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7); color: white;">Welcome to Vibe Shopping</h1>
            <h3 style="text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7); color: white;">What the AI assistant want to show you will appear here.</h3>
            <p style="font-size: 1.1rem; margin-top: 16px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7); color: white;">
                You can talk to the AI assistant to find products, virtual try on clothes, check how products like furniture look in your home, and more.</br>
                You can also upload an image to show to the AI assistant what you are looking for or upload images of yourself to try on clothes.
            </p>
        </div>"""


def ProductList(products: list[dict[str, str]]):
    print("Rendering product list with", len(products), "products")

    if not products:
        return """
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
                width: 100%;
            ">
                <p style="font-size: 1.2rem; text-align: center;">Empty products list. Try searching for something!</p>
            </div>
        """

    cards = ""
    for product in products:
        name = product.get("name", "Unnamed Product")
        price = product.get("price", "N/A")
        image = product.get("image_url", "")

        cards += f"""
        <div style="
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            width: 200px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        ">
            <img src="{image}" alt="{name}" style="
                width: 100%;
                height: 160px;
                object-fit: contain;
                border-radius: 8px 8px 0 0;
            " />
            <div style="width: 100%; padding: 8px; display: flex; flex-direction: column; align-items: center; justify-content: space-between; height: 50px;">
                <h5 style="
                    opacity: 0.7;
                    margin: 0 5px;
                    text-align: center;
                    text-overflow: ellipsis;
                    -webkit-line-clamp: 2; display: -webkit-box; -webkit-box-orient: vertical;
                ">{name}</h5>
                <p style="
                    margin: 0 8px;
                    margin-top: 0.5rem;
                    font-weight: bold;
                    font-size: 0.82rem;
                ">{price}</p>
            </div>
        </div>
        """

    return f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.8rem;
        padding: 1rem 0;
        height: 600px;
        width: 100%;
        overflow-y: auto;
        align-items: center;
        justify-items: center;
    ">
        {cards}
    </div>
    """


def ImageDisplay(image_url: str):
    if not image_url:
        return ""

    return f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        height: 90vh;
        width: 100%;
    ">
        {FullSizeImage(image_url)}
    </div>
    """


def FullSizeImage(image_url, fit: str = "contain", border_radius=12) -> str:
    return f"""
        <img src="{image_url}" alt="Vibe Shopping Image" style="
            max-width: 100%;
            max-height: 100%;
            border-radius: {border_radius}px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            object-fit: {fit};
        " />
    """


def ColdBootUI():
    return gr.HTML(
        """
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            width: 100%;
            text-align: center;
        ">
            <h2>Starting Inference Server...</h2>
            <p style="font-size: 1.2rem; margin-top: 16px;">
                If this happen to be a cold-boot, it may take up to 2 minutes. Please wait...
            </p>
        </div>
        """,
    )
