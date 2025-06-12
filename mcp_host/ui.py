import gradio as gr

from utils import get_hf_space_file_url_prefix


def UI(products_state: gr.State, image_state: gr.State):
    ui_container = gr.HTML(
        # Placeholder for initial UI Use the image as background with a dark overlay
        f"""<div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 600px;
            width: 100%;
            background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)), url('{get_hf_space_file_url_prefix()}static/welcome-to-vibe-shopping.webp');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
            text-align: center;
            padding: 32px;
        ">
            <h1 style="text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);">Welcome to Vibe Shopping</h1>
            <h3 style="text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);">What the AI assistant want to show you will appear here.</h3>
            <p style="font-size: 1.1rem; margin-top: 16px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);">
                You can talk to the AI assistant to find products, virtual try on clothes, check how products like furniture look in your home, and more.</br>
                You can also upload an image to show to the AI assistant what you are looking for or upload images of yourself to try on clothes.
            </p>
        </div>""",
        container=True,
        max_height=600,
        min_height=450,
        padding=False,
    )

    if products_state.value:
        products_state.change(
            fn=ProductList,
            inputs=[products_state],
            outputs=[ui_container],
        )
    elif image_state.value:
        image_state.change(
            fn=ImageDisplay,
            inputs=[image_state],
            outputs=[ui_container],
        )

    return ui_container


def ProductList(products: list[dict[str, str]]):
    if not products:
        return gr.update(value="")

    cards = ""
    for product in products:
        name = product.get("name", "Unnamed Product")
        price = product.get("price", "N/A")
        image = product.get("image", "")

        cards += f"""
        <div style="
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            width: 180px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        ">
            <img src="{image}" alt="{name}" style="
                width: 100%;
                height: 120px;
                object-fit: cover;
                border-radius: 8px 8px 0 0;
            " />
            <div style="width: 100%; padding: 8px;">
                <h4 style="
                    margin: 0 5px;
                    font-size: 1rem;
                    font-weight: 600;
                    text-align: left;
                ">{name}</h4>
                <p style="
                    margin: 0 8px;
                    margin-top: 0.5rem;
                    color: #2c3e50;
                    font-weight: bold;
                    font-size: 0.8rem;
                    text-align: right;
                ">{price}</p>
            </div>
        </div>
        """

    html = f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
        height: 100%;
    ">
        {cards}
    </div>
    """

    return gr.update(value=html)


def ImageDisplay(image_url: str):
    if not image_url:
        return gr.update(value="")

    # html = FullSizeImage(image_url)
    html = f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        width: 100%;
    ">
        {FullSizeImage(image_url)}
    </div>
    """

    return gr.update(value=html)


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
