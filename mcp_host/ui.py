import gradio as gr


def ProductList(products_state: gr.State) -> gr.HTML:
    product_html = gr.HTML()

    def render_products(products: list[dict[str, str]]):

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
            display: flex;
            flex-wrap: nowrap;
            gap: 1rem;
            overflow-x: auto;
            padding: 1rem 0;
            max-height: 320px;
        ">
            {cards}
        </div>
        """

        return gr.update(value=html)

    products_state.change(
        fn=render_products,
        inputs=[products_state],
        outputs=[product_html],
        show_progress="hidden",
        queue=False,
    )

    return product_html
