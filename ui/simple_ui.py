# # /ui/simple_ui.py
# import gradio as gr
# import torch
# import numpy as np
# import torchvision.transforms as T
#
# def tensor_to_pil(img_tensor):
#     """
#     Converts a torch image tensor (NCHW or CHW) to a PIL image.
#     """
#     if img_tensor.ndim == 4:
#         img_tensor = img_tensor[0]  # Remove batch dimension
#     img = T.ToPILImage()(img_tensor.cpu().clamp(0, 1))
#     return img
#
# def build_app(gan_model):
#     def generate_from_sketch(sketch_img):
#         # Right now, we IGNORE the sketch.
#         # Later, we'll optimize based on sketch.
#         z = gan_model.sample_latent(batch_size=1)
#         img = gan_model.generate(z)
#         img_pil = tensor_to_pil(img)
#         return img_pil
#
#     with gr.Blocks() as app:
#         gr.Markdown("# 🎨 Interactive GAN - Sketch & Generate")
#
#         with gr.Row():
#             with gr.Column():
#                 sketchpad = gr.Sketchpad(
#                     label="🖌️ Draw here!",
#                     shape=(512, 512),
#                     brush_radius=5,
#                     brush_color="#000000",  # Default black
#                     background="#FFFFFF"    # White background
#                 )
#                 brush_size = gr.Slider(minimum=1, maximum=50, value=5, label="✏️ Brush Thickness")
#                 brush_color = gr.ColorPicker(value="#000000", label="🎨 Brush Color")
#
#             with gr.Column():
#                 output = gr.Image(label="🖼️ Generated Output")
#
#         # Update brush settings
#         brush_size.change(lambda s: sketchpad.update(brush_radius=int(s)), inputs=brush_size, outputs=sketchpad)
#         brush_color.change(lambda c: sketchpad.update(brush_color=c), inputs=brush_color, outputs=sketchpad)
#
#         # Generate button
#         generate_btn = gr.Button("✨ Generate from Sketch")
#         generate_btn.click(fn=generate_from_sketch, inputs=[sketchpad], outputs=[output])
#
#     return app


# /ui/simple_ui.py
import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

def tensor_to_pil(img_tensor):
    """
    Converts a torch image tensor (NCHW or CHW) to a PIL image.
    """
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]  # Remove batch dimension
    img = T.ToPILImage()(img_tensor.cpu().clamp(0, 1))
    return img

def build_app(gan_model):
    def generate_from_sketch(sketch_img):
        if sketch_img is None:
            return None

        # Convert NumPy array (from Gradio) to PIL
        if isinstance(sketch_img, np.ndarray):
            sketch_img = Image.fromarray(sketch_img.astype("uint8"))

        # Optimize latent vector
        output_tensor = gan_model.optimize_latent_for_sketch(sketch_img, steps=100, lr=0.05)
        return tensor_to_pil(output_tensor)

    with gr.Blocks() as app:
        gr.Markdown("# 🎨 Interactive GAN - Sketch → Image")

        with gr.Row():
            with gr.Column():
                sketchpad = gr.Sketchpad(
                    label="🖌️ Draw here!",
                    shape=(512, 512),
                    brush_radius=5,
                    brush_color="#000000",
                    background="#FFFFFF"
                )
                brush_size = gr.Slider(minimum=1, maximum=50, value=5, label="✏️ Brush Thickness")
                brush_color = gr.ColorPicker(value="#000000", label="🎨 Brush Color")

            with gr.Column():
                output = gr.Image(label="🖼️ Generated Output")

        # Live brush updates
        brush_size.change(lambda s: sketchpad.update(brush_radius=int(s)), inputs=brush_size, outputs=sketchpad)
        brush_color.change(lambda c: sketchpad.update(brush_color=c), inputs=brush_color, outputs=sketchpad)

        # Generate button
        generate_btn = gr.Button("✨ Generate from Sketch")
        generate_btn.click(fn=generate_from_sketch, inputs=[sketchpad], outputs=[output])

    return app
