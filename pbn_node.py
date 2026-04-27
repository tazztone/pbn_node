import torch
import numpy as np
import cv2
from comfy_api.latest import io, ui

from .pbn_pipeline import ImageProcessor
from .pbn_renderer import PBNRenderer
from .backend.models import ProcessingParameters

class PaintByNumberNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PaintByNumberNode",
            display_name="Paint By Number",
            category="image/process",
            description="Transforms an image into a paint-by-number template.",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("num_colors", default=0, min=0, max=30, description="Number of colors. 0 for auto."),
                io.Float.Input("simplification", default=1.0, min=0.5, max=2.0, step=0.1, description="Contour simplification tolerance."),
                io.Boolean.Input("use_watershed", default=False, description="Use watershed segmentation (slower but spec-accurate)."),
                io.Combo.Input("output_mode", options=["colored", "outline", "quantized"], default="colored", description="Output mode."),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
                io.String.Output("SVG"),
                io.Int.Output("COLOR_COUNT"),
            ],
        )

    @classmethod
    def execute(cls, image, num_colors, simplification, use_watershed, output_mode):
        # image is [B, H, W, C] RGB float32
        # We process only the first image in batch for now
        img_tensor = image[0]
        h, w, c = img_tensor.shape
        
        # Convert torch tensor to OpenCV BGR
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Setup processor and params
        processor = ImageProcessor()
        params = ProcessingParameters(
            num_colors=num_colors if num_colors > 0 else None,
            simplification=simplification,
            use_watershed=use_watershed
        )
        
        # Process image
        result = processor.process_array(img_bgr, params)
        
        # Render result
        renderer = PBNRenderer()
        
        if output_mode == "quantized":
            # Just return the quantized image
            result_bgr = processor.last_quantized
        else:
            result_bgr = renderer.render(
                processor.last_cleaned_regions,
                processor.last_label_data,
                processor.last_palette,
                w, h,
                mode=output_mode
            )
        
        # Convert back to torch tensor [1, H, W, 3] RGB float32
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        
        return io.NodeOutput(
            result_tensor,
            result.svg_content,
            result.color_palette.color_count
        )
