import logging

import cv2
import numpy as np
import torch
from comfy_api.latest import ComfyAPISync, io, ui

from .backend.models import ProcessingParameters
from .pbn_pipeline import ImageProcessor
from .pbn_renderer import PBNRenderer

# Configure logging
logger = logging.getLogger(__name__)


class PaintByNumberNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PaintByNumberNode",
            display_name="Paint By Number",
            category="image/process",
            description="Transforms an image into a paint-by-number template.",
            is_output_node=True,
            inputs=[
                io.Image.Input("image", tooltip="Input image(s) to process."),
                io.Int.Input(
                    "num_colors",
                    default=0,
                    min=0,
                    max=30,
                    tooltip="Number of colors in the palette. Use 0 for automatic detection.",
                ),
                io.Float.Input(
                    "simplification",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.slider,
                    tooltip=(
                        "Contour simplification tolerance. "
                        "Higher values mean smoother but less detailed shapes."
                    ),
                ),
                io.Boolean.Input(
                    "use_watershed",
                    default=False,
                    tooltip=(
                        "Use watershed segmentation. Slower but more accurate for complex images."
                    ),
                ),
                io.Combo.Input(
                    "output_mode",
                    options=["colored", "outline", "quantized", "print_svg"],
                    default="colored",
                    tooltip="Choose the visualization style of the result.",
                ),
                # Phase 5 inputs
                io.Combo.Input(
                    "preset",
                    options=["fast", "balanced", "portrait", "custom"],
                    default="balanced",
                    tooltip="Presets for advanced settings. Select 'custom' to use the manual advanced parameters below.",
                ),
                io.Boolean.Input("use_slic", default=True, advanced=True),
                io.Boolean.Input("use_ciede2000", default=True, advanced=True),
                io.Boolean.Input("use_palette_merge", default=True, advanced=True),
                io.Float.Input(
                    "ciede2000_merge_thresh",
                    default=8.0,
                    min=2.0,
                    max=20.0,
                    step=0.5,
                    advanced=True,
                ),
                io.Boolean.Input("use_thin_cleanup", default=True, advanced=True),
                io.Int.Input("min_region_width", default=5, min=2, max=20, advanced=True),
                io.Boolean.Input("use_shared_borders", default=True, advanced=True),
                io.Combo.Input(
                    "label_mode",
                    options=["centroid", "polylabel"],
                    default="polylabel",
                    advanced=True,
                ),
                io.Boolean.Input("use_bezier_smooth", default=False, advanced=True),
                io.Boolean.Input("use_content_protect", default=False, advanced=True),
                io.Boolean.Input("use_budget_split", default=False, advanced=True),
            ],
            outputs=[
                io.Image.Output("IMAGE", tooltip="The rendered paint-by-number image."),
                io.String.Output(
                    "SVG",
                    display_name="SVG Content",
                    tooltip="The vector representation in SVG format.",
                ),
                io.Int.Output(
                    "COLOR_COUNT",
                    display_name="Color Count",
                    tooltip="The final number of colors in the palette.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        num_colors,
        simplification,
        use_watershed,
        output_mode,
        preset="balanced",
        use_slic=True,
        use_ciede2000=True,
        use_palette_merge=True,
        ciede2000_merge_thresh=8.0,
        use_thin_cleanup=True,
        min_region_width=5,
        use_shared_borders=True,
        label_mode="polylabel",
        use_bezier_smooth=False,
        use_content_protect=False,
        use_budget_split=False,
    ):
        # Apply presets
        if preset != "custom":
            if preset == "fast":
                use_slic = False
                use_ciede2000 = True
                use_palette_merge = True
                use_bezier_smooth = False
            elif preset == "balanced":
                use_slic = True
                use_ciede2000 = True
                use_palette_merge = True
                use_thin_cleanup = True
                use_shared_borders = True
                use_bezier_smooth = False
                use_content_protect = False
            elif preset == "portrait":
                use_slic = True
                use_ciede2000 = True
                use_palette_merge = True
                use_thin_cleanup = True
                use_shared_borders = True
                use_bezier_smooth = False
                use_content_protect = True
                use_budget_split = True
            else:
                raise ValueError(f"Unknown preset: {preset}")

        # image is [B, H, W, C] RGB float32
        batch_size = image.shape[0]
        result_images = []
        svg_contents = []
        color_counts = []

        api = ComfyAPISync()
        processor = ImageProcessor()
        renderer = PBNRenderer()

        params = ProcessingParameters(
            num_colors=num_colors if num_colors > 0 else None,
            simplification=simplification,
            use_watershed=use_watershed,
            use_slic=use_slic,
            use_ciede2000=use_ciede2000,
            use_palette_merge=use_palette_merge,
            ciede2000_merge_thresh=ciede2000_merge_thresh,
            use_thin_cleanup=use_thin_cleanup,
            min_region_width=min_region_width,
            use_shared_borders=use_shared_borders,
            label_mode=label_mode,
            use_bezier_smooth=use_bezier_smooth,
            use_content_protect=use_content_protect,
            use_budget_split=use_budget_split,
            preset=preset,
            output_mode=output_mode,
        )

        for i in range(batch_size):
            # Report which image we are processing if batch_size > 1
            if batch_size > 1:
                logger.info(f"Processing image {i + 1}/{batch_size}")

            img_tensor = image[i]
            h, w, c = img_tensor.shape

            # Convert torch tensor to OpenCV BGR
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Process image with progress reporting
            result = processor.process_array(img_bgr, params, api=api)

            # Render result
            if output_mode == "quantized":
                result_bgr = processor.last_quantized
            else:
                result_bgr = renderer.render(
                    processor.last_cleaned_regions,
                    processor.last_label_data,
                    processor.last_palette,
                    w,
                    h,
                    mode=output_mode,
                )

            # Convert back to torch tensor [H, W, 3] RGB float32
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0)

            result_images.append(result_tensor)
            svg_contents.append(result.svg_content)
            color_counts.append(result.color_palette.color_count)

        # Stack results back into [B, H, W, 3]
        final_image = torch.stack(result_images, dim=0)

        # In ComfyUI V3, for non-image outputs that are lists, we return the list if the
        # schema says so.
        # But here we didn't mark SVG or COLOR_COUNT as list outputs.
        # For simplicity, if batch > 1, we return the first one or a combined string.
        # However, usually SVG results for batches should probably be handled carefully.
        # For now, let's return the first one as per previous behavior but supporting
        # batch for images.

        return io.NodeOutput(
            final_image, svg_contents[0], color_counts[0], ui=ui.PreviewImage(final_image, cls=cls)
        )
