import hashlib
import logging
import os

import cv2
import folder_paths
import numpy as np
import torch
from comfy_api.latest import ComfyAPISync, io, ui

from .backend.models import PerceptionInputs, ProcessingParameters
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
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Input image to transform into a Paint-by-Number template. "
                        "Supports batch processing."
                    ),
                ),
                io.Image.Input(
                    "albedo",
                    optional=True,
                    tooltip=(
                        "Optional albedo (flat color) map. Helps improve color accuracy by "
                        "ignoring lighting and shading details."
                    ),
                ),
                io.Image.Input(
                    "segmentation",
                    optional=True,
                    tooltip=(
                        "Optional segmentation map to enforce region boundaries. Highly "
                        "recommended for complex scenes or to protect specific subjects."
                    ),
                ),
                # io.Image.Input("normal", optional=True, tooltip="Optional normal map image."),
                io.Int.Input(
                    "num_colors",
                    default=0,
                    min=0,
                    max=30,
                    tooltip=(
                        "Target number of colors in the final palette. Set to 0 to automatically "
                        "determine the optimal count based on image complexity."
                    ),
                ),
                io.Float.Input(
                    "simplification",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.slider,
                    tooltip=(
                        "Tolerance for smoothing region boundaries. Higher values (up to 2.0) "
                        "create simpler, more 'painterly' shapes, while lower values (down to 0.5) "
                        "preserve fine details."
                    ),
                ),
                io.Boolean.Input(
                    "use_watershed",
                    default=False,
                    tooltip=(
                        "Enables watershed-based segmentation. Recommended for complex images "
                        "with many fine details, though it is slower than the default method."
                    ),
                ),
                io.Combo.Input(
                    "output_mode",
                    options=["colored", "outline", "quantized", "print_svg"],
                    default="colored",
                    tooltip=(
                        "Choose the visualization style: 'colored' for the final template, "
                        "'outline' for line-art, 'quantized' for the posterized image, "
                        "or 'print_svg' for vector output."
                    ),
                ),
                # Phase 5 inputs
                io.Combo.Input(
                    "preset",
                    options=["fast", "balanced", "portrait", "custom"],
                    default="balanced",
                    tooltip=(
                        "Presets for advanced settings. 'portrait' protects faces/skin tones; "
                        "'balanced' is for general use; 'fast' prioritizes speed. "
                        "Select 'custom' to manually adjust advanced parameters below."
                    ),
                ),
                io.Boolean.Input(
                    "use_slic",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Use SLIC clustering for more uniform and shape-consistent "
                        "initial segmentation."
                    ),
                ),
                io.Boolean.Input(
                    "use_ciede2000",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Use the CIEDE2000 formula for perceptually accurate color "
                        "matching and palette merging."
                    ),
                ),
                io.Boolean.Input(
                    "use_palette_merge",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Automatically merges visually similar colors to optimize "
                        "the palette size."
                    ),
                ),
                io.Float.Input(
                    "ciede2000_merge_thresh",
                    default=8.0,
                    min=2.0,
                    max=20.0,
                    step=0.5,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Threshold for merging similar colors. Higher values result "
                        "in a more condensed palette."
                    ),
                ),
                io.Boolean.Input(
                    "use_thin_cleanup",
                    default=True,
                    advanced=True,
                    tooltip="(Advanced) Removes very thin regions that are difficult to paint.",
                ),
                io.Int.Input(
                    "min_region_width",
                    default=5,
                    min=2,
                    max=20,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Minimum pixel width for a region. Smaller regions are "
                        "merged into neighbors."
                    ),
                ),
                io.Boolean.Input(
                    "use_shared_borders",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Uses shared paths in SVG to prevent 'white gaps' between "
                        "regions when rendering."
                    ),
                ),
                io.Combo.Input(
                    "label_mode",
                    options=["centroid", "polylabel"],
                    default="polylabel",
                    advanced=True,
                    tooltip=(
                        "(Advanced) 'polylabel' ensures labels are placed in the most visible "
                        "part of complex shapes; 'centroid' uses the mathematical center."
                    ),
                ),
                io.Boolean.Input(
                    "use_bezier_smooth",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Smooths boundaries using Bezier curves for a more "
                        "professional, hand-drawn look."
                    ),
                ),
                io.Boolean.Input(
                    "use_content_protect",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Uses the segmentation map to prevent important subjects "
                        "from being overly simplified."
                    ),
                ),
                io.Float.Input(
                    "subject_priority",
                    default=2.0,
                    min=1.0,
                    max=5.0,
                    step=0.1,
                    advanced=True,
                    tooltip="(Advanced) Weighting multiplier for the protected subject regions.",
                ),
                io.Float.Input(
                    "material_weight",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Balance between the original photo and the albedo map. "
                        "1.0 uses albedo only, 0.0 uses photo only."
                    ),
                ),
                # io.Float.Input(
                #     "edge_influence", default=0.3, min=0.0, max=1.0, step=0.1, advanced=True
                # ),
            ],
            outputs=[
                io.Image.Output(
                    "IMAGE",
                    tooltip=(
                        "The rendered result (colored template, line-art, or quantized image) "
                        "as a pixel image."
                    ),
                ),
                io.String.Output(
                    "SVG",
                    display_name="SVG Content",
                    tooltip=(
                        "High-quality vector SVG content, ideal for high-resolution printing "
                        "or editing in vector software."
                    ),
                ),
                io.Int.Output(
                    "COLOR_COUNT",
                    display_name="Color Count",
                    tooltip="The final number of colors used in the generated palette.",
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
        albedo=None,
        segmentation=None,
        normal=None,
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
        subject_priority=2.0,
        material_weight=0.5,
        edge_influence=0.3,
    ):
        use_auto_mask = False
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
                use_auto_mask = True
            else:
                raise ValueError(f"Unknown preset: {preset}")

        # Convert torch tensors to numpy BGR for perception inputs
        def torch_to_bgr(t):
            if t is None:
                return None
            # [B, H, W, C] -> [H, W, C] (take first in batch)
            img_np = (t[0].cpu().numpy() * 255).astype(np.uint8)
            if len(img_np.shape) == 2:  # Grayscale
                return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        albedo_np = torch_to_bgr(albedo)
        segmentation_np = None
        if segmentation is not None:
            # For segmentation, we want the raw class labels if possible,
            # but if it's an RGB image we'll use torch_to_bgr and then unique later.
            segmentation_np = (segmentation[0].cpu().numpy() * 255).astype(np.uint8)
            if len(segmentation_np.shape) == 3:
                # Convert RGB to a 1D label map
                segmentation_np = (
                    segmentation_np[:, :, 0].astype(np.uint32) * 65536
                    + segmentation_np[:, :, 1].astype(np.uint32) * 256
                    + segmentation_np[:, :, 2].astype(np.uint32)
                )

        normal_np = torch_to_bgr(normal)

        has_perception = (
            any(x is not None for x in [albedo_np, segmentation_np, normal_np]) or use_auto_mask
        )

        perception = None
        if has_perception:
            perception = PerceptionInputs(
                albedo=albedo_np,
                segmentation_mask=segmentation_np,
                normal_map=normal_np,
                subject_priority=subject_priority,
                material_weight=material_weight,
                edge_influence=edge_influence,
                use_auto_mask=use_auto_mask,
            )

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
            perception=perception,
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

        # Save SVGs to temp directory using content hash to prevent accumulation
        svg_results = []
        temp_dir = folder_paths.get_temp_directory()

        for svg_content in svg_contents:
            # Use MD5 hash of content for deterministic filenames
            content_hash = hashlib.md5(svg_content.encode("utf-8")).hexdigest()[:16]
            svg_filename = f"pbn_{content_hash}.svg"
            filepath = os.path.join(temp_dir, svg_filename)

            # Note: Non-atomic check is acceptable here as writes are idempotent
            if not os.path.exists(filepath):
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(svg_content)

            svg_results.append({"filename": svg_filename, "subfolder": "", "type": "temp"})

        # Prepare UI output with both pixel preview and SVG references
        pixel_preview = ui.PreviewImage(final_image, cls=cls)
        ui_output = {
            "images": pixel_preview.values,  # Verified attribute for V3 PreviewImage
            "pbn_svg": svg_results,
        }

        # Note: The 'SVG' output pin only returns the first SVG in the batch,
        # but the UI preview shows all generated vector graphics.
        return io.NodeOutput(final_image, svg_contents[0], color_counts[0], ui=ui_output)
