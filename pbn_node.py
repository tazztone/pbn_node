import concurrent.futures
import dataclasses
import hashlib
import logging
import os

import cv2
import folder_paths
import numpy as np
import torch
from comfy_api.latest import ComfyAPISync, io

from .backend.models import PerceptionInputs, ProcessingParameters
from .pbn_pipeline import ImageProcessor
from .pbn_renderer import PBNRenderer

# Configure logging
logger = logging.getLogger(__name__)

# Preset configuration table
PRESETS = {
    "fast": {
        "use_ciede2000": True,
        "use_palette_merge": True,
        "use_bezier_smooth": False,
    },
    "balanced": {
        "use_ciede2000": True,
        "use_palette_merge": True,
        "use_thin_cleanup": True,
        "use_shared_borders": True,
        "use_bezier_smooth": False,
        "use_auto_albedo": False,
    },
    "portrait": {
        "use_ciede2000": True,
        "use_palette_merge": True,
        "use_thin_cleanup": True,
        "use_shared_borders": True,
        "use_bezier_smooth": False,
        "use_auto_albedo": True,
    },
}


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
                    tooltip="The source image to transform. Supports batch processing.",
                ),
                io.Image.Input(
                    "segmentation",
                    optional=True,
                    tooltip=(
                        "Optional segmentation map (e.g., from SAM or Mask2Former). This tells the "
                        "node where objects start/end, preventing 'color bleeding' and helping to "
                        "protect important details like faces or hands."
                    ),
                ),
                io.Image.Input(
                    "lineart",
                    optional=True,
                    tooltip=(
                        "Optional edge map (e.g., from HED, SoftEdge, or Canny preprocessors). "
                        "This tells the node where strong visual boundaries are, preventing "
                        "color regions from bleeding across lines. Any single-channel edge "
                        "map works — wire any 'comfyui_controlnet_aux' preprocessor here."
                    ),
                ),
                io.Float.Input(
                    "lineart_strength",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                    tooltip=(
                        "(Advanced) How strongly the lineart edges influence region boundaries. "
                        "Higher values create sharper boundaries at edges but may fragment regions."
                    ),
                ),
                io.Int.Input(
                    "num_colors",
                    default=24,
                    min=0,
                    max=40,
                    tooltip=(
                        "How many unique paint pots you want. Use 0 for 'Auto' mode. "
                        "Beginners should start with 8-12 colors. High-detail projects "
                        "use 24-30."
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
                        "Controls how 'wiggly' the lines are. Higher values (1.5+) make it "
                        "easier to paint but lose detail. Lower values (0.5-0.8) keep the "
                        "photo's shapes more accurately but are much harder to paint."
                    ),
                ),
                io.Combo.Input(
                    "output_mode",
                    options=["colored", "outline", "quantized", "print_svg"],
                    default="colored",
                    tooltip=(
                        "'colored': Template with colors and labels; 'outline': Line-art for "
                        "printing; 'quantized': Posterized test image; 'print_svg': "
                        "High-quality vector file for large printing."
                    ),
                ),
                io.Combo.Input(
                    "segmentation_format",
                    options=["auto", "grayscale", "rgb_packed"],
                    default="auto",
                    tooltip=(
                        "How to interpret the segmentation map. 'auto' tries to detect. "
                        "'grayscale': single-channel 0–N class labels (SAM, Sapiens). "
                        "'rgb_packed': RGB image where each unique color = one class."
                    ),
                ),
                io.Combo.Input(
                    "preset",
                    options=["fast", "balanced", "portrait", "custom"],
                    default="balanced",
                    tooltip=(
                        "Quick settings: 'portrait' protects faces; 'balanced' is a safe default; "
                        "'fast' is for quick previews. Use 'custom' to unlock the manual "
                        "advanced sliders below."
                    ),
                ),
                io.Boolean.Input(
                    "use_ciede2000",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Uses the industry-standard CIEDE2000 formula which matches "
                        "colors how human eyes see them (best for skin tones) rather than "
                        "simple math."
                    ),
                ),
                io.Boolean.Input(
                    "use_palette_merge",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Automatically combines very similar colors (e.g., two "
                        "slightly different greys) into one to optimize your paint kit."
                    ),
                ),
                io.Float.Input(
                    "ciede2000_merge_thresh",
                    default=10.0,
                    min=2.0,
                    max=20.0,
                    step=0.5,
                    advanced=True,
                    tooltip=(
                        "(Advanced) How aggressive to be when merging similar colors. Higher "
                        "values result in a smaller, more condensed palette. Default 10.0."
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
                        "(Advanced) Minimum pixel size for a region. Smaller bits will be "
                        "merged into neighbors to prevent 'confetti' noise in your template."
                    ),
                ),
                io.Boolean.Input(
                    "use_shared_borders",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Uses shared paths in SVG to prevent 'white gaps' between regions when rendering."
                    ),
                ),
                io.Combo.Input(
                    "label_mode",
                    options=["centroid", "polylabel"],
                    default="polylabel",
                    advanced=True,
                    tooltip=(
                        "(Advanced) 'polylabel' ensures numbers are placed in the widest part "
                        "of complex shapes; 'centroid' uses the exact mathematical center."
                    ),
                ),
                io.Boolean.Input(
                    "use_bezier_smooth",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Converts jagged pixel edges into smooth, flowing curves. "
                        "Gives the template a professional, hand-drawn look."
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
                    tooltip=("1.0 uses pure albedo (flattest look); 0.5 blends them for balance."),
                ),
                io.Float.Input(
                    "edge_influence",
                    default=0.3,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Weight of lineart edges on color quantization. High values "
                        "ensure color boundaries strictly follow edges."
                    ),
                ),
                io.Boolean.Input(
                    "use_auto_albedo",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Automatically estimates shadow-free colors using Retinex. "
                        "Useful for portraits with harsh lighting. Only activates when no "
                        "external Albedo map is wired."
                    ),
                ),
                io.Int.Input(
                    "smoothing_kernel_size",
                    default=9,
                    min=3,
                    max=21,
                    step=2,
                    advanced=True,
                    tooltip="(Advanced) Size of the majority smoothing kernel (must be odd).",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "IMAGE",
                    tooltip=("The rendered template (colored, outline, or quantized) as a pixel image."),
                ),
                io.String.Output(
                    "SVG",
                    display_name="SVG Content",
                    tooltip=(
                        "High-quality vector SVG file, ideal for large-format printing or professional vector editing."
                    ),
                ),
                io.Int.Output(
                    "COLOR_COUNT",
                    display_name="Color Count",
                    tooltip="The total number of unique paint colors required for this template.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        num_colors=24,
        simplification=1.0,
        output_mode="colored",
        preset="balanced",
        segmentation=None,
        lineart=None,
        lineart_strength=0.7,
        invert_lineart=False,
        segmentation_format="auto",
        use_ciede2000=True,
        use_palette_merge=True,
        ciede2000_merge_thresh=10.0,
        use_thin_cleanup=True,
        min_region_width=5,
        use_shared_borders=True,
        label_mode="polylabel",
        use_bezier_smooth=False,
        subject_priority=2.0,
        material_weight=0.5,
        edge_influence=0.3,
        use_auto_albedo=False,
        smoothing_kernel_size=9,
    ):
        # 1. Build ProcessingParameters directly from declared args
        proc_params = ProcessingParameters(
            num_colors=num_colors if num_colors > 0 else None,
            simplification=simplification,
            use_ciede2000=use_ciede2000,
            use_palette_merge=use_palette_merge,
            ciede2000_merge_thresh=ciede2000_merge_thresh,
            use_thin_cleanup=use_thin_cleanup,
            min_region_width=min_region_width,
            use_shared_borders=use_shared_borders,
            label_mode=label_mode,
            use_bezier_smooth=use_bezier_smooth,
            preset=preset,
            output_mode=output_mode,
            use_auto_albedo=use_auto_albedo,
            smoothing_kernel_size=smoothing_kernel_size,
        )

        # 2. Apply preset overrides via dataclasses.replace
        proc_params = cls._apply_preset(proc_params, preset)

        # 3. Setup batch processing
        batch_size = image.shape[0]
        result_images = []
        svg_contents = []
        color_counts = []

        api = ComfyAPISync()
        processor = ImageProcessor()
        renderer = PBNRenderer()

        # 4. Batch loop
        for i in range(batch_size):
            if batch_size > 1:
                logger.info(f"Processing image {i + 1}/{batch_size}")

            # Slice the batch tensors to the i-th frame
            img_i = image[i]
            lineart_i = lineart[i : i + 1] if lineart is not None else None
            seg_i = segmentation[i : i + 1] if segmentation is not None else None

            result_tensor, svg_content, color_count = cls._execute_single(
                img_i,
                lineart_i,
                seg_i,
                processor,
                renderer,
                api,
                proc_params,
                lineart_strength=lineart_strength,
                subject_priority=subject_priority,
                material_weight=material_weight,
                edge_influence=edge_influence,
                segmentation_format=segmentation_format,
                invert_lineart=invert_lineart,
            )

            result_images.append(result_tensor)
            svg_contents.append(svg_content)
            color_counts.append(color_count)

        # 5. Finalize outputs
        final_image = torch.stack(result_images, dim=0)
        svg_results = cls._save_svg_batch(svg_contents)

        ui_output = {
            "pbn_svg": svg_results,
        }

        # Handle single vs batch for non-tensor outputs
        out_svg = svg_contents if batch_size > 1 else svg_contents[0]
        out_colors = color_counts if batch_size > 1 else color_counts[0]

        return io.NodeOutput(final_image, out_svg, out_colors, ui=ui_output)

    @classmethod
    def _execute_single(
        cls,
        img_tensor,
        lineart_t,
        seg_t,
        processor,
        renderer,
        api,
        proc_params,
        lineart_strength,
        subject_priority,
        material_weight,
        edge_influence,
        segmentation_format,
        invert_lineart,
    ):
        """Processes a single image frame."""
        h, w, _ = img_tensor.shape

        # Decode per-frame perception inputs
        perception_i = cls._prepare_perception_inputs_for_frame(
            lineart_t,
            seg_t,
            lineart_strength=lineart_strength,
            subject_priority=subject_priority,
            material_weight=material_weight,
            edge_influence=edge_influence,
            segmentation_format=segmentation_format,
            invert_lineart=invert_lineart,
        )

        # Build per-frame params with frame-specific perception
        proc_params_i = dataclasses.replace(proc_params, perception=perception_i)

        # Convert to OpenCV BGR
        img_bgr = cls._torch_to_bgr(img_tensor)

        # Process
        result = processor.process_array(img_bgr, proc_params_i, api=api)

        # Render keyed exactly on the frame-specific output_mode
        output_mode_resolved = proc_params_i.output_mode
        if output_mode_resolved == "quantized":
            result_bgr = result.quantized
        else:
            result_bgr = renderer.render(
                result.cleaned_regions,
                result.label_data,
                result.color_palette,
                w,
                h,
                mode=output_mode_resolved,
                region_colors=result.region_colors,
                shared_borders=result.shared_borders,
                use_shared_borders=proc_params_i.use_shared_borders,
            )

        # Convert back to torch RGB
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0)

        return result_tensor, result.svg_content, result.color_palette.color_count

    @classmethod
    def _apply_preset(cls, params: ProcessingParameters, preset: str) -> ProcessingParameters:
        """Applies preset overrides directly to a ProcessingParameters dataclass."""
        if preset != "custom" and preset in PRESETS:
            overrides = PRESETS[preset]
            return dataclasses.replace(params, preset=preset, **overrides)
        return dataclasses.replace(params, preset=preset)

    @classmethod
    def _prepare_perception_inputs_for_frame(
        cls,
        lineart_t: torch.Tensor | None,
        seg_t: torch.Tensor | None,
        lineart_strength: float,
        subject_priority: float,
        material_weight: float,
        edge_influence: float,
        segmentation_format: str,
        invert_lineart: bool,
    ) -> PerceptionInputs | None:
        """Decodes frame-specific input tensors into the PerceptionInputs structure."""
        lineart_np = cls._decode_lineart(lineart_t, invert_lineart)
        segmentation_np = cls._decode_segmentation(seg_t, segmentation_format)

        has_perception = segmentation_np is not None or lineart_np is not None

        if not has_perception:
            return None

        return PerceptionInputs(
            albedo=None,  # Handled internally in pipeline now
            segmentation_mask=segmentation_np,
            lineart=lineart_np,
            lineart_strength=lineart_strength,
            subject_priority=subject_priority,
            material_weight=material_weight,
            edge_influence=edge_influence,
        )

    @staticmethod
    def _torch_to_bgr(t: torch.Tensor | None) -> np.ndarray | None:
        """Converts [H,W,C] or [1,H,W,C] torch RGB float32 to BGR uint8."""
        if t is None:
            return None
        # Handle both single image and batch-of-1
        arr = t[0] if t.ndim == 4 else t
        img_np = (arr.cpu().numpy() * 255).astype(np.uint8)
        if img_np.ndim == 2:
            return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _decode_lineart(t: torch.Tensor | None, invert: bool) -> np.ndarray | None:
        """Decodes Lineart map tensor to [0, 1] grayscale float32."""
        if t is None:
            return None
        arr = t[0].cpu().numpy()
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2)
        lineart = arr.astype(np.float32)
        vmin, vmax = lineart.min(), lineart.max()
        if vmax - vmin > 1e-6:
            lineart = (lineart - vmin) / (vmax - vmin)
        return 1.0 - lineart if invert else lineart

    @staticmethod
    def _decode_segmentation(t: torch.Tensor | None, fmt: str) -> np.ndarray | None:
        """Decodes Segmentation map tensor based on format."""
        if t is None:
            return None
        seg_arr = (t[0].cpu().numpy() * 255).astype(np.uint8)
        if fmt == "auto":
            fmt = "grayscale" if seg_arr.ndim == 2 or seg_arr.shape[-1] == 1 else "rgb_packed"

        if fmt == "grayscale":
            res = seg_arr[:, :, 0] if seg_arr.ndim == 3 else seg_arr
            return res.astype(np.int32)

        # rgb_packed
        return (
            seg_arr[:, :, 0].astype(np.uint32) * 65536
            + seg_arr[:, :, 1].astype(np.uint32) * 256
            + seg_arr[:, :, 2].astype(np.uint32)
        )

    @staticmethod
    def _write_single_svg(args: tuple[str, str]) -> dict[str, str]:
        content, temp_dir = args
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        filename = f"pbn_{content_hash}.svg"
        filepath = os.path.join(temp_dir, filename)

        # Only write if file doesn't exist to avoid redundant I/O
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

        return {"filename": filename, "subfolder": "", "type": "temp"}

    @staticmethod
    def _save_svg_batch(svg_contents: list[str]) -> list[dict[str, str]]:
        """Saves SVGs to temp directory using content-addressed hashing and a thread pool."""
        temp_dir = folder_paths.get_temp_directory()
        args_list = [(content, temp_dir) for content in svg_contents]

        # Use a ThreadPoolExecutor to prevent synchronous I/O blocking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            svg_results = list(executor.map(PaintByNumberNode._write_single_svg, args_list))

        return svg_results
