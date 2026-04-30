import hashlib
import logging
import os
from typing import Any

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
        "use_slic": False,
        "use_ciede2000": True,
        "use_palette_merge": True,
        "use_bezier_smooth": False,
    },
    "balanced": {
        "use_slic": True,
        "use_ciede2000": True,
        "use_palette_merge": True,
        "use_thin_cleanup": True,
        "use_shared_borders": True,
        "use_bezier_smooth": False,
        "use_content_protect": False,
        "use_auto_albedo": True,
    },
    "portrait": {
        "use_slic": True,
        "use_ciede2000": True,
        "use_palette_merge": True,
        "use_thin_cleanup": True,
        "use_shared_borders": True,
        "use_bezier_smooth": False,
        "use_content_protect": True,
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
                io.Boolean.Input(
                    "invert_lineart",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Enable if your preprocessor outputs white lines on black "
                        "background (e.g., standard Canny)."
                    ),
                ),
                io.Image.Input(
                    "normals",
                    optional=True,
                    tooltip=(
                        "Optional surface normal map (from Sapiens or a Depth-to-Normal node). "
                        "Causes superpixel boundaries to align with 3D surface creases (shoulder "
                        "edges, facial structure) even when colors on both sides are similar. "
                        "Expects a standard RGB normal map where R=X, G=Y, B=Z in [-1,1] range."
                    ),
                ),
                io.Float.Input(
                    "normal_strength",
                    default=0.4,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                    tooltip=(
                        "(Advanced) How strongly the normal map influences superpixel shapes. "
                        "0 = pure color-based SLIC; 1 = normals dominate shape. "
                        "0.3–0.5 is a good starting range for portraits."
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
                io.Boolean.Input(
                    "use_watershed",
                    default=False,
                    tooltip=(
                        "An alternative segmentation method. Use this if the default results "
                        "in 'messy' shapes. It is slower but handles busy, complex backgrounds "
                        "much better."
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
                    "use_slic",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Uses SLIC clustering to group pixels into natural, organic "
                        "blocks. This usually creates more 'aesthetic' shapes than raw "
                        "color-based grouping."
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
                io.Boolean.Input(
                    "use_content_protect",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Requires a segmentation map. It 'shields' the subject "
                        "from being overly simplified, keeping it recognizable."
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
                io.Boolean.Input(
                    "use_auto_mask",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Automatically generates a segmentation mask for portrait "
                        "protection when none is provided."
                    ),
                ),
                io.Boolean.Input(
                    "use_painterly_preprocess",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "(Advanced) Applies a 'painterly' filter to the image before processing. "
                        "This helps simplify textures and can lead to cleaner color regions."
                    ),
                ),
                io.Float.Input(
                    "painterly_sigma_s",
                    default=60.0,
                    min=10.0,
                    max=200.0,
                    advanced=True,
                    tooltip="(Advanced) Spatial sigma for the painterly filter.",
                ),
                io.Float.Input(
                    "painterly_sigma_r",
                    default=0.45,
                    min=0.1,
                    max=1.0,
                    advanced=True,
                    tooltip="(Advanced) Range sigma for the painterly filter.",
                ),
                io.Int.Input(
                    "slic_n_segments",
                    default=500,
                    min=100,
                    max=5000,
                    advanced=True,
                    tooltip="(Advanced) Number of superpixel segments for SLIC clustering.",
                ),
                io.Float.Input(
                    "slic_compactness",
                    default=10.0,
                    min=0.01,
                    max=100.0,
                    advanced=True,
                    tooltip="(Advanced) Compactness factor for SLIC clustering.",
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
        use_watershed=False,
        output_mode="colored",
        preset="balanced",
        segmentation=None,
        lineart=None,
        lineart_strength=0.7,
        invert_lineart=False,
        normals=None,
        normal_strength=0.4,
        segmentation_format="auto",
        use_slic=True,
        use_ciede2000=True,
        use_palette_merge=True,
        ciede2000_merge_thresh=10.0,
        use_thin_cleanup=True,
        min_region_width=5,
        use_shared_borders=True,
        label_mode="polylabel",
        use_bezier_smooth=False,
        use_content_protect=False,
        subject_priority=2.0,
        material_weight=0.5,
        edge_influence=0.3,
        use_auto_albedo=False,
        use_auto_mask=False,
        use_painterly_preprocess=False,
        painterly_sigma_s=60.0,
        painterly_sigma_r=0.45,
        slic_n_segments=500,
        slic_compactness=10.0,
        smoothing_kernel_size=9,
    ):
        # 1. Resolve Presets
        kwargs = {
            "image": image,
            "num_colors": num_colors,
            "simplification": simplification,
            "use_watershed": use_watershed,
            "output_mode": output_mode,
            "preset": preset,
            "segmentation": segmentation,
            "lineart": lineart,
            "lineart_strength": lineart_strength,
            "invert_lineart": invert_lineart,
            "normals": normals,
            "normal_strength": normal_strength,
            "segmentation_format": segmentation_format,
            "use_slic": use_slic,
            "use_ciede2000": use_ciede2000,
            "use_palette_merge": use_palette_merge,
            "ciede2000_merge_thresh": ciede2000_merge_thresh,
            "use_thin_cleanup": use_thin_cleanup,
            "min_region_width": min_region_width,
            "use_shared_borders": use_shared_borders,
            "label_mode": label_mode,
            "use_bezier_smooth": use_bezier_smooth,
            "use_content_protect": use_content_protect,
            "subject_priority": subject_priority,
            "material_weight": material_weight,
            "edge_influence": edge_influence,
            "use_auto_albedo": use_auto_albedo,
            "use_auto_mask": use_auto_mask,
            "use_painterly_preprocess": use_painterly_preprocess,
            "painterly_sigma_s": painterly_sigma_s,
            "painterly_sigma_r": painterly_sigma_r,
            "slic_n_segments": slic_n_segments,
            "slic_compactness": slic_compactness,
            "smoothing_kernel_size": smoothing_kernel_size,
        }
        params = cls._resolve_presets(kwargs)

        # 2. Extract perception inputs from tensors
        perception = cls._prepare_perception_inputs(kwargs, params)

        # 3. Setup batch processing
        batch_size = image.shape[0]
        result_images = []
        svg_contents = []
        color_counts = []

        api = ComfyAPISync()
        processor = ImageProcessor()
        renderer = PBNRenderer()

        n_colors_param = params.get("num_colors", 0)
        proc_params = ProcessingParameters(
            num_colors=n_colors_param if n_colors_param > 0 else None,
            simplification=params.get("simplification", 1.0),
            use_watershed=params.get("use_watershed", False),
            use_slic=params.get("use_slic", True),
            use_ciede2000=params.get("use_ciede2000", True),
            use_palette_merge=params.get("use_palette_merge", True),
            ciede2000_merge_thresh=params.get("ciede2000_merge_thresh", 10.0),
            use_thin_cleanup=params.get("use_thin_cleanup", True),
            min_region_width=params.get("min_region_width", 5),
            use_shared_borders=params.get("use_shared_borders", True),
            label_mode=params.get("label_mode", "polylabel"),
            use_bezier_smooth=params.get("use_bezier_smooth", False),
            use_content_protect=params.get("use_content_protect", False),
            perception=perception,
            preset=params.get("preset", "balanced"),
            output_mode=params.get("output_mode", "colored"),
            use_auto_albedo=params.get("use_auto_albedo", False),
            use_painterly_preprocess=params.get("use_painterly_preprocess", False),
            painterly_sigma_s=params.get("painterly_sigma_s", 60.0),
            painterly_sigma_r=params.get("painterly_sigma_r", 0.45),
            slic_n_segments=params.get("slic_n_segments", 500),
            slic_compactness=params.get("slic_compactness", 10.0),
            smoothing_kernel_size=params.get("smoothing_kernel_size", 9),
        )

        # 4. Batch loop
        for i in range(batch_size):
            if batch_size > 1:
                logger.info(f"Processing image {i + 1}/{batch_size}")

            img_tensor = image[i]
            h, w, _ = img_tensor.shape

            # Convert to OpenCV BGR
            img_bgr = cls._torch_to_bgr(img_tensor)

            # Process
            result = processor.process_array(img_bgr, proc_params, api=api)

            # Render
            output_mode = params.get("output_mode", "colored")
            if output_mode == "quantized":
                result_bgr = result.quantized
            else:
                result_bgr = renderer.render(
                    result.cleaned_regions,
                    result.label_data,
                    result.color_palette,
                    w,
                    h,
                    mode=output_mode,
                    region_colors=result.region_colors,
                    shared_borders=result.shared_borders,
                    use_shared_borders=params.get("use_shared_borders", True),
                )

            # Convert back to torch RGB
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0)

            result_images.append(result_tensor)
            svg_contents.append(result.svg_content)
            color_counts.append(result.color_palette.color_count)

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

    @staticmethod
    def _resolve_presets(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolves preset overrides into a final parameter dictionary."""
        params = kwargs.copy()
        preset = kwargs.get("preset", "balanced")

        if preset != "custom" and preset in PRESETS:
            overrides = PRESETS[preset]
            params.update(overrides)

            if preset == "portrait":
                params["use_auto_albedo"] = True
                if kwargs.get("segmentation") is None:
                    params["use_auto_mask"] = True

        return params

    @classmethod
    def _prepare_perception_inputs(cls, kwargs: dict[str, Any], params: dict[str, Any]) -> PerceptionInputs | None:
        """Decodes various input tensors into the PerceptionInputs structure."""
        normals_np = cls._decode_normals(kwargs.get("normals"))
        lineart_np = cls._decode_lineart(kwargs.get("lineart"), kwargs.get("invert_lineart", False))
        segmentation_np = cls._decode_segmentation(
            kwargs.get("segmentation"), kwargs.get("segmentation_format", "auto")
        )

        use_auto_mask = params.get("use_auto_mask", False)

        has_perception = any(x is not None for x in [segmentation_np, normals_np, lineart_np]) or use_auto_mask

        if not has_perception:
            return None

        return PerceptionInputs(
            albedo=None,  # Handled internally in pipeline now
            segmentation_mask=segmentation_np,
            normal_map=normals_np,
            normal_strength=kwargs.get("normal_strength", 0.4),
            lineart=lineart_np,
            lineart_strength=kwargs.get("lineart_strength", 0.7),
            invert_lineart=kwargs.get("invert_lineart", False),
            subject_priority=kwargs.get("subject_priority", 2.0),
            material_weight=kwargs.get("material_weight", 0.5),
            edge_influence=kwargs.get("edge_influence", 0.3),
            use_auto_mask=use_auto_mask,
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
    def _decode_normals(t: torch.Tensor | None) -> np.ndarray | None:
        """Decodes Normal map tensor to [-1, 1] unit vectors."""
        if t is None:
            return None
        arr = t[0].cpu().numpy()
        normals = arr * 2.0 - 1.0
        norms = np.linalg.norm(normals, axis=2, keepdims=True).clip(min=1e-6)
        return normals / norms

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
    def _save_svg_batch(svg_contents: list[str]) -> list[dict[str, str]]:
        """Saves SVGs to temp directory using content-addressed hashing."""
        svg_results = []
        temp_dir = folder_paths.get_temp_directory()

        for content in svg_contents:
            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
            filename = f"pbn_{content_hash}.svg"
            filepath = os.path.join(temp_dir, filename)

            # Only write if file doesn't exist to avoid redundant I/O
            if not os.path.exists(filepath):
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            svg_results.append({"filename": filename, "subfolder": "", "type": "temp"})

        return svg_results
