import dataclasses
import logging
import time

import cv2
import numpy as np
import skimage.color
import skimage.segmentation

from .backend.labeling.label_placer import LabelPlacer
from .backend.models import PerceptionInputs, ProcessingParameters, SVGResult
from .backend.preprocessing.normal_features import augment_image_with_normals
from .backend.preprocessing.preprocessor import Preprocessor
from .backend.preprocessing.protector import Protector
from .backend.preprocessing.retinex import multiscale_retinex
from .backend.preprocessing.sapiens_priority import build_priority_map
from .backend.quantization.quantizer import ColorQuantizer
from .backend.segmentation.segmenter import RegionSegmenter
from .backend.svg_generation.svg_generator import SVGGenerator
from .backend.vectorization.vectorizer import Vectorizer

# Configure logging
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Main orchestrator for the complete image-to-SVG processing pipeline.
    """

    def __init__(self):
        """Initialize the image processor with all module instances."""
        self.preprocessor = Preprocessor()
        self.svg_generator = SVGGenerator()

    def process_array(
        self, image_bgr: cv2.Mat, params: ProcessingParameters, api=None
    ) -> SVGResult:
        """
        Process image array through complete pipeline.

        Args:
            image_bgr: Input image in BGR format (numpy array)
            params: Processing parameters
            api: Optional ComfyAPISync instance for progress reporting

        Returns:
            SVGResult with generated SVG and metadata
        """
        start_time = time.time()

        try:
            # Stage 2: Preprocessing
            logger.info("Stage 1/6: Preprocessing image")
            if api:
                api.execution.set_progress(1, 6)
            preprocessed, metadata = self.preprocessor.preprocess(
                image_bgr,
                use_painterly=params.use_painterly_preprocess,
                painterly_sigma_s=params.painterly_sigma_s,
                painterly_sigma_r=params.painterly_sigma_r,
            )

            # Generate protection map if enabled
            use_content_protect = params.use_content_protect
            protection_map = None
            if use_content_protect:
                protector = Protector()
                logger.info("Generating content protection map")
                protection_map = protector.generate_protection_map(preprocessed)

            # Apply SLIC superpixels if enabled
            use_slic = params.use_slic
            if use_slic:
                perception = params.perception
                normal_map = perception.normal_map if perception else None
                normal_strength = perception.normal_strength if perception else 0.0

                if normal_map is not None and normal_strength > 0:
                    # Build 5-channel LAB + normal-feature image for SLIC
                    lab_image = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2LAB).astype(np.float32)
                    slic_input = augment_image_with_normals(lab_image, normal_map, normal_strength)
                    # skimage.slic with channel_axis=-1, multichannel feature space
                    segments = skimage.segmentation.slic(
                        slic_input,
                        n_segments=params.slic_n_segments,
                        compactness=params.slic_compactness,
                        start_label=1,
                        channel_axis=-1,  # treat last dim as features, not spatial
                    )
                else:
                    # Standard RGB SLIC (existing behavior)
                    rgb_image = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
                    segments = skimage.segmentation.slic(
                        rgb_image,
                        n_segments=params.slic_n_segments,
                        compactness=params.slic_compactness,
                        start_label=1,
                    )

                superpixel_img = skimage.color.label2rgb(
                    segments, preprocessed, kind="avg", bg_label=-1
                ).astype(np.uint8)
                input_for_quantization = superpixel_img
            else:
                input_for_quantization = preprocessed

            # Make quantizer use protection map
            quantizer = ColorQuantizer(
                use_palette_merge=params.use_palette_merge,
                ciede2000_merge_thresh=params.ciede2000_merge_thresh,
                use_ciede2000=params.use_ciede2000,
            )
            quantizer.protection_map = protection_map

            if (
                params.perception is not None
                and params.perception.segmentation_mask is not None
                and params.perception.segmentation_mask.ndim == 2  # grayscale class map
            ):
                priority_map = build_priority_map(params.perception.segmentation_mask)
                # Merge with any existing protection map
                if protection_map is not None:
                    protection_map = protection_map * priority_map
                else:
                    protection_map = priority_map
                quantizer.protection_map = protection_map

            # stage metadata
            perception = params.perception
            lineart_map = perception.lineart if perception else None
            lineart_strength = perception.lineart_strength if perception else 0.0

            # Auto-albedo: estimate albedo via MSR if no external albedo provided
            # This must run BEFORE quantization to influence the palette
            if params.use_auto_albedo and (perception is None or perception.albedo is None):
                logger.info("Estimating auto-albedo via MSR Retinex")
                auto_albedo = multiscale_retinex(input_for_quantization)
                perception = (
                    dataclasses.replace(perception, albedo=auto_albedo)
                    if perception
                    else PerceptionInputs(albedo=auto_albedo)
                )
                # Ensure updated perception is used for quantization
                params = dataclasses.replace(params, perception=perception)

            # Stage 3: Color Quantization
            logger.info("Stage 2/6: Quantizing image colors")
            if api:
                api.execution.set_progress(2, 6)

            quantized, palette = quantizer.quantize(
                input_for_quantization, params.num_colors, perception=params.perception
            )

            # Stage 4: Region Segmentation
            logger.info("Stage 3/6: Segmenting regions")
            if api:
                api.execution.set_progress(3, 6)

            segmenter = RegionSegmenter(
                use_watershed=params.use_watershed,
                use_ciede2000=params.use_ciede2000,
                use_thin_cleanup=params.use_thin_cleanup,
                min_region_width=params.min_region_width,
                edge_weight_map=lineart_map,
                lineart_strength=lineart_strength,
            )
            region_data = segmenter.segment(quantized, palette.colors)

            # Stage 5: Vectorization
            logger.info("Stage 4/6: Vectorizing regions")
            if api:
                api.execution.set_progress(4, 6)

            vectorizer = Vectorizer(use_bezier_smooth=params.use_bezier_smooth)
            vectorized_regions = vectorizer.vectorize(region_data, params.simplification)

            # Optional: Remove speckles
            logger.info("Removing speckles")
            cleaned_regions, updated_region_colors = vectorizer.remove_speckles(
                vectorized_regions,
                dict(region_data.region_colors),
                palette.colors,
                threshold=vectorizer.speckle_threshold,
            )

            # Renumber regions to have consecutive IDs (1, 2, 3, ...)
            cleaned_regions, renumbered_colors = self._renumber_regions(
                cleaned_regions, updated_region_colors
            )

            # Stage 6: Label Placement
            logger.info("Stage 5/6: Placing labels")
            if api:
                api.execution.set_progress(5, 6)

            label_placer = LabelPlacer(label_mode=params.label_mode, lineart=lineart_map)
            label_data = label_placer.place_labels(cleaned_regions)

            # Stage 7: SVG Generation
            logger.info("Stage 6/6: Generating SVG")
            if api:
                api.execution.set_progress(6, 6)

            # Pass shared borders to SVG generator
            shared_borders = region_data.shared_borders

            svg_content = self.svg_generator.generate_svg(
                cleaned_regions,
                label_data,
                palette,
                region_colors=renumbered_colors,
                shared_borders=shared_borders,
                use_shared_borders=params.use_shared_borders,
                print_mode=(params.output_mode == "print_svg"),
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create result
            return SVGResult(
                svg_content=svg_content,
                color_palette=palette,
                processing_time=processing_time,
                region_count=len(cleaned_regions),
                label_count=len(label_data.positions),
                cleaned_regions=cleaned_regions,
                label_data=label_data,
                quantized=quantized,
                region_colors=renumbered_colors,
            )

        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise ValueError(f"Image processing failed: {str(e)}") from e

    def _renumber_regions(self, regions: dict, region_colors: dict) -> tuple[dict, dict]:
        """
        Renumber regions to have consecutive IDs starting from 1,
        preserving their color identity.
        """
        renumbered_regions = {}
        renumbered_colors = {}
        new_id = 1
        for old_id in sorted(regions.keys()):
            renumbered_regions[new_id] = regions[old_id]
            if old_id in region_colors:
                renumbered_colors[new_id] = region_colors[old_id]
            new_id += 1
        return renumbered_regions, renumbered_colors
