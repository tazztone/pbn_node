import dataclasses
import logging
import time

import cv2
import numpy as np
import skimage.color
import skimage.segmentation

from .backend.labeling.label_placer import LabelPlacer
from .backend.models import PerceptionInputs, ProcessingParameters, SVGResult
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
        self.quantizer = ColorQuantizer()
        self.svg_generator = SVGGenerator()
        self._protector = None

    @property
    def protector(self):
        """Lazy-load the Protector to avoid heavy imports (mediapipe) if not used."""
        if self._protector is None:
            self._protector = Protector()
        return self._protector

    def process_array(self, image_bgr: cv2.Mat, params: ProcessingParameters, api=None) -> SVGResult:
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
        # Local copy of params to avoid mutating original
        p = dataclasses.replace(params)

        try:
            # Stage 1: Preprocessing
            logger.info("Stage 1/6: Preprocessing image")
            if api:
                api.execution.set_progress(1, 6)
            preprocessed = self.preprocessor.preprocess(
                image_bgr,
                use_painterly=p.use_painterly_preprocess,
                painterly_sigma_s=p.painterly_sigma_s,
                painterly_sigma_r=p.painterly_sigma_r,
            )

            # Stage 2: Content Protection & Perception
            logger.info("Stage 2/6: Analyzing image perception")
            if api:
                api.execution.set_progress(2, 6)
            perception = p.perception
            lineart_map = perception.lineart if perception else None
            lineart_strength = perception.lineart_strength if perception else 0.0

            # Use lineart map as edge map
            edge_map = lineart_map
            edge_strength = lineart_strength

            # Generate protection map if enabled
            protection_map = None
            if p.use_content_protect:
                logger.info("Generating content protection map")
                protection_map = self.protector.generate_protection_map(preprocessed)

            if perception and perception.segmentation_mask is not None:
                if perception.segmentation_mask.ndim == 2:  # grayscale class map
                    priority_map = build_priority_map(perception.segmentation_mask)
                    # Merge with any existing protection map
                    if protection_map is not None:
                        protection_map = protection_map * priority_map
                    else:
                        protection_map = priority_map

            # Apply SLIC superpixels if enabled
            if p.use_slic:
                # Standard RGB SLIC
                rgb_image = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
                segments = skimage.segmentation.slic(
                    rgb_image,
                    n_segments=p.slic_n_segments,
                    compactness=p.slic_compactness,
                    start_label=1,
                )

                input_for_quantization = skimage.color.label2rgb(
                    segments, preprocessed, kind="avg", bg_label=-1
                ).astype(np.uint8)
            else:
                input_for_quantization = preprocessed

            # Stage 3: Color Quantization
            logger.info("Stage 3/6: Quantizing image colors")
            if api:
                api.execution.set_progress(3, 6)

            # Update quantizer state
            self.quantizer.use_palette_merge = p.use_palette_merge
            self.quantizer.ciede2000_merge_thresh = p.ciede2000_merge_thresh
            self.quantizer.use_ciede2000 = p.use_ciede2000
            self.quantizer.protection_map = protection_map

            # Auto-albedo: estimate albedo via MSR if enabled
            if p.use_auto_albedo:
                logger.info("Estimating auto-albedo via MSR Retinex")
                auto_albedo = multiscale_retinex(input_for_quantization)
                if perception is None:
                    perception = PerceptionInputs(albedo=auto_albedo)
                else:
                    perception = dataclasses.replace(perception, albedo=auto_albedo)
                p = dataclasses.replace(p, perception=perception)

            quantized, palette = self.quantizer.quantize(input_for_quantization, p.num_colors, perception=p.perception)

            # Stage 4: Region Segmentation
            logger.info("Stage 4/6: Segmenting regions")
            if api:
                api.execution.set_progress(4, 6)

            segmenter = RegionSegmenter(
                use_watershed=p.use_watershed,
                use_ciede2000=p.use_ciede2000,
                use_thin_cleanup=p.use_thin_cleanup,
                min_region_width=p.min_region_width,
                edge_weight_map=edge_map,
                lineart_strength=edge_strength,
                smoothing_kernel_size=p.smoothing_kernel_size,
            )
            region_data = segmenter.segment(quantized, palette.colors)

            # Stage 5: Vectorization
            logger.info("Stage 5/6: Vectorizing regions")
            if api:
                api.execution.set_progress(5, 6)

            vectorizer = Vectorizer(use_bezier_smooth=p.use_bezier_smooth)
            vectorized_regions = vectorizer.vectorize(region_data, p.simplification)

            logger.info("Removing speckles")
            cleaned_regions, updated_region_colors = vectorizer.remove_speckles(
                vectorized_regions,
                dict(region_data.region_colors),
                palette.colors,
                threshold=vectorizer.speckle_threshold,
            )

            # Renumber regions to have consecutive IDs (1, 2, 3, ...)
            cleaned_regions, renumbered_colors = self._renumber_regions(cleaned_regions, updated_region_colors)

            # Stage 6: Label Placement & SVG Generation
            logger.info("Stage 6/6: Finalizing template")
            if api:
                api.execution.set_progress(6, 6)

            label_placer = LabelPlacer(label_mode=p.label_mode, lineart=lineart_map)
            label_data = label_placer.place_labels(cleaned_regions)

            svg_content = self.svg_generator.generate_svg(
                cleaned_regions,
                label_data,
                palette,
                region_colors=renumbered_colors,
                shared_borders=region_data.shared_borders,
                use_shared_borders=p.use_shared_borders,
                print_mode=(p.output_mode == "print_svg"),
            )

            if api:
                api.execution.set_progress(6, 6)

            # Create final high-fidelity preview from simplified polygons
            # This reflects the ACTUAL smooth boundaries and simplification in the final SVG
            h, w = quantized.shape[:2]
            final_preview = np.zeros((h, w, 3), dtype=np.uint8)

            # Convert palette to BGR for rasterization using OpenCV's internal mapping
            palette_lab_uint8 = palette.colors.astype(np.uint8).reshape(1, -1, 3)
            palette_bgr = cv2.cvtColor(palette_lab_uint8, cv2.COLOR_LAB2BGR).reshape(-1, 3)

            # Draw each polygon
            for rid, polygon in cleaned_regions.items():
                if rid in renumbered_colors:
                    color_idx = renumbered_colors[rid]
                    color = palette_bgr[color_idx].tolist()

                    # Convert polygon coordinates to numpy for cv2.fillPoly
                    if polygon.exterior:
                        pts = np.array(polygon.exterior.coords, dtype=np.int32)
                        cv2.fillPoly(final_preview, [pts], color)

                        # Handle holes (interiors)
                        for _ in polygon.interiors:
                            # Note: We don't really have 'holes' in a typical PBN,
                            # but for completeness we fill them with background or handle as needed.
                            # In PBN, polygons are usually touching, not nested.
                            pass

            # If there are any unfilled pixels (black), fill them with the original quantized image
            # as a fallback, though vectorized regions should cover the whole image.
            mask = np.all(final_preview == 0, axis=2)
            final_preview[mask] = quantized[mask]

            processing_time = time.time() - start_time

            return SVGResult(
                svg_content=svg_content,
                color_palette=palette,
                processing_time=processing_time,
                region_count=len(cleaned_regions),
                label_count=len(label_data.positions),
                cleaned_regions=cleaned_regions,
                label_data=label_data,
                quantized=final_preview,
                region_colors=renumbered_colors,
                shared_borders=region_data.shared_borders,
            )

        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise ValueError(f"Image processing failed: {str(e)}") from e

    def _renumber_regions(self, regions: dict, region_colors: dict) -> tuple[dict, dict]:
        """
        Renumber regions to have consecutive IDs starting from 1,
        preserving their color identity.
        """
        sorted_ids = sorted(regions.keys())
        renumbered_regions = {new_id: regions[old_id] for new_id, old_id in enumerate(sorted_ids, 1)}
        renumbered_colors = {
            new_id: region_colors[old_id] for new_id, old_id in enumerate(sorted_ids, 1) if old_id in region_colors
        }
        return renumbered_regions, renumbered_colors
