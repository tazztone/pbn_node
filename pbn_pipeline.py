import dataclasses
import logging
import time

import cv2

from .backend.labeling.label_placer import LabelPlacer
from .backend.models import PerceptionInputs, ProcessingParameters, SVGResult
from .backend.preprocessing.preprocessor import Preprocessor
from .backend.preprocessing.retinex import multiscale_retinex
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
        self.segmenter = RegionSegmenter()
        self.vectorizer = Vectorizer()
        self.svg_generator = SVGGenerator()

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
            preprocessed = self._stage_1_preprocessing(image_bgr, p, api)
            perception, lineart_map, edge_map, edge_strength = self._stage_2_perception(p, api)

            input_for_quantization = preprocessed

            quantized, palette, p = self._stage_3_quantization(input_for_quantization, p, perception, api)

            region_data = self._stage_4_segmentation(quantized, palette, edge_map, edge_strength, p, api)

            cleaned_regions, renumbered_colors = self._stage_5_vectorization(region_data, p, api)

            return self._stage_6_finalization(
                cleaned_regions, renumbered_colors, quantized, lineart_map, region_data, palette, p, start_time, api
            )

        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise ValueError(f"Image processing failed: {str(e)}") from e

    def _stage_1_preprocessing(self, image_bgr: cv2.Mat, p: ProcessingParameters, api=None):
        logger.info("Stage 1/6: Preprocessing image")
        if api:
            api.execution.set_progress(1, 6)
        return self.preprocessor.preprocess(image_bgr, use_clahe=p.use_clahe)

    def _stage_2_perception(self, p: ProcessingParameters, api=None):
        logger.info("Stage 2/6: Analyzing image perception")
        if api:
            api.execution.set_progress(2, 6)
        perception = p.perception
        lineart_map = perception.lineart if perception else None
        lineart_strength = perception.lineart_strength if perception else 0.0

        # Use lineart map as edge map
        edge_map = lineart_map
        edge_strength = lineart_strength
        return perception, lineart_map, edge_map, edge_strength

    def _stage_3_quantization(self, input_for_quantization, p: ProcessingParameters, perception, api=None):
        logger.info("Stage 3/6: Quantizing image colors")
        if api:
            api.execution.set_progress(3, 6)

        # Update quantizer state
        self.quantizer.use_palette_merge = p.use_palette_merge
        self.quantizer.ciede2000_merge_thresh = p.ciede2000_merge_thresh
        self.quantizer.use_ciede2000 = p.use_ciede2000

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
        return quantized, palette, p

    def _stage_4_segmentation(self, quantized, palette, edge_map, edge_strength, p: ProcessingParameters, api=None):
        logger.info("Stage 4/6: Segmenting regions")
        if api:
            api.execution.set_progress(4, 6)

        self.segmenter.use_ciede2000 = p.use_ciede2000
        self.segmenter.use_thin_cleanup = p.use_thin_cleanup
        self.segmenter.min_region_width = p.min_region_width
        self.segmenter.edge_weight_map = edge_map
        self.segmenter.lineart_strength = edge_strength
        self.segmenter.smoothing_kernel_size = p.smoothing_kernel_size
        self.segmenter.min_region_size = p.min_region_size

        return self.segmenter.segment(quantized, palette.colors)

    def _stage_5_vectorization(self, region_data, p: ProcessingParameters, api=None):
        logger.info("Stage 5/6: Vectorizing regions")
        if api:
            api.execution.set_progress(5, 6)
        self.vectorizer.use_bezier_smooth = p.use_bezier_smooth
        vectorized_regions = self.vectorizer.vectorize(region_data, p.simplification)

        # Stage 5: Vectorization - Skip speckle removal as it causes coverage gaps
        # The new majority smoothing logic already handles noise effectively.
        cleaned_regions = vectorized_regions
        updated_region_colors = dict(region_data.region_colors)

        # Renumber regions to have consecutive IDs (1, 2, 3, ...)
        return self._renumber_regions(cleaned_regions, updated_region_colors)

    def _stage_6_finalization(
        self,
        cleaned_regions,
        renumbered_colors,
        quantized,
        lineart_map,
        region_data,
        palette,
        p: ProcessingParameters,
        start_time: float,
        api=None,
    ) -> SVGResult:
        logger.info("Stage 6/6: Finalizing template")
        if api:
            api.execution.set_progress(6, 6)

        h, w = quantized.shape[:2]
        label_placer = LabelPlacer(label_mode=p.label_mode, lineart=lineart_map)
        label_data = label_placer.place_labels(cleaned_regions, width=w, height=h)

        svg_content = self.svg_generator.generate_svg(
            cleaned_regions,
            label_data,
            palette,
            region_colors=renumbered_colors,
            shared_borders=region_data.shared_borders,
            use_shared_borders=p.use_shared_borders,
            print_mode=(p.output_mode in ("outline", "print_svg")),
        )

        if api:
            api.execution.set_progress(6, 6)

        processing_time = time.time() - start_time

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
            shared_borders=region_data.shared_borders,
        )

    def _renumber_regions(self, regions: dict, region_colors: dict) -> tuple[dict, dict]:
        """
        Renumber regions to have consecutive IDs starting from 1,
        preserving their color identity.
        """
        sorted_ids = sorted(regions)
        renumbered_regions = {new_id: regions[old_id] for new_id, old_id in enumerate(sorted_ids, 1)}
        renumbered_colors = {
            new_id: region_colors[old_id] for new_id, old_id in enumerate(sorted_ids, 1) if old_id in region_colors
        }
        return renumbered_regions, renumbered_colors
