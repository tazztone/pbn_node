"""
Main image processing pipeline orchestrator adapted for numpy arrays.
"""
import cv2
import time
import logging
from typing import Optional, Dict

from .backend.models import ProcessingParameters, SVGResult, ColorPalette
from .backend.preprocessing.preprocessor import Preprocessor
from .backend.quantization.quantizer import ColorQuantizer
from .backend.segmentation.segmenter import RegionSegmenter
from .backend.vectorization.vectorizer import Vectorizer
from .backend.labeling.label_placer import LabelPlacer
from .backend.svg_generation.svg_generator import SVGGenerator

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
        self.vectorizer = Vectorizer()
        self.label_placer = LabelPlacer()
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
        
        try:
            # Stage 2: Preprocessing
            logger.info("Stage 1/6: Preprocessing image")
            if api: api.execution.set_progress(1, 6)
            preprocessed, metadata = self.preprocessor.preprocess(image_bgr)
            
            # Stage 3: Color Quantization
            logger.info("Stage 2/6: Quantizing colors")
            if api: api.execution.set_progress(2, 6)
            quantized, palette = self.quantizer.quantize(preprocessed, params.num_colors)
            
            # Stage 4: Region Segmentation
            logger.info("Stage 3/6: Segmenting regions")
            if api: api.execution.set_progress(3, 6)
            segmenter = RegionSegmenter(use_watershed=params.use_watershed)
            region_data = segmenter.segment(quantized, palette.colors)
            
            # Stage 5: Vectorization
            logger.info("Stage 4/6: Vectorizing regions")
            if api: api.execution.set_progress(4, 6)
            vectorized_regions = self.vectorizer.vectorize(region_data, params.simplification)
            
            # Optional: Remove speckles
            logger.info("Removing speckles")
            cleaned_regions = self.vectorizer.remove_speckles(
                vectorized_regions,
                palette.colors,
                threshold=self.vectorizer.speckle_threshold
            )
            
            # Renumber regions to have consecutive IDs (1, 2, 3, ...)
            cleaned_regions = self._renumber_regions(cleaned_regions)
            
            # Stage 6: Label Placement
            logger.info("Stage 5/6: Placing labels")
            if api: api.execution.set_progress(5, 6)
            label_data = self.label_placer.place_labels(cleaned_regions)
            
            # Stage 7: SVG Generation
            logger.info("Stage 6/6: Generating SVG")
            if api: api.execution.set_progress(6, 6)
            svg_content = self.svg_generator.generate_svg(
                cleaned_regions,
                label_data,
                palette
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = SVGResult(
                svg_content=svg_content,
                color_palette=palette,
                processing_time=processing_time,
                region_count=len(cleaned_regions),
                label_count=len(label_data.positions)
            )
            
            # Store internal data for renderer
            self.last_cleaned_regions = cleaned_regions
            self.last_label_data = label_data
            self.last_palette = palette
            self.last_quantized = quantized
            
            return result
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise ValueError(f"Image processing failed: {str(e)}")

    def _renumber_regions(self, regions: dict) -> dict:
        """
        Renumber regions to have consecutive IDs starting from 1.
        """
        renumbered = {}
        new_id = 1
        for old_id in sorted(regions.keys()):
            renumbered[new_id] = regions[old_id]
            new_id += 1
        return renumbered
