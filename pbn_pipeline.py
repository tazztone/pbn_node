"""
Main image processing pipeline orchestrator adapted for numpy arrays.
"""

import logging
import time

import cv2

from .backend.models import ProcessingParameters, SVGResult
from .backend.preprocessing.preprocessor import Preprocessor
from .backend.segmentation.segmenter import RegionSegmenter
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
            preprocessed, metadata = self.preprocessor.preprocess(image_bgr)

            # Generate protection map if enabled
            use_content_protect = params.use_content_protect
            protection_map = None
            if use_content_protect:
                from .backend.preprocessing.protector import Protector

                protector = Protector()
                logger.info("Generating content protection map")
                protection_map = protector.generate_protection_map(preprocessed)

            # Apply SLIC superpixels if enabled
            use_slic = params.use_slic
            if use_slic:
                logger.info("Applying SLIC superpixels")
                import numpy as np
                import skimage.segmentation

                # convert to rgb for slic
                rgb_image = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
                segments = skimage.segmentation.slic(
                    rgb_image, n_segments=500, compactness=10, start_label=1
                )

                # replace each pixel with its superpixel mean color using vectorized approach
                superpixel_img = skimage.color.label2rgb(
                    segments, preprocessed, kind="avg", bg_label=-1
                )
                # label2rgb returns float64 [0,1] or [0,255] depending on input,
                # but with kind='avg' and uint8 input it should be consistent.
                # To be safe, ensure uint8.
                superpixel_img = (superpixel_img).astype(np.uint8)

                input_for_quantization = superpixel_img
            else:
                input_for_quantization = preprocessed

            # Make quantizer use protection map
            from .backend.quantization.quantizer import ColorQuantizer

            quantizer = ColorQuantizer(
                use_palette_merge=params.use_palette_merge,
                ciede2000_merge_thresh=params.ciede2000_merge_thresh,
                use_ciede2000=params.use_ciede2000,
            )
            quantizer.protection_map = protection_map

            # Stage 3: Color Quantization
            logger.info("Stage 2/6: Quantizing colors")
            if api:
                api.execution.set_progress(2, 6)

            use_budget_split = params.use_budget_split
            if use_budget_split and params.num_colors and params.num_colors >= 4:
                import numpy as np

                hsv = cv2.cvtColor(input_for_quantization, cv2.COLOR_BGR2HSV)
                v_channel = hsv[:, :, 2]

                # Otsu's thresholding to separate foreground and background
                _, mask = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Assuming larger area might be background or foreground,
                # simple logic: 75% to fg, 25% to bg. But which is which?
                # Typically, Otsu gives one class 0 and another 255.
                # Let's say foreground has more details, so we oversample it.
                # But actually, the goal is: split color budget.
                # Since we use KMeans, we can just assign weights. If we want 75% budget
                # to foreground, we can duplicate foreground pixels in the array we pass,
                # or we can partition the image, quantize separately, and combine.
                # The easiest robust way to "split budget" across a single KMeans fit is to
                # strongly weight the foreground pixels so they attract more centers.
                # However, the prompt says "allocate 25% budget to background, 75% to foreground".
                # To do this exactly, we can split the image into fg/bg, quantize fg with
                # 0.75*k colors, quantize bg with 0.25*k colors, and merge the palettes.
                # Then requantize the whole image.
                # Let's do that.

                k = params.num_colors
                k_bg = max(1, int(k * 0.25))
                k_fg = max(1, k - k_bg)

                # fg is mask > 0, bg is mask == 0
                fg_pixels = input_for_quantization[mask > 0]
                bg_pixels = input_for_quantization[mask == 0]

                # Fallback if one region is empty
                if len(fg_pixels) == 0 or len(bg_pixels) == 0:
                    quantized, palette = quantizer.quantize(input_for_quantization, k)
                else:
                    # Create dummy images for the quantizer
                    # Quantizer expects a 2D image (H, W, 3).
                    # We can just reshape pixels to (N, 1, 3)
                    fg_img = fg_pixels.reshape(-1, 1, 3)
                    bg_img = bg_pixels.reshape(-1, 1, 3)

                    _, fg_palette = quantizer.quantize(fg_img, k_fg)
                    _, bg_palette = quantizer.quantize(bg_img, k_bg)

                    # Merge palettes
                    merged_colors_lab = np.vstack([fg_palette.colors, bg_palette.colors])

                    # Quantizer expects to return (quantized_image, palette).
                    # We can assign all pixels to the closest color in merged_colors_lab.
                    from .backend.models import ColorPalette

                    lab_image = cv2.cvtColor(input_for_quantization, cv2.COLOR_BGR2LAB).astype(
                        np.float32
                    )
                    h, w = lab_image.shape[:2]
                    pixels = lab_image.reshape(-1, 3)

                    use_ciede2000 = params.use_ciede2000
                    if use_ciede2000:
                        import skimage.color

                        from .backend.utils.color import cv_to_std_lab

                        std_pixels = cv_to_std_lab(pixels)
                        std_colors = cv_to_std_lab(merged_colors_lab)

                        k_total = merged_colors_lab.shape[0]
                        dists = np.zeros((pixels.shape[0], k_total), dtype=np.float32)

                        for ki in range(k_total):
                            dists[:, ki] = skimage.color.deltaE_ciede2000(
                                std_pixels, std_colors[[ki]]
                            )
                    else:
                        diffs = pixels[:, np.newaxis, :] - merged_colors_lab[np.newaxis, :, :]
                        dists = np.sum(diffs**2, axis=2)

                    labels = np.argmin(dists, axis=1)

                    quantized_pixels = merged_colors_lab[labels]
                    quantized_lab = quantized_pixels.reshape(h, w, 3).astype(np.uint8)
                    quantized = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)

                    # Convert LAB centers to RGB for hex
                    centers_lab_uint8 = merged_colors_lab.astype(np.uint8).reshape(1, -1, 3)
                    centers_bgr = cv2.cvtColor(centers_lab_uint8, cv2.COLOR_LAB2BGR)
                    centers_rgb = cv2.cvtColor(centers_bgr, cv2.COLOR_BGR2RGB)
                    hex_colors = ["#{:02x}{:02x}{:02x}".format(*rgb) for rgb in centers_rgb[0]]

                    palette = ColorPalette(
                        colors=merged_colors_lab,
                        hex_colors=hex_colors,
                        color_count=len(merged_colors_lab),
                    )
            else:
                quantized, palette = quantizer.quantize(input_for_quantization, params.num_colors)

            # Stage 4: Region Segmentation
            logger.info("Stage 3/6: Segmenting regions")
            if api:
                api.execution.set_progress(3, 6)
            segmenter = RegionSegmenter(
                use_watershed=params.use_watershed,
                use_ciede2000=params.use_ciede2000,
                use_thin_cleanup=params.use_thin_cleanup,
                min_region_width=params.min_region_width,
            )
            region_data = segmenter.segment(quantized, palette.colors)

            # Stage 5: Vectorization
            logger.info("Stage 4/6: Vectorizing regions")
            if api:
                api.execution.set_progress(4, 6)
            from .backend.vectorization.vectorizer import Vectorizer

            vectorizer = Vectorizer(use_bezier_smooth=params.use_bezier_smooth)
            vectorized_regions = vectorizer.vectorize(region_data, params.simplification)

            # Optional: Remove speckles
            logger.info("Removing speckles")
            cleaned_regions = vectorizer.remove_speckles(
                vectorized_regions, palette.colors, threshold=vectorizer.speckle_threshold
            )

            # Renumber regions to have consecutive IDs (1, 2, 3, ...)
            cleaned_regions = self._renumber_regions(cleaned_regions)

            # Stage 6: Label Placement
            logger.info("Stage 5/6: Placing labels")
            if api:
                api.execution.set_progress(5, 6)
            from .backend.labeling.label_placer import LabelPlacer

            label_placer = LabelPlacer(label_mode=params.label_mode)
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
                shared_borders=shared_borders,
                use_shared_borders=params.use_shared_borders,
                print_mode=(params.output_mode == "print_svg"),
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create result
            result = SVGResult(
                svg_content=svg_content,
                color_palette=palette,
                processing_time=processing_time,
                region_count=len(cleaned_regions),
                label_count=len(label_data.positions),
            )

            # Store internal data for renderer
            self.last_cleaned_regions = cleaned_regions
            self.last_label_data = label_data
            self.last_palette = palette
            self.last_quantized = quantized

            return result

        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise ValueError(f"Image processing failed: {str(e)}") from e

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
