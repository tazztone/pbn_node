"""
Quality metrics module for Paint By Number generation.
Computes measurable quality dimensions from SVGResult data.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pbn_node.backend.models import SVGResult


@dataclass
class QualityReport:
    # === Region health ===
    total_regions: int
    speck_count: int  # regions < area_threshold
    speck_ratio: float  # speck_count / total_regions
    largest_region_ratio: float  # largest region / image area

    # === Color palette ===
    actual_color_count: int
    requested_color_count: int
    color_efficiency: float  # actual / requested

    # === Label coverage ===
    label_coverage: float  # labeled_regions / total_regions
    unlabeled_count: int

    # === Edge fidelity ===
    edge_violation_ratio: float | None  # % of lineart edges missing in output

    # === Coverage ===
    fill_coverage: float  # % of pixels covered by regions


def analyze(
    result: SVGResult, image_shape: tuple, requested_colors: int, lineart: np.ndarray | None = None
) -> QualityReport:
    """
    Analyze PBN results and compute quality metrics.
    """
    h, w = image_shape[:2]
    total_pixels = h * w

    regions = result.cleaned_regions
    total_regions = len(regions)

    # Speck detection and Area sum for geometric coverage
    speck_count = 0
    area_threshold = total_pixels * 0.001
    total_geometric_area = 0.0
    areas = []
    for _rid, poly in regions.items():
        area = poly.area
        areas.append(area)
        total_geometric_area += area
        if area < area_threshold:
            speck_count += 1

    largest_region_ratio = max(areas) / total_pixels if areas else 0.0
    speck_ratio = speck_count / total_regions if total_regions > 0 else 0.0

    # Label coverage
    labeled = len(result.label_data.positions)
    unlabeled = len(result.label_data.skipped_regions)
    label_coverage = labeled / total_regions if total_regions > 0 else 0.0

    # Fill coverage: use rasterization to get exact pixel count
    # This avoids the underestimation inherent in geometric area sums
    mask = np.zeros((h, w), dtype=np.uint8)
    for _rid, poly in regions.items():
        # Convert shapely polygon to cv2-compatible contour
        ext_coords = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [ext_coords], 255)

    fill_pixels = np.count_nonzero(mask)
    fill_coverage = fill_pixels / total_pixels

    # Color efficiency
    actual_colors = result.color_palette.color_count
    color_efficiency = actual_colors / requested_colors if requested_colors > 0 else 1.0

    # Edge fidelity
    edge_violation_ratio = None
    if lineart is not None:
        # Resize lineart to match output preview
        resized_lineart = cv2.resize(lineart, (w, h))
        if len(resized_lineart.shape) == 3:
            resized_lineart = cv2.cvtColor(resized_lineart, cv2.COLOR_BGR2GRAY)

        # Binary mask of strong lineart edges (tightened to 0.5)
        strong_edges = (resized_lineart > 0.5).astype(np.uint8)

        # Detect edges in output quantized preview
        preview = result.quantized
        gray_preview = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        # Canny to find boundaries in the PBN result
        output_edges = cv2.Canny(gray_preview, 50, 150) > 0

        # Violations: where strong lineart exists but no PBN boundary was created
        # We dilate the output edges slightly to allow for small misalignments
        kernel = np.ones((3, 3), np.uint8)
        dilated_output_edges = cv2.dilate(output_edges.astype(np.uint8), kernel, iterations=1)

        violations = strong_edges & ~dilated_output_edges
        edge_violation_ratio = np.sum(violations) / max(np.sum(strong_edges), 1)

    return QualityReport(
        total_regions=total_regions,
        speck_count=speck_count,
        speck_ratio=speck_ratio,
        largest_region_ratio=largest_region_ratio,
        actual_color_count=actual_colors,
        requested_color_count=requested_colors,
        color_efficiency=color_efficiency,
        label_coverage=label_coverage,
        unlabeled_count=unlabeled,
        edge_violation_ratio=edge_violation_ratio,
        fill_coverage=fill_coverage,
    )
