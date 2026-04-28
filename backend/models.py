"""
Data models for the Paint By Number Generator.
Defines core data structures used throughout the processing pipeline.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon


@dataclass
class ProcessingJob:
    """Represents a single image processing job."""

    job_id: str
    status: str  # "processing" | "complete" | "failed"
    input_path: str
    output_svg: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ProcessingParameters:
    """Parameters for image processing."""

    num_colors: Optional[int] = None  # None for auto-detection
    simplification: float = 1.0  # 0.5-2.0 pixel tolerance
    use_watershed: bool = False  # Whether to use watershed segmentation (slower but original spec)

    # Advanced parameters
    use_slic: bool = True
    use_ciede2000: bool = True
    use_palette_merge: bool = True
    ciede2000_merge_thresh: float = 8.0
    use_thin_cleanup: bool = True
    min_region_width: int = 5
    use_shared_borders: bool = True
    label_mode: str = "polylabel"
    use_bezier_smooth: bool = False
    use_content_protect: bool = False
    use_budget_split: bool = False
    preset: str = "balanced"
    output_mode: str = "colored"

    def __post_init__(self):
        # Validate simplification range
        if not (0.5 <= self.simplification <= 2.0):
            raise ValueError("Simplification must be between 0.5 and 2.0")
        if not (2.0 <= self.ciede2000_merge_thresh <= 20.0):
            raise ValueError("ciede2000_merge_thresh must be between 2.0 and 20.0")
        if not (2 <= self.min_region_width <= 20):
            raise ValueError("min_region_width must be between 2 and 20")
        if self.label_mode not in ["centroid", "polylabel"]:
            raise ValueError("label_mode must be 'centroid' or 'polylabel'")


@dataclass
class ImageMetadata:
    """Metadata about the input image."""

    width: int
    height: int
    channels: int
    file_size: int
    image_type: str  # "portrait" | "landscape"


@dataclass
class ColorPalette:
    """Color palette information."""

    colors: np.ndarray  # LAB color values
    hex_colors: List[str]  # Hex representation for SVG
    color_count: int


@dataclass
class RegionData:
    """Data about segmented regions."""

    regions: Dict[int, Polygon]  # Region ID -> Polygon
    shared_borders: Dict[int, List[LineString]]  # Shared border segments
    adjacency_graph: nx.Graph  # Region adjacency


@dataclass
class LabelData:
    """Data about label placement."""

    positions: Dict[int, Point]  # Region ID -> Label position
    font_sizes: Dict[int, int]  # Region ID -> Font size
    skipped_regions: Set[int]  # Regions too small for labels


@dataclass
class SVGResult:
    """Final SVG generation result."""

    svg_content: str
    color_palette: ColorPalette
    processing_time: float
    region_count: int
    label_count: int
