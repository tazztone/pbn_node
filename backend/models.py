"""
Data models for the Paint By Number Generator.
Defines core data structures used throughout the processing pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon


@dataclass
class ProcessingJob:
    """Represents a single image processing job."""

    job_id: str
    status: str  # "processing" | "complete" | "failed"
    input_path: str
    output_svg: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PerceptionInputs:
    """Container for perception-stack inputs like albedo, segmentation, and normals."""

    albedo: np.ndarray | None = None
    segmentation_mask: np.ndarray | None = None
    normal_map: np.ndarray | None = None
    lineart: np.ndarray | None = None  # [H,W] float32 [0,1] edge weight map
    lineart_strength: float = 0.7
    invert_lineart: bool = False
    background_ids: list[int] = field(default_factory=lambda: [0])
    subject_priority: float = 2.0
    edge_influence: float = 0.3
    material_weight: float = 0.5  # Blend factor between albedo and original photo
    use_auto_mask: bool = False  # Whether to generate an Otsu mask if segmentation_mask is None

    def __post_init__(self):
        if not (0.0 <= self.material_weight <= 1.0):
            raise ValueError("material_weight must be between 0.0 and 1.0")
        if not (1.0 <= self.subject_priority <= 10.0):
            raise ValueError("subject_priority must be >= 1.0")
        if not (0.0 <= self.edge_influence <= 1.0):
            raise ValueError("edge_influence must be between 0.0 and 1.0")
        if not (0.0 <= self.lineart_strength <= 1.0):
            raise ValueError("lineart_strength must be between 0.0 and 1.0")


@dataclass
class ProcessingParameters:
    """Parameters for image processing."""

    num_colors: int | None = None  # None for auto-detection
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
    label_mode: Literal["centroid", "polylabel"] = "polylabel"
    use_bezier_smooth: bool = False
    use_content_protect: bool = False
    perception: PerceptionInputs | None = None
    preset: str = "balanced"
    output_mode: str = "colored"
    use_auto_albedo: bool = False
    use_painterly_preprocess: bool = False
    painterly_sigma_s: float = 60.0
    painterly_sigma_r: float = 0.45

    def __post_init__(self):
        # Validate simplification range
        if not (0.5 <= self.simplification <= 2.0):
            raise ValueError("Simplification must be between 0.5 and 2.0")
        if not (2.0 <= self.ciede2000_merge_thresh <= 20.0):
            raise ValueError("ciede2000_merge_thresh must be between 2.0 and 20.0")
        if not (2 <= self.min_region_width <= 20):
            raise ValueError("min_region_width must be between 2 and 20")


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
    hex_colors: list[str]  # Hex representation for SVG
    color_count: int


@dataclass
class RegionData:
    """Data about segmented regions."""

    regions: dict[int, Polygon]  # Region ID -> Polygon
    region_colors: dict[int, int] = field(
        default_factory=dict
    )  # Region ID -> Color Index (0-based)
    shared_borders: dict[int, list[LineString]] = field(
        default_factory=dict
    )  # Shared border segments
    adjacency_graph: nx.Graph = field(default_factory=nx.Graph)  # Region adjacency


@dataclass
class LabelData:
    """Data about label placement."""

    positions: dict[int, Point]  # Region ID -> Label position
    font_sizes: dict[int, int]  # Region ID -> Font size
    skipped_regions: set[int]  # Regions too small for labels


@dataclass
class SVGResult:
    """Final SVG generation result."""

    svg_content: str
    color_palette: ColorPalette
    processing_time: float
    region_count: int
    label_count: int
