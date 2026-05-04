"""
Region segmentation module implementing direct color segmentation and shared border segmentation.
"""

from collections import deque
from typing import cast

import cv2
import networkx as nx
import numpy as np
import skimage.color
from shapely.geometry import LineString

from ..models import RegionData
from ..utils.color import cv_to_std_lab


class RegionSegmenter:
    """
    Implements region segmentation using watershed transform and the novel
    Shared Border Segmentation algorithm to eliminate gaps between regions.
    """

    def __init__(
        self,
        use_ciede2000: bool = True,
        use_thin_cleanup: bool = True,
        min_region_width: int = 5,
        edge_weight_map: np.ndarray | None = None,
        lineart_strength: float = 0.0,
        smoothing_kernel_size: int = 9,
    ):
        """
        Initialize segmenter with configuration options.

        Args:
            lineart_strength: How much to trust the edge map [0, 1]
        """
        self.min_distance = 10  # Minimum distance for peak detection
        self.use_ciede2000 = use_ciede2000
        self.use_thin_cleanup = use_thin_cleanup
        self.min_region_width = min_region_width
        self.edge_weight_map = edge_weight_map
        self.lineart_strength = lineart_strength
        self.smoothing_kernel_size = smoothing_kernel_size

    def direct_color_segmentation(self, quantized: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
        """
        Direct color-based segmentation following pbnify's approach.
        """
        h, w = quantized.shape[:2]

        # Ensure edge_weight_map matches the image dimensions
        if self.edge_weight_map is not None and self.edge_weight_map.shape[:2] != (h, w):
            self.edge_weight_map = cv2.resize(self.edge_weight_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Step 1: Convert quantized BGR image to color ID matrix
        color_id_matrix = self._create_color_id_matrix(quantized, colors)

        # Step 2: Apply multi-pass median filter to remove pixel-level quantization noise
        # Using multiple passes with increasing kernel sizes is much more effective
        # for images of varying resolutions than a single 3x3 pass.
        if color_id_matrix.max() < 256:
            # OpenCV medianBlur requires uint8 or float32
            color_id_matrix = color_id_matrix.astype(np.uint8)
            for k in [3, 5, 7]:
                color_id_matrix = cv2.medianBlur(color_id_matrix, k)
            color_id_matrix = color_id_matrix.astype(np.int32)

        # Step 3: Apply vectorized majority filter (smoothing)
        smoothed = self._smooth_pbnify_vectorized(color_id_matrix)

        # Step 2.5: Thin-region scanline removal
        if self.use_thin_cleanup:
            smoothed = self._thin_region_cleanup(smoothed, self.min_region_width)

        # Step 3: Apply pbnify's getLabelLocs function (flood-fill + region filtering)
        regions_matrix, region_colors = self._get_regions_pbnify(smoothed)

        return regions_matrix, region_colors

    def _thin_region_cleanup(self, mat: np.ndarray, min_width: int) -> np.ndarray:
        """
        Horizontal and vertical scanline pass to merge ribbon regions
        below min_width with their dominant neighbor.
        """
        h, w = mat.shape
        cleaned = mat.copy()

        # Horizontal scanline
        for y in range(h):
            row = cleaned[y, :]
            # Find run lengths
            runs = []
            start = 0
            for x in range(1, w):
                if row[x] != row[x - 1]:
                    runs.append((start, x - 1, row[start]))
                    start = x
            runs.append((start, w - 1, row[start]))

            for start, end, _ in runs:
                width = end - start + 1
                if width < min_width:
                    # Edge veto: skip merge if run crosses strong edge
                    if self.edge_weight_map is not None:
                        edge_vals = self.edge_weight_map[y, start : end + 1]
                        # Veto threshold scales with lineart_strength:
                        # strong lineart (1.0) -> low threshold (0.1) -> easy veto
                        # weak lineart (0.0) -> high threshold (0.8) -> hard veto
                        veto_threshold = max(0.1, 0.8 - (self.lineart_strength * 0.7))
                        if np.max(edge_vals) > veto_threshold:
                            continue

                    # Find dominant neighbor
                    left_val = row[start - 1] if start > 0 else -1
                    right_val = row[end + 1] if end < w - 1 else -1

                    if left_val != -1 and right_val != -1:
                        # Count total occurrences in row to pick dominant, or just pick left for
                        # simplicity.
                        row[start : end + 1] = left_val
                    elif left_val != -1:
                        row[start : end + 1] = left_val
                    elif right_val != -1:
                        row[start : end + 1] = right_val

            cleaned[y, :] = row

        # Vertical scanline
        for x in range(w):
            col = cleaned[:, x]
            # Find run lengths
            runs = []
            start = 0
            for y in range(1, h):
                if col[y] != col[y - 1]:
                    runs.append((start, y - 1, col[start]))
                    start = y
            runs.append((start, h - 1, col[start]))

            for start, end, _ in runs:
                width = end - start + 1
                if width < min_width:
                    # Edge veto: skip merge if run crosses strong edge
                    if self.edge_weight_map is not None:
                        edge_vals = self.edge_weight_map[start : end + 1, x]
                        veto_threshold = max(0.1, 0.8 - (self.lineart_strength * 0.7))
                        if np.max(edge_vals) > veto_threshold:
                            continue

                    # Find dominant neighbor
                    top_val = col[start - 1] if start > 0 else -1
                    bottom_val = col[end + 1] if end < h - 1 else -1

                    if top_val != -1 and bottom_val != -1:
                        col[start : end + 1] = top_val
                    elif top_val != -1:
                        col[start : end + 1] = top_val
                    elif bottom_val != -1:
                        col[start : end + 1] = bottom_val

            cleaned[:, x] = col

        return cleaned

    def _create_color_id_matrix(self, quantized: np.ndarray, colors: np.ndarray) -> np.ndarray:
        """
        Convert quantized BGR image to color ID matrix.
        """
        h, w = quantized.shape[:2]

        # Convert quantized image to LAB for distance calculation
        quantized_lab = cv2.cvtColor(quantized, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Vectorized distance calculation
        # Reshape image to (H*W, 3)
        pixels = quantized_lab.reshape(-1, 3)
        if self.use_ciede2000:
            std_pixels = cv_to_std_lab(pixels)
            std_colors = cv_to_std_lab(colors)

            # (H*W, 3) and (K, 3) -> we need dists (H*W, K)
            k_total = colors.shape[0]
            dists = np.zeros((pixels.shape[0], k_total), dtype=np.float32)

            for k in range(k_total):
                # ciede2000 takes (..., 3) arrays
                # Ensure the color array has at least 2 dimensions for correct broadcasting
                dists[:, k] = skimage.color.deltaE_ciede2000(std_pixels, std_colors[[k]])
        else:
            # Calculate squared Euclidean distances to each color center
            # (H*W, 1, 3) - (1, K, 3) -> (H*W, K, 3) -> sum -> (H*W, K)
            diffs = pixels[:, np.newaxis, :] - colors[np.newaxis, :, :]
            dists = np.sum(diffs**2, axis=2)

        # Find closest color ID for each pixel
        closest_ids = np.argmin(dists, axis=1) + 1  # 1-indexed

        return closest_ids.reshape(h, w).astype(np.int32)

    def _smooth_pbnify_vectorized(self, mat: np.ndarray) -> np.ndarray:
        """
        Vectorized implementation of pbnify's smooth function.
        Uses majority voting in 9x9 neighborhoods.
        """
        h, w = mat.shape
        unique_ids = np.unique(mat)

        # If only one color, no smoothing needed
        if len(unique_ids) <= 1:
            return mat

        counts = np.zeros((len(unique_ids), h, w), dtype=np.float32)

        # Calculate dynamic kernel size: ~1% of image diagonal, min 9
        # This ensures smoothing scales correctly with resolution.
        diagonal = int(np.sqrt(h**2 + w**2))
        k_size = max(9, (diagonal // 100) | 1)  # ensure odd

        kernel = np.ones((k_size, k_size), dtype=np.float32)

        # Build per-pixel attenuation from lineart
        # 1.0 = full vote, 0.0 = no vote (strong edge)
        attenuation = None
        if self.edge_weight_map is not None and self.lineart_strength > 0:
            attenuation = (1.0 - (self.edge_weight_map * self.lineart_strength)).astype(np.float32)

        # For each color, count its neighbors in a 9x9 window
        for i, val in enumerate(unique_ids):
            mask = (mat == val).astype(np.float32)
            if attenuation is not None:
                mask *= attenuation  # edge pixels contribute less to neighbor votes
            # Use OpenCV's filter2D for fast convolution
            counts[i] = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REFLECT)

        # Get the ID with the maximum count at each pixel
        idx = np.argmax(counts, axis=0)
        smoothed = unique_ids[idx]

        # Phase 1 refinement: preserve original colors at strong edges
        # This prevents the majority vote from 'eating' thin details on lines
        if self.edge_weight_map is not None and self.lineart_strength > 0:
            # Scale threshold based on lineart_strength:
            # strength 1.0 -> threshold 0.2 (very protective)
            # strength 0.0 -> threshold 1.0 (no protection)
            threshold = 1.0 - (self.lineart_strength * 0.8)
            edge_mask = self.edge_weight_map > threshold
            smoothed[edge_mask] = mat[edge_mask]

        return smoothed

    def _get_regions_pbnify(self, mat: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
        """
        Exact implementation of pbnify's getLabelLocs logic.
        Finds connected regions and removes small ones or regions with severe convexity deficits.
        """
        h, w = mat.shape
        covered = np.zeros((h, w), dtype=bool)
        regions_matrix = np.zeros((h, w), dtype=np.int32)
        region_colors = {}
        next_region_id = 1
        # Minimum region size is 1 here; we let the Vectorizer handle
        # speckle removal via merging to avoid leaving coverage gaps.
        min_region_size = 1

        for y in range(h):
            for x in range(w):
                if not covered[y, x]:
                    # Capture original color index (0-based) from mat
                    # mat contains 1-indexed color IDs from _create_color_id_matrix
                    color_idx = int(mat[y, x]) - 1

                    # Get connected region
                    region = self._get_region_pbnify(mat, covered, x, y, w, h)

                    keep_region = len(region["x"]) > min_region_size

                    # Convexity deficit filter - removed to prioritize coverage
                    # over strict shape constraints.

                    if keep_region:
                        # Keep this region - assign it a region ID
                        for i in range(len(region["x"])):
                            px, py = region["x"][i], region["y"][i]
                            regions_matrix[py, px] = next_region_id

                        region_colors[next_region_id] = color_idx
                        next_region_id += 1
                    else:
                        # Remove small or convoluted region (merge with neighbor)
                        self._remove_region_pbnify(mat, region)

        return regions_matrix, region_colors

    def _get_region_pbnify(self, mat: np.ndarray, covered: np.ndarray, x: int, y: int, width: int, height: int) -> dict:
        """
        Implementation of pbnify's getRegion function.
        Uses deque-based flood fill for O(1) pops.
        """
        region = {"value": mat[y, x], "x": [], "y": []}
        value = mat[y, x]

        # Queue-based flood fill
        queue = deque([[x, y]])

        while queue:
            coord = queue.popleft()  # O(1)
            cx, cy = coord[0], coord[1]

            if not covered[cy, cx] and mat[cy, cx] == value:
                region["x"].append(cx)
                region["y"].append(cy)
                covered[cy, cx] = True  # Update the main covered array

                # Add 4-connected neighbors
                if cx > 0:
                    queue.append([cx - 1, cy])
                if cx < width - 1:
                    queue.append([cx + 1, cy])
                if cy > 0:
                    queue.append([cx, cy - 1])
                if cy < height - 1:
                    queue.append([cx, cy + 1])

        return region

    def _remove_region_pbnify(self, mat: np.ndarray, region: dict):
        """
        Implementation of pbnify's removeRegion function, enhanced with edge-awareness.
        Merges small region with the neighbor separated by the WEAKEST edge.
        """
        if not region["x"]:
            return

        x_coords = np.array(region["x"])
        y_coords = np.array(region["y"])
        region_value = region["value"]
        h, w = mat.shape

        # Find all neighbor pixels (4-connected) that are not part of this region
        neighbor_votes: dict[int, float] = {}  # (value) -> min_edge_weight

        # Check all boundary pixels of the region
        for i in range(len(x_coords)):
            x, y = x_coords[i], y_coords[i]

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    neighbor_val = mat[ny, nx]
                    if neighbor_val != region_value:
                        # Get edge weight at this boundary
                        edge_val = 0.0
                        if self.edge_weight_map is not None:
                            # Use max of the two pixels as the boundary weight
                            edge_val = max(self.edge_weight_map[y, x], self.edge_weight_map[ny, nx])

                        if neighbor_val not in neighbor_votes or edge_val < neighbor_votes[neighbor_val]:
                            neighbor_votes[neighbor_val] = edge_val

        # Pick neighbor with the weakest boundary edge
        if neighbor_votes:
            best_neighbor = min(neighbor_votes, key=lambda k: neighbor_votes[k])
            min_edge = neighbor_votes[best_neighbor]

            # Hard Veto: If the weakest boundary is still a strong edge,
            # don't merge. This preserves small geometric details.
            # Scaling: higher strength makes the veto more sensitive.
            veto_threshold = max(0.2, 0.9 - (self.lineart_strength * 0.8))
            if min_edge > veto_threshold:
                return  # Skip merge! Preserve this small detail.

            new_value = best_neighbor
        else:
            # Fallback to original pbnify logic if no neighbors found
            new_value = region_value
            if y_coords[0] > 0:
                new_value = mat[y_coords[0] - 1, x_coords[0]]
            else:
                y = y_coords[0]
                while y < h and mat[y, x_coords[0]] == region_value:
                    y += 1
                if y < h:
                    new_value = mat[y, x_coords[0]]

        # Assign all pixels in the region to the new value
        mat[y_coords, x_coords] = new_value

    def build_adjacency_graph(self, regions: np.ndarray) -> nx.Graph:
        """
        Build adjacency graph of regions using vectorized shifts.
        """
        graph = nx.Graph()
        region_ids = np.unique(regions)
        region_ids = region_ids[region_ids > 0]
        for region_id in region_ids:
            graph.add_node(int(region_id))

        # Horizontal adjacency
        h_adj = (regions[:, :-1] != regions[:, 1:]) & (regions[:, :-1] > 0) & (regions[:, 1:] > 0)
        if np.any(h_adj):
            pairs = np.column_stack((regions[:, :-1][h_adj], regions[:, 1:][h_adj]))
            # Unique pairs to reduce graph add_edge calls
            unique_pairs = np.unique(np.sort(pairs, axis=1), axis=0)
            for u, v in unique_pairs:
                graph.add_edge(int(u), int(v))

        # Vertical adjacency
        v_adj = (regions[:-1, :] != regions[1:, :]) & (regions[:-1, :] > 0) & (regions[1:, :] > 0)
        if np.any(v_adj):
            pairs = np.column_stack((regions[:-1, :][v_adj], regions[1:, :][v_adj]))
            unique_pairs = np.unique(np.sort(pairs, axis=1), axis=0)
            for u, v in unique_pairs:
                graph.add_edge(int(u), int(v))

        return graph

    def shared_border_segmentation(self, regions: np.ndarray) -> dict[int, list[LineString]]:
        """
        Implement Shared Border Segmentation algorithm using vectorized shifts.
        """
        shared_borders: dict[int, list[LineString]] = {}
        region_ids = np.unique(regions)
        region_ids = region_ids[region_ids > 0]
        for region_id in region_ids:
            shared_borders[int(region_id)] = []

        # (region1, region2) -> points
        border_segments: dict[tuple[int, int], list[tuple[float, float]]] = {}

        # Scan horizontally for borders
        h_adj = (regions[:, :-1] != regions[:, 1:]) & (regions[:, :-1] > 0) & (regions[:, 1:] > 0)
        if np.any(h_adj):
            y_coords, x_coords = np.where(h_adj)
            left_ids = regions[:, :-1][h_adj]
            right_ids = regions[:, 1:][h_adj]

            for i in range(len(y_coords)):
                pair = cast(tuple[int, int], tuple(sorted([int(left_ids[i]), int(right_ids[i])])))
                if pair not in border_segments:
                    border_segments[pair] = []
                border_segments[pair].append((float(x_coords[i] + 0.5), float(y_coords[i])))

        # Scan vertically for borders
        v_adj = (regions[:-1, :] != regions[1:, :]) & (regions[:-1, :] > 0) & (regions[1:, :] > 0)
        if np.any(v_adj):
            y_coords, x_coords = np.where(v_adj)
            top_ids = regions[:-1, :][v_adj]
            bottom_ids = regions[1:, :][v_adj]

            for i in range(len(y_coords)):
                pair = cast(tuple[int, int], tuple(sorted([int(top_ids[i]), int(bottom_ids[i])])))
                if pair not in border_segments:
                    border_segments[pair] = []
                border_segments[pair].append((float(x_coords[i]), float(y_coords[i] + 0.5)))

        # Convert border points to LineStrings
        for (region1, region2), points in border_segments.items():
            if len(points) >= 2:
                line = LineString(points)
                shared_borders[region1].append(line)
                shared_borders[region2].append(line)

        return shared_borders

    def segment(self, quantized: np.ndarray, colors: np.ndarray) -> RegionData:
        """
        Complete segmentation pipeline.

        Args:
            quantized: Quantized image (BGR format)
            colors: Color centers in LAB space

        Returns:
            RegionData with segmented regions, borders, and adjacency graph
        """
        # Centralized resizing of edge weight map
        if self.edge_weight_map is not None:
            h, w = quantized.shape[:2]
            if self.edge_weight_map.shape[:2] != (h, w):
                self.edge_weight_map = cv2.resize(self.edge_weight_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Initialize region colors mapping
        region_colors: dict[int, int] = {}

        # High-quality PBNify-inspired segmentation
        color_id_matrix = self._create_color_id_matrix(quantized, colors)
        smoothed = self._smooth_pbnify_vectorized(color_id_matrix)
        segmented, region_colors = self._get_regions_pbnify(smoothed)

        # Build adjacency graph
        adjacency_graph = self.build_adjacency_graph(segmented)

        # Apply shared border segmentation
        shared_borders = self.shared_border_segmentation(segmented)

        # Create polygons from segmented regions
        from shapely.geometry import Polygon

        regions = {}
        region_ids = np.unique(segmented)
        region_ids = region_ids[region_ids > 0]

        for region_id in region_ids:
            # Create mask for this region
            mask = (segmented == region_id).astype(np.uint8) * 255

            # Find contours - use CHAIN_APPROX_SIMPLE for efficiency
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Use the largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Convert to polygon - need at least 3 points
                if len(largest_contour) >= 3:
                    # We skip simplification here and let the Vectorizer handle it
                    # using the topology-preserving Visvalingam-Whyatt algorithm.
                    points = largest_contour.reshape(-1, 2).tolist()

                    # Need at least 3 unique points for a polygon
                    if len(points) >= 3:
                        try:
                            polygon = Polygon(points)

                            # Fix invalid polygons
                            if not polygon.is_valid:
                                # Try to fix with buffer(0) trick first
                                polygon = polygon.buffer(0)

                            # Extract polygon from result (buffer can return MultiPolygon)
                            final_polygon = None

                            if polygon.geom_type == "Polygon":
                                final_polygon = polygon
                            elif polygon.geom_type == "GeometryCollection":
                                # Get all polygons from collection
                                polygons = [geom for geom in polygon.geoms if geom.geom_type == "Polygon"]
                                if polygons:
                                    final_polygon = max(polygons, key=lambda p: p.area)
                            elif polygon.geom_type == "MultiPolygon":
                                # Get largest polygon from multipolygon
                                final_polygon = max(polygon.geoms, key=lambda p: p.area)

                            # Only add if it's a valid polygon with area
                            if (
                                final_polygon
                                and final_polygon.geom_type == "Polygon"
                                and final_polygon.is_valid
                                and final_polygon.area > 0
                            ):
                                regions[int(region_id)] = final_polygon
                        except Exception:
                            # Skip regions that can't be converted to valid polygons
                            pass

        return RegionData(
            regions=regions,
            region_colors=region_colors,
            shared_borders=shared_borders,
            adjacency_graph=adjacency_graph,
            segmented_matrix=segmented,
        )
