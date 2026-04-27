"""
Vectorization module implementing contour detection, Visvalingam-Whyatt simplification,
and speckle removal.
"""
import cv2
import numpy as np
from typing import List, Dict
from shapely.geometry import Polygon
from shapely.ops import unary_union
import heapq

from ..models import RegionData


class Vectorizer:
    """
    Implements vectorization operations including contour detection,
    topology-preserving simplification, and speckle removal.
    """
    
    def __init__(self):
        """Initialize vectorizer with default parameters."""
        self.speckle_threshold = 0.001  # 0.1% of total area
    
    def find_contours(self, region_mask: np.ndarray) -> List[np.ndarray]:
        """
        Find contours using OpenCV.
        
        Args:
            region_mask: Binary mask for a single region
            
        Returns:
            List of contours as numpy arrays
        """
        contours, _ = cv2.findContours(
            region_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def visvalingam_whyatt(self, contour: np.ndarray, tolerance: float) -> np.ndarray:
        """
        Apply Visvalingam-Whyatt algorithm for topology-preserving simplification.
        
        This algorithm progressively removes points with the smallest effective area
        until the desired tolerance is reached.
        
        Args:
            contour: Input contour as numpy array of shape (N, 1, 2) or (N, 2)
            tolerance: Minimum area threshold for point removal (in pixels²)
            
        Returns:
            Simplified contour
        """
        # Reshape contour to (N, 2) if needed
        if len(contour.shape) == 3:
            points = contour.reshape(-1, 2)
        else:
            points = contour.copy()
        
        # Need at least 3 points for a polygon
        if len(points) < 3:
            return contour
        
        # Calculate effective area for each point
        # Effective area = area of triangle formed by point and its neighbors
        def calculate_area(p1, p2, p3):
            """Calculate area of triangle formed by three points."""
            return abs(0.5 * (
                p1[0] * (p2[1] - p3[1]) +
                p2[0] * (p3[1] - p1[1]) +
                p3[0] * (p1[1] - p2[1])
            ))
        
        # Create list of points with their effective areas
        n = len(points)
        point_data = []
        
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            area = calculate_area(points[prev_idx], points[i], points[next_idx])
            point_data.append({
                'index': i,
                'area': area,
                'removed': False
            })
        
        # Use min-heap to efficiently find points with smallest area
        heap = [(pd['area'], i) for i, pd in enumerate(point_data)]
        heapq.heapify(heap)
        
        # Remove points with area below tolerance
        removed_count = 0
        max_removable = n - 3  # Keep at least 3 points
        
        while heap and removed_count < max_removable:
            area, idx = heapq.heappop(heap)
            
            # Skip if already removed or area exceeds tolerance
            if point_data[idx]['removed'] or area > tolerance:
                continue
            
            # Mark as removed
            point_data[idx]['removed'] = True
            removed_count += 1
            
            # Recalculate areas for neighbors
            # Find previous non-removed point
            prev_idx = idx
            while True:
                prev_idx = (prev_idx - 1) % n
                if not point_data[prev_idx]['removed']:
                    break
            
            # Find next non-removed point
            next_idx = idx
            while True:
                next_idx = (next_idx + 1) % n
                if not point_data[next_idx]['removed']:
                    break
            
            # Update areas for neighbors
            for neighbor_idx in [prev_idx, next_idx]:
                if point_data[neighbor_idx]['removed']:
                    continue
                
                # Find neighbors of neighbor
                nn_prev = neighbor_idx
                while True:
                    nn_prev = (nn_prev - 1) % n
                    if not point_data[nn_prev]['removed']:
                        break
                
                nn_next = neighbor_idx
                while True:
                    nn_next = (nn_next + 1) % n
                    if not point_data[nn_next]['removed']:
                        break
                
                new_area = calculate_area(
                    points[nn_prev],
                    points[neighbor_idx],
                    points[nn_next]
                )
                point_data[neighbor_idx]['area'] = new_area
                heapq.heappush(heap, (new_area, neighbor_idx))
        
        # Build simplified contour from remaining points
        simplified_points = [points[i] for i in range(n) if not point_data[i]['removed']]
        
        # Convert back to original shape
        if len(contour.shape) == 3:
            return np.array(simplified_points).reshape(-1, 1, 2)
        else:
            return np.array(simplified_points)
    
    def remove_speckles(self, regions: Dict[int, Polygon], 
                       colors: np.ndarray, threshold: float = 0.001) -> Dict[int, Polygon]:
        """
        Remove speckle regions smaller than threshold.
        
        Merges small regions into their nearest neighbor based on LAB color distance.
        
        Args:
            regions: Dictionary of region ID to Polygon
            colors: Color centers in LAB space
            threshold: Area threshold as fraction of total area (default 0.1%)
            
        Returns:
            Dictionary with speckles removed
        """
        # Calculate total area
        total_area = self.calculate_total_area(regions)
        min_area = total_area * threshold
        
        # Identify speckles
        speckles = []
        large_regions = {}
        
        for region_id, polygon in regions.items():
            if polygon.area < min_area:
                speckles.append((region_id, polygon))
            else:
                large_regions[region_id] = polygon
        
        # If no large regions, keep everything
        if not large_regions:
            return regions
        
        # Merge each speckle into nearest neighbor by color distance
        for speckle_id, speckle_polygon in speckles:
            if speckle_id >= len(colors):
                continue
            
            speckle_color = colors[speckle_id - 1]  # Assuming 1-indexed regions
            
            # Find nearest neighbor by LAB color distance
            min_distance = float('inf')
            nearest_id = None
            
            for region_id in large_regions.keys():
                if region_id >= len(colors):
                    continue
                
                region_color = colors[region_id - 1]
                distance = np.linalg.norm(speckle_color - region_color)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_id = region_id
            
            # Merge speckle into nearest neighbor
            if nearest_id is not None:
                try:
                    # Union the polygons
                    merged = unary_union([large_regions[nearest_id], speckle_polygon])
                    
                    # Ensure we have a single Polygon
                    if merged.geom_type == 'Polygon' and merged.is_valid:
                        large_regions[nearest_id] = merged
                    elif merged.geom_type == 'MultiPolygon':
                        # Take the largest polygon from the multipolygon
                        largest = max(merged.geoms, key=lambda p: p.area)
                        if largest.is_valid:
                            large_regions[nearest_id] = largest
                except:
                    # If merge fails, just skip this speckle
                    pass
        
        return large_regions
    
    def calculate_total_area(self, regions: Dict[int, Polygon]) -> float:
        """
        Calculate total area of all regions.
        
        Args:
            regions: Dictionary of region ID to Polygon
            
        Returns:
            Total area in square pixels
        """
        return sum(polygon.area for polygon in regions.values())
    
    def vectorize(self, region_data: RegionData, simplification: float) -> Dict[int, Polygon]:
        """
        Complete vectorization pipeline.
        
        Args:
            region_data: RegionData from segmentation
            simplification: Tolerance for simplification (0.5-2.0 pixels)
            
        Returns:
            Dictionary of simplified polygons
        """
        # Validate simplification parameter
        if not (0.5 <= simplification <= 2.0):
            raise ValueError("Simplification must be between 0.5 and 2.0")
        
        simplified_regions = {}
        
        for region_id, polygon in region_data.regions.items():
            try:
                # Get polygon exterior coordinates
                coords = np.array(polygon.exterior.coords)
                
                # Apply Visvalingam-Whyatt simplification
                # Convert tolerance from pixels to area (pixels²)
                tolerance_area = simplification ** 2
                simplified_coords = self.visvalingam_whyatt(coords, tolerance_area)
                
                # Create new polygon
                if len(simplified_coords) >= 3:
                    simplified_polygon = Polygon(simplified_coords)
                    
                    # Validate polygon
                    if simplified_polygon.is_valid and simplified_polygon.area > 0:
                        simplified_regions[region_id] = simplified_polygon
                    else:
                        # Keep original if simplification failed
                        simplified_regions[region_id] = polygon
                else:
                    # Keep original if too few points
                    simplified_regions[region_id] = polygon
            except Exception:
                # Keep original on any error
                simplified_regions[region_id] = polygon
        
        return simplified_regions
