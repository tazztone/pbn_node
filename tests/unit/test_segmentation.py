import cv2
import numpy as np
import pytest
from pbn_node.backend.segmentation.segmenter import RegionSegmenter


@pytest.fixture
def quantized_mock():
    """Create a simple 3-color quantized image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [255, 0, 0]  # Red
    img[50:, 50:] = [0, 255, 0]  # Green
    img[:50, 50:] = [0, 0, 255]  # Blue

    colors = np.array(
        [
            cv2.cvtColor(np.array([[[255, 0, 0]]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0],
            cv2.cvtColor(np.array([[[0, 255, 0]]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0],
            cv2.cvtColor(np.array([[[0, 0, 255]]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0],
        ],
        dtype=np.float32,
    )

    return img, colors


@pytest.mark.unit
def test_segment_single_color():
    """Test segmentation with a single color image to ensure base case (1 region) is handled."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = [255, 0, 0]  # Red

    # Use standard LAB colors
    colors = np.array(
        [
            cv2.cvtColor(np.array([[[255, 0, 0]]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0],
        ],
        dtype=np.float32,
    )

    segmenter = RegionSegmenter()
    region_data = segmenter.segment(img, colors)

    assert region_data.segmented_matrix.shape == (10, 10)
    assert isinstance(region_data.region_colors, dict)
    assert np.max(region_data.segmented_matrix) == 1
    assert len(region_data.region_colors) == 1
    assert list(region_data.region_colors.values())[0] == 0


@pytest.mark.unit
def test_build_adjacency_graph():
    # 2x2 image with 2 regions
    regions = np.array([[1, 1], [2, 2]], dtype=np.int32)

    segmenter = RegionSegmenter()
    graph = segmenter.build_adjacency_graph(regions)

    assert graph.has_node(1)
    assert graph.has_node(2)
    assert graph.has_edge(1, 2)


@pytest.mark.unit
def test_build_adjacency_graph_single_region():
    # 2x2 image with a single region
    regions = np.array([[1, 1], [1, 1]], dtype=np.int32)

    segmenter = RegionSegmenter()
    graph = segmenter.build_adjacency_graph(regions)

    assert graph.number_of_nodes() == 1
    assert graph.has_node(1)
    assert graph.number_of_edges() == 0


@pytest.mark.unit
def test_build_adjacency_graph_empty():
    # 2x2 empty region (all 0s, typical for background/invalid)
    regions = np.zeros((2, 2), dtype=np.int32)

    segmenter = RegionSegmenter()
    graph = segmenter.build_adjacency_graph(regions)

    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


@pytest.mark.unit
def test_segment_pipeline(quantized_mock):
    img, colors = quantized_mock
    segmenter = RegionSegmenter()
    region_data = segmenter.segment(img, colors)

    assert len(region_data.regions) > 0
    assert len(region_data.shared_borders) > 0
    assert region_data.adjacency_graph.number_of_nodes() > 0
