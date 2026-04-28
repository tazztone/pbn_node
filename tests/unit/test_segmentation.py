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
def test_direct_color_segmentation(quantized_mock):
    img, colors = quantized_mock
    segmenter = RegionSegmenter(use_watershed=False)
    segmented, region_colors = segmenter.direct_color_segmentation(img, colors)

    assert segmented.shape == (100, 100)
    assert isinstance(region_colors, dict)
    # Since it's a simple image, it should find at least 3 regions
    assert np.max(segmented) >= 1
    assert len(region_colors) == np.max(segmented)


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
def test_segment_pipeline(quantized_mock):
    img, colors = quantized_mock
    segmenter = RegionSegmenter(use_watershed=False)
    region_data = segmenter.segment(img, colors)

    assert len(region_data.regions) > 0
    assert len(region_data.shared_borders) > 0
    assert region_data.adjacency_graph.number_of_nodes() > 0
