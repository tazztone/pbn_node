from pathlib import Path

import cv2
import numpy as np
import pytest

from pbn_node.backend.models import PerceptionInputs, ProcessingParameters
from pbn_node.pbn_pipeline import ImageProcessor

from .metrics import analyze

EXAMPLE_DIR = Path(__file__).parents[2] / "example_inputs"


def load_fixture(name, gray=False):
    path = EXAMPLE_DIR / name
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    if gray:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.imread(str(path))


@pytest.mark.quality
@pytest.mark.parametrize("color_count", [8, 16, 24])
def test_region_quality_baseline(color_count):
    """
    Baseline quality test using boat fixture.
    Ensures that for various color counts, basic sanity metrics hold.
    """
    img = load_fixture("boat.webp")
    processor = ImageProcessor()
    params = ProcessingParameters(num_colors=color_count, min_region_size=100)

    result = processor.process_array(img, params)
    report = analyze(result, img.shape, requested_colors=color_count)

    # Calibrated baseline expectations per color count to prevent loose guards.
    # TODO: While these thresholds match the current performance baseline, they are
    # still high (allowing nearly 3/4 of regions to be specks). As the island removal
    # and aggregation segmenter pipelines are optimized, these thresholds should
    # be tightened to converge toward < 0.30.
    expectations = {
        8: {"max_speck_ratio": 0.82, "min_label_coverage": 0.58},
        16: {"max_speck_ratio": 0.77, "min_label_coverage": 0.52},
        24: {"max_speck_ratio": 0.74, "min_label_coverage": 0.48},
    }
    exp = expectations[color_count]

    assert report.speck_ratio < exp["max_speck_ratio"], f"Too many specks: {report.speck_ratio:.1%}"
    assert report.render_coverage > 0.95, f"Low visual render coverage: {report.render_coverage:.1%}"
    assert report.fill_coverage > 0.90, f"Low geometric fill coverage: {report.fill_coverage:.1%}"
    assert report.label_coverage > exp["min_label_coverage"], f"Low label coverage: {report.label_coverage:.1%}"
    assert report.actual_color_count <= color_count

    # Geometric/Render consistency invariants
    assert report.render_coverage >= report.fill_coverage, "Visual coverage must equal or exceed geometric coverage."
    delta = report.render_coverage - report.fill_coverage
    assert delta < 0.05, f"Excessive renderer patching (delta > 5pp): {delta:.1%}"


@pytest.mark.quality
def test_lineart_edge_fidelity():
    """
    Verifies that lineart perception input improves edge fidelity.
    """
    img = load_fixture("boat.webp")
    lineart = load_fixture("boat_lineart.webp", gray=True).astype(np.float32) / 255.0

    processor = ImageProcessor()

    # Run with lineart
    perception = PerceptionInputs(lineart=lineart, lineart_strength=0.8, edge_influence=0.5)
    params = ProcessingParameters(num_colors=16, perception=perception)

    result = processor.process_array(img, params)
    report = analyze(result, img.shape, requested_colors=16, lineart=lineart)

    assert report.edge_violation_ratio is not None
    assert report.edge_violation_ratio < 0.55, f"Poor lineart edge fidelity: {report.edge_violation_ratio:.1%}"


@pytest.mark.quality
@pytest.mark.parametrize("map_name", ["boat_cannyedge.webp", "boat_HED.webp"])
def test_alternative_maps_fidelity(map_name):
    """
    Verifies alternative edge maps (Canny, HED) also work with the loop.
    """
    img = load_fixture("boat.webp")
    edge_map = load_fixture(map_name, gray=True).astype(np.float32) / 255.0

    processor = ImageProcessor()
    perception = PerceptionInputs(lineart=edge_map, lineart_strength=0.8, edge_influence=0.5)
    params = ProcessingParameters(num_colors=16, perception=perception)

    result = processor.process_array(img, params)
    report = analyze(result, img.shape, requested_colors=16, lineart=edge_map)

    assert report.edge_violation_ratio is not None
    assert report.edge_violation_ratio < 0.55, f"Poor edge fidelity for {map_name}: {report.edge_violation_ratio:.1%}"
