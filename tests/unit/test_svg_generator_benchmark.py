import time

import numpy as np
import pytest
from pbn_node.backend.models import ColorPalette, LabelData
from pbn_node.backend.svg_generation.svg_generator import SVGGenerator
from shapely.geometry import Point, Polygon


@pytest.mark.unit
def test_benchmark_svg_generator():
    generator = SVGGenerator()

    # 10 colors
    num_colors = 10
    colors_array = np.random.randint(0, 255, size=(num_colors, 3))
    hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors_array]
    colors = ColorPalette(colors=colors_array, hex_colors=hex_colors, color_count=num_colors)

    # 50,000 regions to amplify the loop overhead
    num_regions = 50000
    regions = {}
    positions = {}
    font_sizes = {}
    region_colors = {}

    for i in range(1, num_regions + 1):
        regions[i] = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        positions[i] = Point(0.5, 0.5)
        font_sizes[i] = 12
        region_colors[i] = (i - 1) % num_colors

    labels = LabelData(positions=positions, font_sizes=font_sizes, skipped_regions=set())

    # Run once to warm up
    _ = generator.generate_svg(regions, labels, colors, region_colors, print_mode=False)

    # Run benchmark
    iterations = 5
    start = time.time()
    for _ in range(iterations):
        _ = generator.generate_svg(regions, labels, colors, region_colors, print_mode=False)

    end = time.time()
    avg_time = (end - start) / iterations
    print(f"\nAverage time over {iterations} iterations: {avg_time:.4f} seconds")
