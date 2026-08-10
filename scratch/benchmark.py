import time
import numpy as np
from pbn_node.backend.segmentation.segmenter import RegionSegmenter

def benchmark_remove_region():
    # Setup
    segmenter = RegionSegmenter(lineart_strength=0.0)

    # Create a dummy matrix and region
    mat = np.zeros((1000, 1000), dtype=np.int32)

    # Create a large region to make the loop noticeable
    N = 100000
    x_coords = np.random.randint(1, 999, size=N).tolist()
    y_coords = np.random.randint(1, 999, size=N).tolist()

    for x, y in zip(x_coords, y_coords):
        mat[y, x] = 1

    region = {"value": 1, "x": x_coords, "y": y_coords}

    # Warmup
    segmenter._remove_region_pbnify(mat, region)

    # Benchmark
    start = time.perf_counter()
    iterations = 10
    for _ in range(iterations):
        segmenter._remove_region_pbnify(mat, region)
    end = time.perf_counter()

    print(f"Time taken for {iterations} iterations: {end - start:.4f} seconds")

if __name__ == "__main__":
    benchmark_remove_region()
