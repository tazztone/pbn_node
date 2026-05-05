import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# Add project root and its parent to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
parent_dir = os.path.dirname(root_dir)
for p in [root_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Mock ComfyUI
from tests.mock_comfyui import install_comfyui_mocks  # noqa: E402

install_comfyui_mocks()

from pbn_node.backend.models import ProcessingParameters  # noqa: E402
from pbn_node.pbn_pipeline import ImageProcessor  # noqa: E402
from tests.quality.metrics import analyze  # noqa: E402

OUT_DIR = os.path.join(root_dir, "tests", "quality", "output")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "iteration_log.csv")


def main():
    parser = argparse.ArgumentParser(description="PBN Quick Iteration Tool")
    parser.add_argument("--image", default="boat.webp", help="Image filename in example_inputs/")
    parser.add_argument("--colors", type=int, default=32, help="Number of colors")
    parser.add_argument("--simplification", type=float, default=1.0, help="Simplification tolerance")
    parser.add_argument("--smoothing", type=int, default=9, help="Smoothing kernel size")
    parser.add_argument("--min-width", type=int, default=5, help="Min region width for scanline cleanup")
    parser.add_argument("--min-region-size", type=int, default=None, help="Explicit min region size for segmenter")
    parser.add_argument("--tag", default="latest", help="Tag for the output filename")
    args = parser.parse_args()

    # 1. Load Image
    img_path = os.path.join(root_dir, "example_inputs", args.image)
    if not os.path.exists(img_path):
        print(f"ERROR: Image not found at {img_path}")
        sys.exit(1)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 2. Process
    processor = ImageProcessor()
    params = ProcessingParameters(
        num_colors=args.colors,
        simplification=args.simplification,
        smoothing_kernel_size=args.smoothing,
        min_region_width=args.min_width,
        min_region_size=args.min_region_size,
    )

    print(f"--- Running Iteration: {args.tag} ---")
    print(
        f"Settings: Colors={args.colors}, Simpl={args.simplification}, Smooth={args.smoothing}, MinW={args.min_width}"
    )

    start_time = time.time()
    result = processor.process_array(img, params)
    elapsed = time.time() - start_time

    # 3. Analyze
    report = analyze(result, img.shape, requested_colors=args.colors)

    # 4. Create Diagnostic Panels
    # Scale down for report if very large, but keep enough resolution for analysis
    max_w = 1200
    scale = min(1.0, max_w / w)
    dw, dh = int(w * scale), int(h * scale)

    def res(image):
        return cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA)

    # Panel 1: Source
    p1 = res(img)
    cv2.putText(p1, "SOURCE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Panel 2: Result
    p2 = res(result.quantized)
    cv2.putText(p2, "RESULT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Panel 3: Gap Map (Red where black)
    p3_res = res(result.quantized)
    p3_img = res(img)
    p3_gap_mask = np.all(p3_res == 0, axis=2)
    p3 = (p3_img * 0.2).astype(np.uint8)
    p3[p3_gap_mask] = [0, 0, 255]
    cv2.putText(p3, "GAP MAP (RED)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Panel 4: Boundary Overlay (Draw from polygon data)
    boundary_overlay = result.quantized.copy()
    p4 = res(boundary_overlay)
    for _rid, poly in result.cleaned_regions.items():
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        # Draw on the scaled image for the report
        # Note: We need to scale the points too
        scaled_pts = (pts * scale).astype(np.int32)
        cv2.polylines(p4, [scaled_pts], isClosed=True, color=(0, 255, 0), thickness=1)

    cv2.putText(p4, "BOUNDARIES (DATA)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Combine into 2x2 grid
    top = np.hstack((p1, p2))
    bottom = np.hstack((p3, p4))
    grid = np.vstack((top, bottom))

    # Metrics Bar
    bar_h = 80
    bar = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
    metrics_text = (
        f"Tag: {args.tag} | Regions: {report.total_regions} | "
        f"Fill: {report.fill_coverage:.2%} | "
        f"Speck: {report.speck_ratio:.1%} | Time: {elapsed:.2f}s"
    )
    settings_text = (
        f"Params: colors={args.colors}, simpl={args.simplification}, "
        f"smooth={args.smoothing}, min_w={args.min_width}, "
        f"min_reg={args.min_region_size}"
    )

    cv2.putText(bar, metrics_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(bar, settings_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    final_img = np.vstack((grid, bar))

    # 5. Save and Log
    out_path = os.path.join(OUT_DIR, f"iter_{args.tag}.png")
    cv2.imwrite(out_path, final_img)
    print(f"SAVED: {out_path}")

    # Log to CSV
    log_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(
                [
                    "Timestamp",
                    "Tag",
                    "Image",
                    "Colors",
                    "Simpl",
                    "Smooth",
                    "MinW",
                    "MinReg",
                    "Regions",
                    "Fill",
                    "Speck",
                    "Time",
                ]
            )
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                args.tag,
                args.image,
                args.colors,
                args.simplification,
                args.smoothing,
                args.min_width,
                args.min_region_size,
                report.total_regions,
                f"{report.fill_coverage:.4%}",
                f"{report.speck_ratio:.4%}",
                f"{elapsed:.2f}",
            ]
        )

    print(f"Metrics: Regions={report.total_regions}, Fill={report.fill_coverage:.2%}, Speck={report.speck_ratio:.1%}")


if __name__ == "__main__":
    main()
