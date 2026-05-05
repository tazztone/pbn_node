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


def run_iteration(args, img, simplification, smoothing, tag_suffix=""):
    h, w = img.shape[:2]
    tag = f"{args.tag}{tag_suffix}"

    # 1. Process
    processor = ImageProcessor()
    params = ProcessingParameters(
        num_colors=args.colors,
        simplification=simplification,
        smoothing_kernel_size=smoothing,
        min_region_width=args.min_width,
        min_region_size=args.min_region_size,
    )

    print(f"--- Running Iteration: {tag} ---")
    print(f"Settings: Colors={args.colors}, Simpl={simplification}, Smooth={smoothing}, MinW={args.min_width}")

    start_time = time.time()
    result = processor.process_array(img, params)
    elapsed = time.time() - start_time

    # 2. Analyze
    report = analyze(result, img.shape, requested_colors=args.colors)

    # 3. Create Diagnostic Panels
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

    # Panel 3: Gap Map (Using render_coverage logic)
    # Highlight unfilled pixels (black in quantized)
    p3_res = res(result.quantized)
    p3_img = res(img)
    p3_gap_mask = np.all(p3_res == 0, axis=2)
    p3 = (p3_img * 0.2).astype(np.uint8)
    p3[p3_gap_mask] = [0, 0, 255]
    cv2.putText(p3, "GAP MAP (RED)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Panel 4: Boundary Overlay
    boundary_overlay = result.quantized.copy()
    for _rid, poly in result.cleaned_regions.items():
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.polylines(boundary_overlay, [pts], True, (0, 255, 0), 1)
    p4 = res(boundary_overlay)
    cv2.putText(p4, "VECTOR OVERLAY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Assemble Grid
    top = np.hstack((p1, p2))
    bottom = np.hstack((p3, p4))
    grid = np.vstack((top, bottom))

    # Info Bar
    bar_h = 80
    bar = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
    metrics_text = (
        f"Tag: {tag} | Regions: {report.total_regions} | "
        f"Fill(Geo): {report.fill_coverage:.2%} | Render: {report.render_coverage:.2%} | "
        f"Speck: {report.speck_ratio:.1%} | Time: {elapsed:.2f}s"
    )
    settings_text = (
        f"Params: colors={args.colors}, simpl={simplification}, "
        f"smooth={smoothing}, min_w={args.min_width}, "
        f"min_reg={args.min_region_size}"
    )

    cv2.putText(bar, metrics_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(bar, settings_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    final_img = np.vstack((grid, bar))

    # 4. Save and Log
    out_path = os.path.join(OUT_DIR, f"iter_{tag}.png")
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
                    "Fill_Geo",
                    "Render",
                    "Speck",
                    "Time",
                ]
            )
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tag,
                args.image,
                args.colors,
                simplification,
                smoothing,
                args.min_width,
                args.min_region_size,
                report.total_regions,
                f"{report.fill_coverage:.4%}",
                f"{report.render_coverage:.4%}",
                f"{report.speck_ratio:.4%}",
                f"{elapsed:.2f}",
            ]
        )

    return report, p3  # Return report and the gap map panel for sweep grid


def main():
    parser = argparse.ArgumentParser(description="PBN Quick Iteration Tool")
    parser.add_argument("--image", default="boat.webp", help="Image filename in example_inputs/")
    parser.add_argument("--colors", type=int, default=32, help="Number of colors")
    parser.add_argument("--simplification", type=float, default=1.0, help="Simplification tolerance")
    parser.add_argument("--smoothing", type=int, default=9, help="Smoothing kernel size")
    parser.add_argument("--min-width", type=int, default=5, help="Min region width for scanline cleanup")
    parser.add_argument("--min-region-size", type=int, default=None, help="Explicit min region size for segmenter")
    parser.add_argument("--tag", default="latest", help="Tag for the output filename")
    parser.add_argument("--sweep", action="store_true", help="Run a parameter sweep grid")
    args = parser.parse_args()

    # 1. Load Image
    img_path = os.path.join(root_dir, "example_inputs", args.image)
    if not os.path.exists(img_path):
        print(f"ERROR: Image not found at {img_path}")
        sys.exit(1)

    img = cv2.imread(img_path)

    if not args.sweep:
        run_iteration(args, img, args.simplification, args.smoothing)
    else:
        # Parameter Sweep Grid
        smooth_vals = [5, 9, 13, 17, 21]
        simpl_vals = [0.5, 1.0, 1.5, 2.0]

        print(f"Starting SWEEP across {len(smooth_vals)} smooth x {len(simpl_vals)} simpl values...")

        sweep_panels = []
        for simpl in simpl_vals:
            row_panels = []
            for smooth in smooth_vals:
                tag_suffix = f"_s{smooth}_v{simpl}"
                report, gap_panel = run_iteration(args, img, simpl, smooth, tag_suffix)

                # Add label to panel
                label = f"S:{smooth} V:{simpl} | R:{report.total_regions} | F:{report.render_coverage:.1%}"
                cv2.putText(
                    gap_panel, label, (20, gap_panel.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                row_panels.append(gap_panel)
            sweep_panels.append(np.hstack(row_panels))

        sweep_grid = np.vstack(sweep_panels)
        sweep_out = os.path.join(OUT_DIR, f"sweep_{args.tag}.png")
        cv2.imwrite(sweep_out, sweep_grid)
        print(f"\nSWEEP COMPLETE. Grid saved to: {sweep_out}")


if __name__ == "__main__":
    main()
