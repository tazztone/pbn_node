import argparse
import csv
import sys
import time
from datetime import datetime

# Initialize paths and ComfyUI mocks explicitly
import bootstrap
import cv2
import numpy as np

bootstrap.setup()
import metrics  # noqa: E402
import visuals  # noqa: E402
from bootstrap import EXAMPLE_DIR, OUT_DIR  # noqa: E402

from pbn_node.backend.models import ProcessingParameters  # noqa: E402
from pbn_node.pbn_pipeline import ImageProcessor  # noqa: E402

LOG_FILE = OUT_DIR / "iteration_log.csv"


def run_iteration(args, img, simplification, smoothing, tag_suffix=""):
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
    report = metrics.analyze(result, img.shape, requested_colors=args.colors)

    # 3. Create Diagnostic Panels via visuals module
    final_img, p3 = visuals.build_diagnostic_grid(
        img=img,
        result=result,
        report=report,
        elapsed=elapsed,
        params=params,
        tag=tag,
    )

    # 4. Save Image
    out_path = OUT_DIR / f"iter_{tag}.png"
    cv2.imwrite(str(out_path), final_img)
    print(f"SAVED: {out_path}")

    # 5. Log to CSV with Schema Verification
    expected_header = [
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

    if LOG_FILE.exists():
        try:
            with open(LOG_FILE) as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
            if existing_header != expected_header:
                backup_path = LOG_FILE.with_name(f"iteration_log_backup_{int(time.time())}.csv")
                LOG_FILE.rename(backup_path)
                print(f"WARNING: CSV schema mismatch detected. Archived old log to {backup_path}")
        except Exception as e:
            print(f"WARNING: Could not validate CSV schema ({e}). Re-creating log file.")
            try:
                LOG_FILE.unlink()
            except Exception:
                pass

    log_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(expected_header)
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

    return report, p3


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

    img_path = EXAMPLE_DIR / args.image
    if not img_path.exists():
        print(f"ERROR: Image not found at {img_path}")
        sys.exit(1)

    img = cv2.imread(str(img_path))

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
        sweep_out = OUT_DIR / f"sweep_{args.tag}.png"
        cv2.imwrite(str(sweep_out), sweep_grid)
        print(f"\nSWEEP COMPLETE. Grid saved to: {sweep_out}")


if __name__ == "__main__":
    main()
