"""
Preset comparison script to demonstrate the aesthetic difference
between the default/legacy settings and the optimized golden settings.
Saves comparison image to tests/quality/output/comparison.png
"""

import os
import sys

import cv2
import numpy as np

# Add paths for proper imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
parent_dir = os.path.dirname(root_dir)
for p in [root_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Add the tests directory to path for mock import
tests_dir = os.path.join(root_dir, "tests")
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

# --- MOCKING SETUP ---
from mock_comfyui import install_comfyui_mocks  # noqa: E402

install_comfyui_mocks()

from pbn_node.backend.models import ProcessingParameters  # noqa: E402
from pbn_node.pbn_pipeline import ImageProcessor  # noqa: E402

QUALITY_DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.join(QUALITY_DIR, "..", "..", "example_inputs")
OUT_DIR = os.path.join(QUALITY_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    img_path = os.path.join(EXAMPLE_DIR, "boat.webp")
    if not os.path.exists(img_path):
        print(f"Error: Fixture not found at {img_path}")
        sys.exit(1)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"Loaded source image {w}x{h}")

    processor = ImageProcessor()

    # 1. Legacy Default (Posterized / washed-out albedo / rigid shapes)
    print("Processing Legacy Default Preset...")
    legacy_params = ProcessingParameters(
        num_colors=32,
        simplification=1.0,
        use_auto_albedo=True,
        use_bezier_smooth=False,
        smoothing_kernel_size=9,
    )
    legacy_result = processor.process_array(img, legacy_params)

    # 2. Golden Candidate (Vibrant natural colors / detailed polygons / smooth Bezier)
    print("Processing Golden Candidate Preset...")
    golden_params = ProcessingParameters(
        num_colors=32,
        simplification=0.5,
        use_auto_albedo=False,
        use_bezier_smooth=True,
        smoothing_kernel_size=13,
    )
    golden_result = processor.process_array(img, golden_params)

    # Label and Join side-by-side
    def add_label(image, text):
        out = image.copy()
        # Add semi-transparent dark background for text readability
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
        cv2.putText(
            out,
            text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    label_src = add_label(img, "Original Source")
    label_legacy = add_label(legacy_result.quantized, "Legacy (simpl=1.0, Albedo=On, Bezier=Off)")
    label_golden = add_label(golden_result.quantized, "Golden (simpl=0.5, Albedo=Off, Bezier=On)")

    # Concatenate horizontally
    comparison = np.hstack([label_src, label_legacy, label_golden])

    out_path = os.path.join(OUT_DIR, "comparison.png")
    cv2.imwrite(out_path, comparison)
    print(f"\nSUCCESS: Preset comparison saved to {out_path}!")


if __name__ == "__main__":
    main()
