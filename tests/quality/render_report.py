"""
Run: python tests/quality/render_report.py
Produces: tests/quality/output/report.html
"""

import argparse
import base64
import os
import sys
from datetime import datetime
from string import Template
from typing import TypedDict

import cv2
import numpy as np

# Add the parent of the project root and the root itself to path
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

from llm_review import LLMReviewer  # noqa: E402
from metrics import analyze  # noqa: E402

from pbn_node.backend.models import PerceptionInputs, ProcessingParameters  # noqa: E402
from pbn_node.pbn_pipeline import ImageProcessor  # noqa: E402

QUALITY_DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.join(QUALITY_DIR, "..", "..", "example_inputs")
OUT_DIR = os.path.join(QUALITY_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


class RunConfig(TypedDict):
    name: str
    img: str
    map: str | None
    colors: int


RUNS: list[RunConfig] = [
    {"name": "Boat Vanilla (12 colors)", "img": "boat.webp", "map": None, "colors": 12},
    {"name": "Boat Lineart (12 colors)", "img": "boat.webp", "map": "boat_lineart.webp", "colors": 12},
    {"name": "Boat Canny (16 colors)", "img": "boat.webp", "map": "boat_cannyedge.webp", "colors": 16},
    {"name": "Boat HED (16 colors)", "img": "boat.webp", "map": "boat_HED.webp", "colors": 16},
    {"name": "Boat Vanilla (24 colors)", "img": "boat.webp", "map": None, "colors": 24},
    {"name": "Boat Vanilla (32 colors)", "img": "boat.webp", "map": None, "colors": 32},
]


def img_to_b64(img_bgr):
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode()


def generate_report(save_svgs=False, llm_reviewer: LLMReviewer | None = None):
    rows = []
    processor = ImageProcessor()

    print(f"Generating quality report to {OUT_DIR}...")

    for run in RUNS:
        img_path = os.path.join(EXAMPLE_DIR, run["img"])
        if not os.path.exists(img_path):
            print(f"SKIP: {img_path}")
            continue

        img = cv2.imread(img_path)

        edge_map = None
        perception = None
        if run["map"]:
            map_path = os.path.join(EXAMPLE_DIR, run["map"])
            if os.path.exists(map_path):
                edge_map_raw = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
                edge_map = cv2.resize(edge_map_raw, (img.shape[1], img.shape[0]))
                edge_map_float = edge_map.astype(np.float32) / 255.0
                perception = PerceptionInputs(lineart=edge_map_float, lineart_strength=0.8, edge_influence=0.5)

        params = ProcessingParameters(num_colors=run["colors"], perception=perception)
        print(f"Processing {run['name']}...")
        result = processor.process_array(img, params)

        # Save Artifacts
        clean_name = run["name"].replace(" ", "_").replace("(", "").replace(")", "").lower()
        cv2.imwrite(os.path.join(OUT_DIR, f"{clean_name}_input.png"), img)
        cv2.imwrite(os.path.join(OUT_DIR, f"{clean_name}_result.png"), result.quantized)

        if save_svgs:
            svg_path = os.path.join(OUT_DIR, f"{clean_name}.svg")
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(result.svg_content)

        # Analyze
        lineart_for_metrics = None
        if edge_map is not None:
            lineart_for_metrics = edge_map.astype(np.float32) / 255.0

        report = analyze(result, img.shape, requested_colors=run["colors"], lineart=lineart_for_metrics)

        # Thumbnails (Larger for detailed inspection)
        def resize_maintain_ar(image, max_dim=1024):
            h, w = image.shape[:2]
            if max(h, w) <= max_dim:
                return image
            scale = max_dim / max(h, w)
            return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        thumb_in = resize_maintain_ar(img)
        thumb_out = resize_maintain_ar(result.quantized)

        # Status logic
        is_warning = (
            report.speck_ratio > 0.1
            or report.fill_coverage < 0.7
            or (report.edge_violation_ratio is not None and report.edge_violation_ratio > 0.42)
        )
        status_cls = "status-warn" if is_warning else "status-ok"
        status_label = "⚠ ATTENTION" if is_warning else "✓ OPTIMAL"

        llm_column = ""
        if llm_reviewer:
            print(f"Requesting LLM review for {run['name']}...")
            _, img_encoded = cv2.imencode(".jpg", result.quantized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            critique = llm_reviewer.review_image(img_encoded.tobytes(), mime_type="image/jpeg") or "N/A (Review Failed)"
            llm_column = f'<td class="critique-cell"><div class="critique-box">{critique}</div></td>'

        rows.append(
            f"""
        <tr class="{status_cls}">
          <td class="scenario-cell">
            <div class="scenario-header">
                <span class="status-badge">{status_label}</span>
                <div class="scenario-name">{run["name"]}</div>
            </div>
            <div class="scenario-meta">Colors: {run["colors"]} | Map: {run["map"] or "None"}</div>

            <div class="metrics-grid" style="margin-top: 20px;">
                <div class="metric-card">
                    <label>Regions</label>
                    <value>{report.total_regions}</value>
                </div>
                <div class="metric-card {"warn" if report.speck_ratio > 0.05 else ""}">
                    <label>Speck Ratio</label>
                    <value>{report.speck_ratio:.1%}</value>
                </div>
                <div class="metric-card {"warn" if report.fill_coverage < 0.85 else ""}">
                    <label>Fill</label>
                    <value>{report.fill_coverage:.1%}</value>
                </div>
                <div class="metric-card {"warn" if (report.edge_violation_ratio or 0) > 0.42 else ""}">
                    <label>Edge Fid.</label>
                    <value>{
                f"{100 - report.edge_violation_ratio * 100:.1f}%" if report.edge_violation_ratio is not None else "N/A"
            }</value>
                </div>
            </div>
          </td>
          <td colspan="2">
            <div class="comparison-grid">
                <div class="img-container">
                    <label>SOURCE INPUT</label>
                    <img src="data:image/png;base64,{img_to_b64(thumb_in)}"
                         onclick="this.classList.toggle('zoom')"
                         title="Click to zoom">
                </div>
                <div class="img-container">
                    <label>QUANTIZED RESULT</label>
                    <img src="data:image/png;base64,{img_to_b64(thumb_out)}"
                         onclick="this.classList.toggle('zoom')"
                         title="Click to zoom">
                </div>
            </div>
          </td>
          {llm_column}
        </tr>"""
        )

    # Load template
    template_path = os.path.join(os.path.dirname(__file__), "report_template.html")
    if os.path.exists(template_path):
        with open(template_path, encoding="utf-8") as f:
            template = f.read()
    else:
        template = "<html><body><h1>Report Template Missing</h1>{{rows}}</body></html>"

    html = Template(template).safe_substitute(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        llm_header="<th>AI Visual Critique</th>" if llm_reviewer else "",
        rows="".join(rows),
    )

    report_path = os.path.join(OUT_DIR, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nSUCCESS: Report written to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-svgs", action="store_true", help="Save raw SVG files to output directory")
    parser.add_argument("--llm-review", action="store_true", help="Enable LLM-based visual critique")
    parser.add_argument("--api-key", help="API Key for LLM service (OpenRouter or OpenAI)")
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1", help="Base URL for API")
    parser.add_argument("--model", default="google/gemini-2.0-flash-001", help="Model to use for review")
    args = parser.parse_args()

    reviewer = None
    if args.llm_review:
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: --api-key or OPENROUTER_API_KEY env var required for --llm-review")
            sys.exit(1)
        reviewer = LLMReviewer(api_key=api_key, api_base=args.api_base, model=args.model)

    generate_report(save_svgs=args.save_svgs, llm_reviewer=reviewer)
