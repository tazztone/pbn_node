"""
Run: python tests/quality/render_report.py
Produces: tests/quality/output/report.html
"""
import argparse
import base64
import os
import sys
from typing import TypedDict

import cv2
import numpy as np

# Add the parent of the project root and the root itself to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
parent_dir = os.path.dirname(root_dir)
for p in [root_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# --- MOCKING SETUP ---
from tests.mock_comfyui import install_comfyui_mocks  # noqa: E402

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
    {"name": "Boat Vanilla (32 colors)", "img": "boat.webp", "map": None, "colors": 32},
]


def img_to_b64(img_bgr):
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode()


def generate_report(save_svgs=False, llm_reviewer: LLMReviewer = None):
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

        if save_svgs:
            svg_name = run["name"].replace(" ", "_").replace("(", "").replace(")", "").lower() + ".svg"
            svg_path = os.path.join(OUT_DIR, svg_name)
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(result.svg_content)

        # Analyze
        lineart_for_metrics = None
        if edge_map is not None:
            lineart_for_metrics = edge_map.astype(np.float32) / 255.0

        report = analyze(result, img.shape, requested_colors=run["colors"], lineart=lineart_for_metrics)

        # Thumbnails
        thumb_in = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
        thumb_out = cv2.resize(result.quantized, (320, 320), interpolation=cv2.INTER_AREA)

        # Status logic (calibrated to geometric metrics)
        is_good = report.speck_ratio < 0.15 and report.fill_coverage > 0.70 and report.label_coverage > 0.85
        status = "✅" if is_good else "⚠️"

        # LLM Critique
        llm_column = ""
        if llm_reviewer:
            print(f"Requesting LLM review for {run['name']}...")
            _, img_encoded = cv2.imencode(".webp", result.quantized, [cv2.IMWRITE_WEBP_QUALITY, 80])
            critique = llm_reviewer.review_image(img_encoded.tobytes())
            llm_column = f'<td style="max-width: 300px; font-size: 0.85em; color: #ccc;">{critique}</td>'

        rows.append(
            f"""
        <tr>
          <td>
            <div style="font-weight:bold; font-size: 1.2em;">{status} {run['name']}</div>
            <div style="color: #888; margin-top: 5px;">Colors: {run['colors']} | Map: {run['map'] or 'None'}</div>
          </td>
          <td><img src="data:image/png;base64,{img_to_b64(thumb_in)}" title="Input"></td>
          <td><img src="data:image/png;base64,{img_to_b64(thumb_out)}" title="Output"></td>
          <td class="metrics">
            <div class="metric">Regions: <span>{report.total_regions}</span></div>
            <div class="metric">Specks: <span>{report.speck_count} ({report.speck_ratio:.1%})</span></div>
            <div class="metric">Fill: <span>{report.fill_coverage:.1%}</span></div>
            <div class="metric">Labels: <span>{report.label_coverage:.1%}
                ({report.unlabeled_count} skipped)</span></div>
            <div class="metric">Color Efficiency: <span>{report.color_efficiency:.1%}
                ({report.actual_color_count}/{report.requested_color_count})</span></div>
            <div class="metric">Edge Violation: <span>
                {f"{report.edge_violation_ratio:.1%}" if report.edge_violation_ratio
                 is not None else "N/A"}</span></div>
          </td>
          {llm_column}
        </tr>"""
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PBN Quality Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #0f0f12; color: #e0e0e0; margin: 0; padding: 20px; }}
        h1 {{ color: #fff; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; background: #1a1a1e;
                border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }}
        th {{ background: #25252b; color: #aaa; text-align: left; padding: 15px;
                text-transform: uppercase; font-size: 0.8em; letter-spacing: 1px; }}
        td {{ border-bottom: 1px solid #2a2a30; padding: 15px; vertical-align: top; }}
        tr:last-child td {{ border-bottom: none; }}
        img {{ display: block; border-radius: 4px; border: 1px solid #333; }}
        .metrics {{ min-width: 250px; }}
        .metric {{ margin-bottom: 8px; font-size: 0.9em; color: #bbb; }}
        .metric span {{ color: #fff; font-weight: bold; float: right; }}
    </style>
</head>
<body>
    <h1>PBN Quality Report</h1>
    <table>
        <thead>
            <tr>
                <th>Scenario</th>
                <th>Source</th>
                <th>Result</th>
                <th>Metrics</th>
                { '<th>LLM Critique</th>' if llm_reviewer else '' }
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    <div style="margin-top: 30px; font-size: 0.8em; color: #555; text-align: center;">
        Generated by PBN Quality Suite
    </div>
</body>
</html>"""

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
