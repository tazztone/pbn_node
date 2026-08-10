import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from string import Template
from typing import TypedDict

# Initialize paths and ComfyUI mocks explicitly
import bootstrap

bootstrap.setup()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from bootstrap import EXAMPLE_DIR, OUT_DIR  # noqa: E402
from llm_review import LLMReviewer  # noqa: E402
from metrics import QualityReport, analyze  # noqa: E402
from pbn_node.backend.models import PerceptionInputs, ProcessingParameters  # noqa: E402
from pbn_node.pbn_pipeline import ImageProcessor  # noqa: E402
from visuals import img_to_b64  # noqa: E402


class RunConfig(TypedDict):
    name: str
    img: str
    map: str | None
    colors: int


DEFAULT_RUNS: list[RunConfig] = [
    {"name": "Boat Vanilla (12 colors)", "img": "boat.webp", "map": None, "colors": 12},
    {"name": "Boat Lineart (12 colors)", "img": "boat.webp", "map": "boat_lineart.webp", "colors": 12},
    {"name": "Boat Canny (16 colors)", "img": "boat.webp", "map": "boat_cannyedge.webp", "colors": 16},
    {"name": "Boat HED (16 colors)", "img": "boat.webp", "map": "boat_HED.webp", "colors": 16},
    {"name": "Boat Vanilla (24 colors)", "img": "boat.webp", "map": None, "colors": 24},
    {"name": "Boat Vanilla (32 colors)", "img": "boat.webp", "map": None, "colors": 32},
]


def load_runs_config(config_path: str | None) -> list[RunConfig]:
    """
    Load runs configuration from a JSON file, falling back to default runs.
    """
    if config_path:
        path = Path(config_path)
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config JSON from {path}: {e}. Falling back to default runs.")
        else:
            print(f"Config path {path} not found. Falling back to default runs.")
    return DEFAULT_RUNS


def process_scenario(
    run: RunConfig, processor: ImageProcessor, save_svgs: bool
) -> tuple[np.ndarray, np.ndarray, QualityReport]:
    """
    Process a single scenario run, write outputs/SVGs, and perform quality analysis.
    """
    img_path = EXAMPLE_DIR / run["img"]
    if not img_path.exists():
        raise FileNotFoundError(f"Fixture not found: {img_path}")

    img = cv2.imread(str(img_path))

    edge_map = None
    perception = None
    if run["map"]:
        map_path = EXAMPLE_DIR / run["map"]
        if map_path.exists():
            edge_map_raw = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
            edge_map = cv2.resize(edge_map_raw, (img.shape[1], img.shape[0]))
            edge_map_float = edge_map.astype(np.float32) / 255.0
            perception = PerceptionInputs(lineart=edge_map_float, lineart_strength=0.8, edge_influence=0.5)

    params = ProcessingParameters(num_colors=run["colors"], perception=perception)
    result = processor.process_array(img, params)

    # Save images
    clean_name = run["name"].replace(" ", "_").replace("(", "").replace(")", "").lower()
    cv2.imwrite(str(OUT_DIR / f"{clean_name}_input.png"), img)
    cv2.imwrite(str(OUT_DIR / f"{clean_name}_result.png"), result.quantized)

    if save_svgs:
        svg_path = OUT_DIR / f"{clean_name}.svg"
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(result.svg_content)

    lineart_for_metrics = None
    if edge_map is not None:
        lineart_for_metrics = edge_map.astype(np.float32) / 255.0

    report = analyze(result, img.shape, requested_colors=run["colors"], lineart=lineart_for_metrics)
    return img, result.quantized, report


def resize_maintain_ar(image: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio, capping max dimension.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def build_html_row(
    run: RunConfig, img: np.ndarray, result_img: np.ndarray, report: QualityReport, llm_reviewer: LLMReviewer | None
) -> str:
    """
    Construct the HTML table row representing a single scenario with its metrics and visual previews.
    """
    thumb_in = resize_maintain_ar(img)
    thumb_out = resize_maintain_ar(result_img)

    delta = report.render_coverage - report.fill_coverage
    delta_class = "danger" if delta > 0.10 else ("warn" if delta > 0.05 else "")

    is_warning = (
        report.speck_ratio > 0.1
        or report.fill_coverage < 0.7
        or delta > 0.10
        or (report.edge_violation_ratio is not None and report.edge_violation_ratio > 0.42)
    )

    status_cls = "status-warn" if is_warning else "status-ok"
    status_label = "⚠ ATTENTION" if is_warning else "✓ OPTIMAL"

    speck_warn_cls = "warn" if report.speck_ratio > 0.05 else ""
    fill_warn_cls = "warn" if report.fill_coverage < 0.85 else ""
    render_warn_cls = "warn" if report.render_coverage < 0.90 else ""
    edge_warn_cls = "warn" if (report.edge_violation_ratio or 0) > 0.42 else ""

    edge_fid_text = (
        f"{100 - report.edge_violation_ratio * 100:.1f}%" if report.edge_violation_ratio is not None else "N/A"
    )

    llm_column = ""
    if llm_reviewer:
        print(f"Requesting LLM review for {run['name']}...")
        _, img_encoded = cv2.imencode(".jpg", result_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        critique = llm_reviewer.review_image(img_encoded.tobytes(), mime_type="image/jpeg") or "N/A (Review Failed)"
        llm_column = f'<td class="critique-cell"><div class="critique-box">{critique}</div></td>'

    return f"""
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
            <div class="metric-card {speck_warn_cls}">
                <label>Speck Ratio</label>
                <value>{report.speck_ratio:.1%}</value>
            </div>
            <div class="metric-card {fill_warn_cls}">
                <label>Fill</label>
                <value>{report.fill_coverage:.1%}</value>
            </div>
            <div class="metric-card {render_warn_cls}">
                <label>Render</label>
                <value>{report.render_coverage:.1%}</value>
            </div>
            <div class="metric-card {delta_class}">
                <label>Delta</label>
                <value>+{delta:.1%}</value>
            </div>
            <div class="metric-card {edge_warn_cls}">
                <label>Edge Fid.</label>
                <value>{edge_fid_text}</value>
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


def generate_report(runs: list[RunConfig], save_svgs: bool, llm_reviewer: LLMReviewer | None = None):
    """
    Main runner to process runs config and generate an HTML audit report.
    """
    rows = []
    processor = ImageProcessor()

    print(f"Generating quality report to {OUT_DIR}...")

    for run in runs:
        try:
            print(f"Processing {run['name']}...")
            img, result_img, report = process_scenario(run, processor, save_svgs)
            row_html = build_html_row(run, img, result_img, report, llm_reviewer)
            rows.append(row_html)
        except Exception as e:
            print(f"SKIP {run['name']}: {e}")

    # Load HTML template
    template_path = Path(__file__).parent / "report_template.html"
    if template_path.exists():
        with open(template_path, encoding="utf-8") as f:
            template = f.read()
    else:
        template = "<html><body><h1>Report Template Missing</h1>$rows</body></html>"

    html = Template(template).safe_substitute(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        llm_header="<th>AI Visual Critique</th>" if llm_reviewer else "",
        rows="".join(rows),
    )

    report_path = OUT_DIR / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nSUCCESS: Report written to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to optional JSON configuration file for runs")
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
            raise ValueError("ERROR: --api-key or OPENROUTER_API_KEY env var required for --llm-review")
        reviewer = LLMReviewer(api_key=api_key, api_base=args.api_base, model=args.model)

    runs_config = load_runs_config(args.config)
    generate_report(runs_config, save_svgs=args.save_svgs, llm_reviewer=reviewer)
