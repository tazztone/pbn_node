import base64

import cv2
import numpy as np


def img_to_b64(img_bgr) -> str:
    """
    Encode a BGR OpenCV image to a base64 PNG string.
    """
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")


def build_diagnostic_grid(
    img: np.ndarray,
    result,
    report,
    elapsed: float,
    colors: int,
    simplification: float,
    smoothing: int,
    min_width: int,
    min_region_size: int | None,
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble a 4-panel diagnostic grid (Source, Result, Gap Map, Vector Overlay)
    along with an information bar summarizing key metrics and parameters.

    Returns:
        (final_grid_image, gap_map_panel)
    """
    h, w = img.shape[:2]
    max_w = 1200
    scale = min(1.0, max_w / w)
    dw, dh = int(w * scale), int(h * scale)

    def res(image):
        return cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA)

    # Panel 1: Original Source Image
    p1 = res(img)
    cv2.putText(p1, "SOURCE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Panel 2: Quantized PBN Result
    p2 = res(result.quantized)
    cv2.putText(p2, "RESULT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Panel 3: Gap Map (highlights non-covered pixels in red)
    p3_res = res(result.quantized)
    p3_img = res(img)
    p3_gap_mask = np.all(p3_res == 0, axis=2)
    p3 = (p3_img * 0.2).astype(np.uint8)
    p3[p3_gap_mask] = [0, 0, 255]
    cv2.putText(p3, "GAP MAP (RED)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Panel 4: Vector Boundary Overlay
    boundary_overlay = result.quantized.copy()
    for _rid, poly in result.cleaned_regions.items():
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.polylines(boundary_overlay, [pts], True, (0, 255, 0), 1)
    p4 = res(boundary_overlay)
    cv2.putText(p4, "VECTOR OVERLAY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Assemble panels into 2x2 grid
    top = np.hstack((p1, p2))
    bottom = np.hstack((p3, p4))
    grid = np.vstack((top, bottom))

    # Add a styled info bar at the bottom
    bar_h = 80
    bar = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)

    metrics_text = (
        f"Tag: {tag} | Regions: {report.total_regions} | "
        f"Fill(Geo): {report.fill_coverage:.2%} | Render: {report.render_coverage:.2%} | "
        f"Speck: {report.speck_ratio:.1%} | Time: {elapsed:.2f}s"
    )
    settings_text = (
        f"Params: colors={colors}, simpl={simplification}, "
        f"smooth={smoothing}, min_w={min_width}, "
        f"min_reg={min_region_size}"
    )

    cv2.putText(bar, metrics_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(bar, settings_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    final_img = np.vstack((grid, bar))
    return final_img, p3
