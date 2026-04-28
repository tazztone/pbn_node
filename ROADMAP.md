# Project roadmap

This document outlines the planned improvements and research directions for the
ComfyUI paint-by-number node. These features are ranked by their potential
impact on output quality and their ease of implementation.

## Overview

We have categorized the following candidates into tiers based on their ROI
(Return on Investment). Our goal is to move from a color-first pipeline to a
perception-aware and content-aware system.

| # | Feature | Effectiveness | Impl. Difficulty | Status | ROI |
|---|---|---|---|---|---|
| 1 | **CIEDE2000** | 🟢 High | 🟢 Trivial | ✅ Done | ★★★★★ |
| 2 | **Shared Border SVG wiring** | 🟢 High | 🟢 Low | ✅ Done | ★★★★★ |
| 3 | **SLIC Superpixels** | 🟡 Medium-high | 🟢 Low | ✅ Done | ★★★★☆ |
| 4 | **Content-Aware Preprocessing** | 🟢 High (portraits) | 🟡 Moderate | ✅ Done | ★★★★☆ |
| 5 | **SVG Vector Preview** | 🟡 Medium | 🟢 Low | ✅ Done | ★★★★☆ |
| 6 | **Region numbering by color** | 🔴 Critical bug | 🟢 Low | 🔧 Fix needed | ★★★★★ |
| 7 | **Pairwise palette merge** | 🟢 High | 🟢 Low | 🔧 Fix needed | ★★★★★ |
| 8 | **Flood fill O(n²) fix** | 🟢 Performance | 🟢 Trivial | 🔧 Fix needed | ★★★★★ |
| 9 | **Vectorized adjacency scanning** | 🟡 Performance | 🟢 Low | 🔧 Fix needed | ★★★★☆ |
| 10 | **Palette chart output** | 🟡 UX | 🟢 Low | 📅 Planned | ★★★★☆ |
| 11 | **Palette merge defaults fix** | 🟡 UX | 🟢 Trivial | 📅 Planned | ★★★★☆ |
| 12 | **Lineart exclusion mask** | 🟡 Medium | 🟢 Low | 📅 Planned | ★★★☆☆ |
| 13 | **numbers_density parameter** | 🟡 Medium | 🟡 Moderate | 📅 Planned | ★★★☆☆ |
| 14 | **Semantic mask input** | 🟢 High (portraits) | 🟡 Moderate | 📅 Planned | ★★★★☆ |
| 15 | **Depth Anything 3** | 🟢 High (landscapes) | 🟠 Moderate-hard | 🟡 Partial | ★★★☆☆ |
| 16 | **Interactive Palette Controls** | 🟡 UX only | 🟠 Hard (frontend) | 📅 Planned | ★★★☆☆ |
| 17 | **SAM 3.1** | 🟢 Transformative | 🔴 Hard (arch. change) | 🧪 Research | ★★☆☆☆ |
| 18 | **Learned Edge Detection** | 🟡 Incremental | 🔴 Hard (model tuning) | 🧪 Research | ★★☆☆☆ |

---

## ✅ Completed Features

These features have been implemented and are available in the current version.

### CIEDE2000 color distance
Replaced Euclidean distance in LAB space with the `skimage.color.deltaE_ciede2000`
formula. This provides perceptually correct palette selection, ensuring paint
colors map more naturally to human perception.

### Shared border segmentation
The SVG generator now uses shared borders instead of independent contours for
each region. This eliminates the "white-gap" problem and produces perfectly
aligned vector paths.

### SLIC superpixels
Integrated `skimage.segmentation.slic()` into the pipeline to reduce pixel space
to compact cells before clustering. This dramatically reduces speckle and
isolates the quantizer from noise.

### Content-aware protection (Face detection)
Implemented Mediapipe-based face detection to generate protection maps. This
ensures high-frequency details in portraits are preserved during color
quantization.

### Budget Split (Foreground/Background)
Implemented a color budget splitting mechanism using Otsu's thresholding to
separate foreground and background. This allows allocating a larger portion of
the color palette to the subject of the image.

### Polylabel placement
Implemented the Polylabel algorithm for optimal label positioning at the "pole
of inaccessibility" (the most distant internal point from the polygon outline).

### SVG Vector Preview
Implemented a custom JavaScript extension for the ComfyUI frontend that renders
the generated SVG as a vector graphic directly in the node body. This provides
sharp, zoomable previews and supports batch execution through a scrollable DOM
widget.

---

## 🔧 Active Bug Fixes & Optimizations

These were identified in a pipeline audit and must be addressed before further
feature work.

### Bug: Region numbering by color identity (Critical)

**Problem:** `_renumber_regions()` in `pbn_pipeline.py` assigns a unique
sequential ID to every disconnected polygon patch. Two islands of the same
color get different numbers (e.g. 47 and 83), producing region counts in the
hundreds even with only 15 palette colors. In paint-by-number, all patches of
the same color must share the same number.

**Fix:** Store a `region_colors: dict[int, int]` mapping (region ID → palette
color index) in `RegionData`. Propagate this through vectorization and
renumbering. In `SVGGenerator.group_paths_by_color()`, replace the blind
modulo `(region_id - 1) % len(colors.hex_colors)` with a lookup into
`region_colors`. Labels in the SVG become `color_index + 1`.

**Files:** `backend/models.py`, `backend/segmentation/segmenter.py`,
`backend/vectorization/vectorizer.py`, `pbn_pipeline.py`,
`backend/svg_generation/svg_generator.py`

---

### Bug: Greedy sequential palette merge produces suboptimal results

**Problem:** The merge pass in `quantizer.py` iterates colors in KMeans output
order and compares each against a running group mean. Colors that are
perceptually close but non-adjacent in that order are never merged, leaving
near-identical palette entries even at high thresholds.

**Fix:** Replace with a proper pairwise merge: compute all pairwise CIEDE2000
distances upfront, find the closest pair, merge them (average their LAB
values), repeat until no pair is below the threshold. Add iterative budget
enforcement — if the merged palette still exceeds `num_colors`, raise the
threshold by 1.0 and retry until the budget is met. Lower `k_cap` from 30 to
20.

**File:** `backend/quantization/quantizer.py`

---

### Performance: Flood fill O(n²) due to `list.pop(0)`

**Problem:** `_get_region_pbnify()` in `segmenter.py` uses a plain Python list
as a queue and calls `pop(0)` — an O(n) operation — on every iteration, making
the flood fill O(n²) overall. On a 1024×683 image this is the dominant
performance bottleneck. Additionally, `_get_region_pbnify()` performs a full
`covered.copy()` on every call that is immediately discarded.

**Fix:** Replace the list queue with `collections.deque` and use `popleft()`.
Remove the redundant `covered.copy()`.

**File:** `backend/segmentation/segmenter.py`

---

### Performance: Vectorize adjacency and border scanning

**Problem:** `build_adjacency_graph()` and `shared_border_segmentation()` both
use nested Python `for` loops to scan every pixel, which is extremely slow on
large images.

**Fix:** Replace with NumPy shifted-array comparisons:

```python
# Horizontal borders
h_adj = (regions[:, :-1] != regions[:, 1:]) & (regions[:, :-1] > 0) & (regions[:, 1:] > 0)
ys, xs = np.where(h_adj)
pairs = np.stack([regions[ys, xs], regions[ys, xs + 1]], axis=1)
```

Apply the same pattern vertically. Extract unique pairs for the adjacency
graph and coordinate lists for shared border LineStrings.

**File:** `backend/segmentation/segmenter.py`

---

## 🚀 Planned Features

### Palette chart output

Add a second `IMAGE` output (`PALETTE_CHART`) to the node. A new
`backend/palette_chart.py` module generates a fixed-width raster image showing
one row per color: a filled swatch, the number, hex code, and RGB values. No
new dependencies — implemented with OpenCV `cv2.rectangle` and `cv2.putText`.

This gives painters a physical reference sheet to use alongside the printed
template.

- **Status:** Planned.
- **Impact:** High practical value for end users. Low effort.

---

### Palette merge defaults fix

`use_palette_merge` and `use_ciede2000` are already `True` in the `balanced`
preset, but `ciede2000_merge_thresh` is not explicitly set by the preset and
inherits the raw default of `8.0`, which is too low for natural photos.

Set the `balanced` preset to use `ciede2000_merge_thresh = 10.0` explicitly.
Rename the exposed parameter from `ciede2000_merge_thresh` to
`color_merge_threshold` in the node schema for friendlier UX.

- **Status:** Planned.
- **Impact:** Improves out-of-the-box results with zero architectural change.

---

### Lineart exclusion mask for label placement

Accept an optional `lineart_image` input. Dilate it with `cv2.dilate()` and
use the result as an exclusion zone inside `LabelPlacer` — candidate label
positions that fall within the dilated mask are skipped. This prevents numbers
from being placed directly on stroke lines, improving readability on
high-contrast outlines.

- **Status:** Planned.
- **Impact:** Meaningful improvement for users who supply a lineart pass
  (e.g. from Canny, HED, or Sapiens2).

---

### `numbers_density` parameter

Add a `numbers_density: float` parameter (default `0.0` = one label per
region). When set, compute `n = max(1, int(polygon.area * numbers_density))`
labels per region, placed by tiling polylabel calls on eroded sub-regions with
a minimum spacing of ~30px. `LabelData.positions` changes from
`dict[int, Point]` to `dict[int, list[Point]]`. Update SVG generation
accordingly.

Large regions on a physical print often require multiple number placements for
the painter to follow without confusion.

- **Status:** Planned.
- **Impact:** Medium. Requires `LabelData` interface change.

---

### Semantic mask input

Add an optional `semantic_mask: IMAGE` input. If provided, the mask is
converted to a boundary weight map and fed into the existing `protection_map`
mechanism in `pbn_pipeline.py` to bias KMeans cluster placement toward
semantic boundaries rather than raw color gradients.

This is intentionally generic — the mask can come from any upstream node:
Sapiens2 body segmentation, SAM, a manual ComfyUI mask, or the existing
Mediapipe face detector. The node does not bundle any segmentation model.

A new `portrait` preset will use this input with a per-segment budget split:
run KMeans separately on each semantic region (e.g. skin, hair, clothing,
background) with proportional color budgets, then merge the palettes before
the shared quantization step.

- **Status:** Planned.
- **Impact:** High for portraits. Fixes the core problem of skin tones and
  dark clothing merging into the same color clusters.

---

### Depth Anything 3 (DA3)
Replacing the current Otsu-based budget split with DA3 to produce more accurate
foreground/background masks and splitting the color budget proportionally.

- **Status:** Researching integration.
- **Impact:** Excellent for landscapes and complex depth scenes.

---

### Interactive palette controls
Building ComfyUI V3 widgets to let you manually merge or split colors. This
requires deep integration with the ComfyUI frontend API.

- **Status:** Long-term goal.
- **Impact:** Improved user experience and manual control.

---

### SAM 3.1 (Segment Anything)
Replacing the color-first segmentation with SAM's auto-mask generator to produce
semantic regions (for example, sky, skin, or objects).

- **Status:** Researching.
- **Impact:** Potentially transformative for complex scenes.

---

### Learned edge detection
Replacing morphological smoothing with models like DexiNed or HED to produce
crisper, more artistically natural boundaries.

- **Status:** Researching.
- **Impact:** Cleanest SVG contours possible.

---

## Recommended implementation order

1. **Region numbering bug fix** — highest impact, unblocks correct output
2. **Pairwise palette merge fix** — unblocks correct color counts
3. **Flood fill + vectorized scanning** — performance, easy wins
4. **Palette merge defaults + palette chart** — UX polish, low effort
5. **Semantic mask input** — portrait quality, medium effort
6. **Lineart exclusion + numbers_density** — labeling polish
7. **Depth Anything 3** — landscape quality
8. **Interactive palette controls** — long-term UX
9. **SAM 3.1 / Learned edge detection** — research track
