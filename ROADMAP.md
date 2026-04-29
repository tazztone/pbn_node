# Project roadmap

The project roadmap outlines the planned improvements and research directions for
the ComfyUI paint-by-number node. These features are ranked by their potential
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
| 6 | **Region numbering by color** | 🔴 Critical bug | 🟢 Low | ✅ Done | ★★★★★ |
| 7 | **Pairwise palette merge** | 🟢 High | 🟢 Low | ✅ Done | ★★★★★ |
| 8 | **Flood fill O(n²) fix** | 🟢 Performance | 🟢 Trivial | ✅ Done | ★★★★★ |
| 9 | **Vectorized adjacency scanning** | 🟡 Performance | 🟢 Low | ✅ Done | ★★★★☆ |
| 10 | **Palette chart output** | 🟡 UX | 🟢 Low | 📅 Planned | ★★★★☆ |
| 11 | **Palette merge defaults fix** | 🟡 UX | 🟢 Trivial | ✅ Done | ★★★★☆ |
| 12 | **Lineart exclusion mask** | 🟡 Medium | 🟢 Low | ✅ Done | ★★★☆☆ |
| 13 | **numbers_density parameter** | 🟡 Medium | 🟡 Moderate | 📅 Planned | ★★★☆☆ |
| 14 | **Content-Aware Budget Split** | 🟢 High | 🟡 Moderate | ✅ Done | ★★★★☆ |
| 15 | **Perceptual Palette Sorting** | 🟡 UX | 🟢 Low | 📅 Planned | ★★★☆☆ |
| 16 | **Label Collision Avoidance** | 🟡 UX | 🟡 Moderate | 📅 Planned | ★★★☆☆ |
| 17 | **Exposed Clean-up Controls** | 🟢 Quality | 🟢 Trivial | 📅 Planned | ★★★★☆ |
| 18 | **Interactive Palette Controls**| 🟡 UX only | 🟠 Hard (frontend)| 📅 Planned | ★★★☆☆ |
| 19 | **Sapiens / Normal SLIC** | 🟢 Transformative | 🟡 Moderate | ✅ Done | ★★★★☆ |
| 20 | **Modular Backend Refactor** | 🟢 Maintainability | 🟡 Moderate | ✅ Done | ★★★★★ |

---

## ✅ Completed features

These features have been implemented and are available in the current version.

### CIEDE2000 color distance
Replaced Euclidean distance in LAB space with the
`skimage.color.deltaE_ciede2000` formula. This provides perceptually correct
palette selection, ensuring paint colors map more naturally to human perception.

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
sharp, zoomable previews and supports batch execution.

### Region numbering by color identity
Modified the pipeline, SVG generator, and raster renderer to use color-based
indexing for labels. This ensures all disconnected islands of the same color
share the same number and fill color.

### Pairwise palette merge
Replaced the greedy sequential merge with a robust pairwise CIEDE2000 distance
merger. This ensures the palette is reduced to the target size while
prioritizing the most perceptually similar colors.

### Flood fill O(n) optimization
Replaced O(n²) flood fill logic with a deque-based O(n) implementation,
eliminating performance bottlenecks on high-resolution images.

### Vectorized adjacency and border scanning
Rewrote the adjacency graph building and shared border detection logic using
NumPy vectorized operations, significantly reducing processing time.

### Perception Stack v2
Introduced semantic awareness to the region segmentation and labeling pipeline.
The system uses lineart maps to guide the smoothing filter, implements automated
shadow-removal using Retinex, and optimizes label placement.

### Content-Aware Budget Splitting
Implemented a multi-segment quantization pipeline that uses an external
segmentation mask (or an automatic Otsu-based fallback) to allocate independent
color budgets to different image regions.

### Normal-Map-Guided SLIC
Integrated 3D geometry into the segmentation pipeline. Superpixel boundaries
now respect physical surface creases even in low-contrast areas by leveraging
angular gradients and curvature from normal maps.

### Sapiens Body-Part Adaptive Priority
Added support for Sapiens segmentation masks. The system automatically
identifies anatomical body parts and applies adaptive priority weights to the
color quantization and protection maps.

### Modular backend refactor
Executed a comprehensive architectural overhaul to improve maintainability and
performance. We broke down monolithic node classes into modular components,
implemented lazy-loading for heavy dependencies, and expanded the automated
test suite to 35 comprehensive unit and integration tests.

---

## 🚀 Planned features

### Palette chart output

Add a second `IMAGE` output (`PALETTE_CHART`) to the node. A new
`backend/palette_chart.py` module generates a fixed-width raster image showing
one row per color: a filled swatch, the number, hex code, and RGB values.

- **Status:** Planned.
- **Impact:** High practical value for end users. Low effort.

---

### `numbers_density` parameter

Add a `numbers_density: float` parameter. When set, compute multiple labels per
region placed by tiling polylabel calls on eroded sub-regions.

- **Status:** Planned.
- **Impact:** Medium. Requires `LabelData` interface change.

---

### Perceptual palette sorting

Sort the generated color palette and `PALETTE_CHART` by luminance or hue. This
makes the physical paint selection process much more intuitive for the end-user.

- **Status:** Planned.
- **Impact:** High UX value.

---

### Label collision avoidance

Implement a collision detection pass in `LabelPlacer`. If two labels are too
close, nudge them along their respective region's skeleton or skip the less
important one.

- **Status:** Planned.
- **Impact:** Medium labeling polish.

---

### Exposed clean-up controls

Expose `speckle_threshold` and `min_region_width` as node parameters. Currently
partially hardcoded, these controls are vital for users processing extremely
noisy or flat input images.

- **Status:** Planned.
- **Impact:** High for power users.

---

### Interactive palette controls

Building ComfyUI V3 widgets to let you manually merge or split colors. This
requires deep integration with the ComfyUI frontend API.

- **Status:** Long-term goal.
- **Impact:** Improved user experience and manual control.

---

### SAM 3.1 / DA3 / Sapiens2 (Research track)

Researching deep integration of state-of-the-art models for automated semantic
segmentation and depth-aware clustering.

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

1.  **Palette chart + Perceptual Sorting** — UX polish, low effort
2.  **Exposed Clean-up Controls** — power-user flexibility
3.  **Lineart exclusion + numbers_density** — labeling polish
4.  **Label Collision Avoidance** — final labeling refinement
5.  **Interactive palette controls** — long-term UX
6.  **Research track (SAM/DA3/Learned Edges)** — experimental features
