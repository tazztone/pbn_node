# Project roadmap

This document outlines the planned improvements and research directions for the
ComfyUI paint-by-number node. These features are ranked by their potential
impact on output quality and their ease of implementation.

## Overview

We have categorized the following candidates into tiers based on their ROI
(Return on Investment). Our goal is to move from a color-first pipeline to a
perception-aware and content-aware system.

| # | Feature | Effectiveness | Impl. Difficulty | ROI |
|---|---|---|---|---|
| 1 | **CIEDE2000** | 🟢 High | 🟢 Trivial (10 lines) | ★★★★★ |
| 2 | **Shared Border SVG wiring** | 🟢 High | 🟢 Low (already scaffolded) | ★★★★★ |
| 3 | **SLIC Superpixels** | 🟡 Medium-high | 🟢 Low (no GPU) | ★★★★☆ |
| 4 | **Content-Aware Preprocessing** | 🟢 High (portraits) | 🟡 Moderate | ★★★★☆ |
| 5 | **Depth Anything 3** | 🟢 High (landscapes) | 🟠 Moderate-hard | ★★★☆☆ |
| 6 | **Interactive Palette Controls** | 🟡 UX only | 🟠 Hard (frontend) | ★★★☆☆ |
| 7 | **SAM 3.1** | 🟢 Transformative | 🔴 Hard (arch. change) | ★★☆☆☆ |
| 8 | **Learned Edge Detection** | 🟡 Incremental | 🔴 Hard (model tuning) | ★★☆☆☆ |

---

## Tier 1: High impact, low effort

These features provide immediate quality improvements with minimal architectural
changes.

### CIEDE2000 color distance
Replacing Euclidean distance in LAB space with the `skimage.color.deltaE_ciede2000`
formula. This provides perceptually correct palette selection, ensuring paint
colors map more naturally to human perception.

- **Status:** Recommended next step.
- **Implementation:** pure drop-in in `quantizer.py`.

### Shared border segmentation
The core algorithmic work for shared border segmentation is already scaffolded.
The remaining task is wiring these borders into the SVG path builder instead of
drawing independent contours for each region.

- **Status:** Already scaffolded in `segmenter.py`.
- **Impact:** Eliminates the white-gap problem in SVG outputs.

### SLIC superpixels
Inserting `skimage.segmentation.slic()` into the pipeline to reduce pixel space
to compact cells before clustering. This is a lightweight approach that doesn't
require a GPU.

- **Status:** Planned.
- **Impact:** Dramatically reduces speckle and isolates the quantizer from noise.

---

## Tier 2: High impact, moderate effort

These features require new dependencies or more complex logic but offer
significant quality gains for specific use cases.

### Content-aware preprocessing
Adding optional face or object detection (for example, using Mediapipe or YOLO)
to weight sampling during quantization. This ensures high-frequency details in
portraits are protected.

- **Status:** Under consideration.
- **Impact:** Significant quality improvement for portraits.

### Depth Anything 3 (DA3)
Using DA3 to produce foreground/background masks and splitting the color budget
proportionally. This transforms flat-background images and improves depth
perception in the output.

- **Status:** Under consideration.
- **Impact:** Excellent for landscapes and portraits.

---

## Tier 3: High quality, high effort

These features represent significant architectural changes or complex frontend
development.

### Interactive palette controls
Building ComfyUI V3 widgets to let you manually merge or split colors. This
requires deep integration with the ComfyUI frontend API.

- **Status:** Long-term goal.
- **Impact:** Improved user experience and manual control.

### SAM 3.1 (Segment Anything)
Replacing the color-first segmentation with SAM's auto-mask generator to produce
semantic regions (for example, sky, skin, or objects).

- **Status:** Researching.
- **Impact:** Potentially transformative for complex scenes.

### Learned edge detection
Replacing morphological smoothing with models like DexiNed or HED to produce
crisper, more artistically natural boundaries.

- **Status:** Researching.
- **Impact:** Cleanest SVG contours possible.

## Recommended implementation order

We recommend proceeding in the following order to maximize ROI:

1.  CIEDE2000
2.  Shared Border wiring
3.  SLIC Superpixels
4.  Content-aware preprocessing
5.  Depth Anything 3
6.  Interactive Palette Controls
7.  SAM 3.1
8.  Learned Edge Detection
