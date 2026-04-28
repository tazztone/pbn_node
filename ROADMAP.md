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
| 5 | **Depth Anything 3** | 🟢 High (landscapes) | 🟠 Moderate-hard | 🟡 Partial | ★★★☆☆ |
| 6 | **Interactive Palette Controls** | 🟡 UX only | 🟠 Hard (frontend) | 📅 Planned | ★★★☆☆ |
| 7 | **SAM 3.1** | 🟢 Transformative | 🔴 Hard (arch. change) | 🧪 Research | ★★☆☆☆ |
| 8 | **Learned Edge Detection** | 🟡 Incremental | 🔴 Hard (model tuning) | 🧪 Research | ★★☆☆☆ |

---

## ✅ Completed Features

These features have been implemented and are available in the current version.

### CIEDE2000 color distance
Replaced Euclidean distance in LAB space with the `skimage.color.deltaE_ciede2000` formula. This provides perceptually correct palette selection, ensuring paint colors map more naturally to human perception.

### Shared border segmentation
The SVG generator now uses shared borders instead of independent contours for each region. This eliminates the "white-gap" problem and produces perfectly aligned vector paths.

### SLIC superpixels
Integrated `skimage.segmentation.slic()` into the pipeline to reduce pixel space to compact cells before clustering. This dramatically reduces speckle and isolates the quantizer from noise.

### Content-aware protection (Face detection)
Implemented Mediapipe-based face detection to generate protection maps. This ensures high-frequency details in portraits are preserved during color quantization.

### Budget Split (Foreground/Background)
Implemented a color budget splitting mechanism using Otsu's thresholding to separate foreground and background. This allows allocating a larger portion of the color palette to the subject of the image.

### Polylabel placement
Implemented the Polylabel algorithm for optimal label positioning at the "pole of inaccessibility" (the most distant internal point from the polygon outline).

---

## 🚀 Future Roadmap

### Depth Anything 3 (DA3)
Replacing the current Otsu-based budget split with DA3 to produce more accurate foreground/background masks and splitting the color budget proportionally.

- **Status:** Researching integration.
- **Impact:** Excellent for landscapes and complex depth scenes.

### Interactive palette controls
Building ComfyUI V3 widgets to let you manually merge or split colors. This requires deep integration with the ComfyUI frontend API.

- **Status:** Long-term goal.
- **Impact:** Improved user experience and manual control.

### SAM 3.1 (Segment Anything)
Replacing the color-first segmentation with SAM's auto-mask generator to produce semantic regions (for example, sky, skin, or objects).

- **Status:** Researching.
- **Impact:** Potentially transformative for complex scenes.

### Learned edge detection
Replacing morphological smoothing with models like DexiNed or HED to produce crisper, more artistically natural boundaries.

- **Status:** Researching.
- **Impact:** Cleanest SVG contours possible.

## Recommended implementation order

We recommend proceeding in the following order to maximize ROI:

1.  Depth Anything 3 (DA3) integration
2.  Interactive Palette Controls
3.  SAM 3.1
4.  Learned Edge Detection
