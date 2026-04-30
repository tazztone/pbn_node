# Architecture

This document describes the internal structure of the `pbn_node` ComfyUI
custom node — a modular, six-stage pipeline that transforms a raster image into
a printable paint-by-number SVG template.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Repository Layout](#2-repository-layout)
3. [Data Models](#3-data-models)
4. [Pipeline Stages](#4-pipeline-stages)
   - [Stage 1 — Preprocessing](#stage-1--preprocessing)
   - [Stage 2 — Perception & Protection](#stage-2--perception--protection)
   - [Stage 3 — Color Quantization](#stage-3--color-quantization)
   - [Stage 4 — Region Segmentation](#stage-4--region-segmentation)
   - [Stage 5 — Vectorization](#stage-5--vectorization)
   - [Stage 6 — Label Placement & SVG Generation](#stage-6--label-placement--svg-generation)
5. [Perception Stack](#5-perception-stack)
6. [ComfyUI Integration Layer](#6-comfyui-integration-layer)
7. [Rendering Layer](#7-rendering-layer)
8. [Testing Infrastructure](#8-testing-infrastructure)
9. [Dependency Map](#9-dependency-map)

---

## 1. High-Level Overview

The pipeline follows a strict **linear, single-pass** architecture. Each stage
consumes the output of the previous stage and produces a well-typed dataclass
result. No stage reaches back to a previous stage; all cross-stage communication
is done via `backend/models.py` dataclasses.

```
ComfyUI Tensor Inputs
        │
        ▼
┌─────────────────────────────────────────────────┐
│              PaintByNumberNode                  │  ← pbn_node.py
│  (ComfyUI V3 API, preset resolution,            │
│   tensor decoding, batch loop)                  │
└────────────────────┬────────────────────────────┘
                     │ ProcessingParameters + PerceptionInputs
                     ▼
┌─────────────────────────────────────────────────┐
│               ImageProcessor                    │  ← pbn_pipeline.py
│  Orchestrates all 6 stages, progress reporting  │
└──┬──────────┬──────────┬──────────┬─────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
Stage 1    Stage 2    Stage 3    Stage 4   Stage 5   Stage 6
Preproc.  Perception  Quantize  Segment  Vectorize  Label+SVG
   │          │          │          │         │         │
   └──────────┴──────────┴──────────┴─────────┴─────────┘
                         │
                         ▼
                     SVGResult
                         │
               ┌─────────┴──────────┐
               ▼                    ▼
          PBNRenderer            SVG string
          (raster output)        (vector output)
```

---

## 2. Repository Layout

```
pbn_node/
│
├── pbn_node.py               # ComfyUI node entry point & I/O layer
├── pbn_pipeline.py           # Pipeline orchestrator (ImageProcessor)
├── pbn_renderer.py           # Raster image renderer from SVGResult
├── __init__.py               # ComfyUI registration
│
├── backend/
│   ├── models.py             # ALL shared data contracts (dataclasses)
│   │
│   ├── preprocessing/
│   │   ├── preprocessor.py       # Painterly bilateral filter
│   │   ├── normal_features.py    # LAB+normal channel augmentation for SLIC
│   │   ├── retinex.py            # Multiscale Retinex (auto-albedo)
│   │   ├── protector.py          # Mediapipe face/hand protection maps
│   │   └── sapiens_priority.py   # Body-part priority weight maps
│   │
│   ├── quantization/
│   │   └── quantizer.py          # K-means + CIEDE2000 palette merge
│   │
│   ├── segmentation/
│   │   └── segmenter.py          # Watershed / majority-filter region building
│   │
│   ├── vectorization/
│   │   └── vectorizer.py         # Contour tracing, Bézier smoothing, speckle removal
│   │
│   ├── svg_generation/
│   │   └── svg_generator.py      # Shared-border SVG path assembly
│   │
│   ├── labeling/
│   │   └── label_placer.py       # Polylabel / centroid number placement
│   │
│   └── utils/
│       └── color.py              # LAB ↔ hex conversion helpers
│
├── js/                       # ComfyUI frontend extension (SVG preview widget)
│
└── tests/
    ├── unit/                 # Unit tests per module
    ├── integration/          # End-to-end pipeline tests
    ├── conftest.py           # Shared fixtures & ComfyUI mocks
    └── run_tests.py          # Test runner wrapper
```

---

## 3. Data Models

All inter-stage contracts are defined in `backend/models.py`. Nothing outside
this file defines shared data structures; stages import only from here.

| Dataclass | Owner | Purpose |
|---|---|---|
| `PerceptionInputs` | Input to pipeline | Container for all optional perception priors: albedo, segmentation mask, normal map, lineart, and their influence weights |
| `ProcessingParameters` | Input to pipeline | All scalar configuration — `num_colors`, `use_slic`, `use_watershed`, presets, SLIC tuning, smoothing kernel size, etc. |
| `ColorPalette` | Stage 3 → 4, 5, 6 | LAB color array, hex strings, and total color count |
| `RegionData` | Stage 4 → 5 | `{region_id: Polygon}`, color index map, shared border `LineString` dict, adjacency `nx.Graph` |
| `LabelData` | Stage 6 | `{region_id: Point}` positions, per-region font sizes, and the set of regions too small to label |
| `SVGResult` | Pipeline output | Assembled SVG string, `ColorPalette`, timing, `cleaned_regions`, `LabelData`, quantized raster, and `region_colors` |

`PerceptionInputs` carries all optional influence weights with validated ranges
enforced in `__post_init__`, preventing misconfigured calls from reaching the
pipeline.

---

## 4. Pipeline Stages

`ImageProcessor.process_array()` in `pbn_pipeline.py` executes all six stages
sequentially, reporting progress via the ComfyUI V3 `api.execution.set_progress`
hook.

### Stage 1 — Preprocessing

**Module:** `backend/preprocessing/preprocessor.py`

When `use_painterly_preprocess=True`, a bilateral (edge-preserving) filter is
applied using OpenCV's `stylization` with configurable `sigma_s` and `sigma_r`.
This flattens fine textures (skin pores, grass noise) while keeping structural
edges crisp, making the downstream color quantizer work on a cleaner signal.
When disabled, the raw BGR image is passed through unchanged.

### Stage 2 — Perception & Protection

**Modules:** `preprocessor.py`, `protector.py`, `sapiens_priority.py`,
`normal_features.py`

This stage assembles all optional prior information into a single upstream
influence map before quantization. Four sub-systems run here:

- **Content protection map** (`protector.py`): When `use_content_protect=True`,
  Mediapipe detects faces and hands and produces a float weight map. High-weight
  pixels receive extra color budget during quantization. `Protector` is
  lazy-loaded to avoid importing Mediapipe when not needed.

- **Sapiens priority map** (`sapiens_priority.py`): When a semantic segmentation
  mask is provided as a 2D grayscale class map (e.g., from Sapiens), this
  module converts class IDs to per-pixel priority weights. The resulting map is
  multiplied with the protection map if both are active.

- **Normal-map SLIC augmentation** (`normal_features.py`): When `use_slic=True`
  and a normal map is connected, the LAB image is extended to 5 channels
  `[L, a, b, ang_grad, curvature]` before being passed to
  `skimage.segmentation.slic`. The angular gradient (Sobel of the normal field)
  and curvature (Laplacian of the normal field) are scaled to LAB range
  (`target_scale=50.0`) so SLIC treats them as co-equal color features.
  This causes superpixel boundaries to follow surface creases that are invisible
  to pure color analysis. **This path is only active when `use_slic=True`.**

- **Standard SLIC path**: When no normal map is provided, standard RGB SLIC
  runs on the preprocessed image and its output is averaged into superpixel
  blocks before Stage 3.

### Stage 3 — Color Quantization

**Module:** `backend/quantization/quantizer.py`

The quantizer reduces the image to `num_colors` (or auto-detected count) using
K-means clustering in LAB color space.

Key behaviours:
- **Auto-albedo**: When `use_auto_albedo=True`, `multiscale_retinex()` estimates
  a shadow-free albedo before quantization. The albedo is stored in
  `PerceptionInputs` and blended with the original image at `material_weight`
  ratio during clustering.
- **Content-aware budget split**: When a `segmentation_mask` is present, the
  color budget is split between foreground and background proportionally using
  `subject_priority`. Each region gets an independent K-means run.
- **Pairwise CIEDE2000 palette merge**: After initial clustering, visually
  similar colors (below `ciede2000_merge_thresh` ΔE) are merged iteratively
  using a pairwise distance matrix. This consolidates near-identical paint pots.
- **Lineart-influenced clustering**: The `edge_influence` weight biases cluster
  centroids away from lineart boundaries, improving color coherence at edges.

Output: a `ColorPalette` and a quantized BGR image with pixel values snapped to
palette colors.

### Stage 4 — Region Segmentation

**Module:** `backend/segmentation/segmenter.py`

Contiguous same-color blobs in the quantized image are identified and converted
to Shapely `Polygon` objects. Two segmentation strategies are available,
controlled by `use_watershed`:

- **Default (majority filter)**: A vectorized majority-vote kernel
  (`smoothing_kernel_size × smoothing_kernel_size`) replaces each pixel's color
  with the most common color in its neighbourhood. This is fast and produces
  smooth, painterly regions.
- **Watershed**: Uses OpenCV's marker-based watershed transform for higher
  fidelity to the original photo's shapes. Slower but better for complex,
  textured backgrounds.

In both cases, `lineart_map` and `lineart_strength` are passed to the segmenter.
The lineart edge weights act as a hard barrier in the smoothing kernel,
preventing color bleeding across strong edges.

After segmentation, a shared-border detection pass builds the `shared_borders`
dict and `adjacency_graph` using vectorized NumPy scanning.

Output: `RegionData` with all polygons, color assignments, shared borders, and
the adjacency graph.

### Stage 5 — Vectorization

**Module:** `backend/vectorization/vectorizer.py`

Each region's pixel mask is traced into an ordered contour using OpenCV, then
simplified with the Ramer–Douglas–Peucker algorithm at `simplification`
tolerance. When `use_bezier_smooth=True`, the simplified polyline is converted
to cubic Bézier curves for a smoother, hand-drawn aesthetic.

After contouring, a speckle removal pass merges regions whose bounding box
falls below `speckle_threshold` pixels into the most color-adjacent neighbour.
Regions are then renumbered to consecutive IDs (1, 2, 3, …) in the pipeline
orchestrator.

Output: `cleaned_regions` dict and an updated `region_colors` mapping.

### Stage 6 — Label Placement & SVG Generation

**Modules:** `backend/labeling/label_placer.py`,
`backend/svg_generation/svg_generator.py`

**Label placement** (`LabelPlacer`): Determines where to print each region's
number. Two modes:
- `polylabel`: Uses the Polylabel algorithm (pole of inaccessibility) to find
  the largest inscribed circle centre — optimal for concave or irregular shapes.
- `centroid`: Uses the Shapely centroid — faster but may fall outside the polygon
  for concave regions.

Font size scales with region area. Regions below a minimum size are added to
`skipped_regions` and omitted from SVG labels.

**SVG generation** (`SVGGenerator`): Assembles the final SVG document. When
`use_shared_borders=True`, adjacent region boundaries are rendered as a single
shared `<path>` element rather than two overlapping strokes. This eliminates
white-gap artefacts between regions at any zoom level. Both a coloured preview
mode and a print-ready monochrome mode (`print_svg`) are supported.

---

## 5. Perception Stack

The perception stack is the system of optional prior inputs that guide the
pipeline toward semantically better results. Each prior operates at a different
stage with a different mechanism:

| Prior | Input field | Stage | Mechanism | Key constraint |
|---|---|---|---|---|
| **Segmentation mask** | `segmentation` | Stage 3 | Splits color budget per region; builds priority map | Grayscale class map or RGB-packed |
| **Lineart / edge map** | `lineart` | Stage 3 + 4 | Biases quantizer centroids; acts as barrier in smoothing kernel | Any single-channel edge map; `invert_lineart` to flip polarity |
| **Normal map** | `normals` | Stage 2 (SLIC) | Augments LAB image with angular gradient + curvature channels | **Only active when `use_slic=True`** |
| **Albedo** | Internal / auto | Stage 3 | Blends shadow-free color estimate into quantizer input | Auto-computed via Retinex when `use_auto_albedo=True` |

These inputs are decoded from ComfyUI tensors in `pbn_node.py` and bundled
into a `PerceptionInputs` instance. If none are provided, `perception=None`
is passed to the pipeline and all perception-dependent branches are skipped.

---

## 6. ComfyUI Integration Layer

**File:** `pbn_node.py`

`PaintByNumberNode` is a `io.ComfyNode` subclass using the **ComfyUI V3 API**.
It has two class methods:

- `define_schema()`: Declares all inputs, outputs, and their metadata
  (tooltips, defaults, ranges, `advanced=True` flags). This is the contract
  between ComfyUI and the node.
- `execute()`: Entry point called by ComfyUI per execution. Performs four
  responsibilities:
  1. **Preset resolution** (`_resolve_presets`): Applies preset overrides
     from the `PRESETS` dict, then layer user-set parameters on top.
  2. **Tensor decoding** (`_prepare_perception_inputs`, `_decode_*` statics):
     Converts ComfyUI float32 `[B, H, W, C]` tensors to NumPy arrays at the
     correct type and range for each perception input.
  3. **Batch loop**: Processes each image in the batch independently, sharing
     a single `ImageProcessor` instance.
  4. **SVG persistence** (`_save_svg_batch`): Writes SVGs to ComfyUI's temp
     directory using content-addressed MD5 filenames (idempotent writes).

The `js/` directory contains a ComfyUI frontend extension that registers a
custom widget to render the SVG string as a live, zoomable vector preview
directly in the node body.

---

## 7. Rendering Layer

**File:** `pbn_renderer.py`

`PBNRenderer` is a thin raster-output adapter that sits outside the pipeline.
It takes the `SVGResult`'s cleaned regions and re-draws them to a BGR NumPy
array at the original image resolution. It supports four output modes:

| Mode | Description |
|---|---|
| `colored` | Filled regions with their palette color + number labels |
| `outline` | White background with black region outlines + labels (print-ready) |
| `quantized` | The raw quantized raster before vectorization (pipeline passthrough) |
| `print_svg` | Not rasterized — SVG is the primary output for this mode |

The renderer is intentionally separate from `SVGGenerator` so that raster and
vector outputs can evolve independently.

---

## 8. Testing Infrastructure

**Directory:** `tests/`

The test suite uses `pytest` with a custom runner (`run_tests.py`) that patches
ComfyUI's module system before importing anything from the node, allowing tests
to run without a live ComfyUI installation.

`conftest.py` provides shared fixtures including:
- A synthetic 64×64 BGR test image
- Pre-built `ProcessingParameters` and `PerceptionInputs` instances
- ComfyUI API mocks (`folder_paths`, `torch`, `comfy_api`)

Tests are split into two layers:

- **Unit tests** (`tests/unit/`): Test individual modules in isolation —
  quantizer, segmenter, vectorizer, label placer, SVG generator, and all
  preprocessing utilities.
- **Integration tests** (`tests/integration/`): Run `ImageProcessor.process_array()`
  end-to-end on synthetic images and assert on `SVGResult` properties (region
  count, color count, SVG validity).

---

## 9. Dependency Map

```
pbn_node.py
  └── pbn_pipeline.py (ImageProcessor)
        ├── backend/models.py
        ├── backend/preprocessing/preprocessor.py
        │     └── cv2 (stylization)
        ├── backend/preprocessing/normal_features.py
        │     └── cv2 (Sobel, Laplacian)
        ├── backend/preprocessing/retinex.py
        │     └── numpy, cv2
        ├── backend/preprocessing/protector.py          [lazy-loaded]
        │     └── mediapipe
        ├── backend/preprocessing/sapiens_priority.py
        │     └── numpy
        ├── backend/quantization/quantizer.py
        │     └── numpy, sklearn (KMeans), skimage (deltaE_ciede2000)
        ├── backend/segmentation/segmenter.py
        │     └── numpy, cv2, scipy, shapely, networkx
        ├── backend/vectorization/vectorizer.py
        │     └── numpy, cv2, shapely
        ├── backend/labeling/label_placer.py
        │     └── shapely, polylabel
        └── backend/svg_generation/svg_generator.py
              └── shapely

pbn_renderer.py
  └── backend/models.py, numpy, cv2

pbn_node.py (external deps)
  └── torch, comfy_api (ComfyUI V3), folder_paths
```

Core scientific stack: `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`,
`scipy`, `shapely`, `networkx`.

Optional heavy dependency: `mediapipe` (only imported when
`use_content_protect=True` or `use_auto_mask=True`, via lazy `Protector`
instantiation in `ImageProcessor`).
