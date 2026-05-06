# PBN Node — Hardening & Simplification Implementation Plan

**Repository:** [tazztone/pbn_node](https://github.com/tazztone/pbn_node)
**Date:** 2026-05-06
**Scope:** 11 confirmed issues across 6 files — Critical → Low severity

---

## Executive Summary

A full audit of `pbn_node` surfaced 11 confirmed bugs and structural issues spanning
the ComfyUI integration layer, the processing pipeline, segmentation, and label
placement. Four issues are Critical or High severity and directly produce incorrect
output: wrong perception inputs applied across a batch, double-rendering with wrong
mode semantics, SVG/raster output mismatch in outline mode, and missing outlines on
disjoint island regions in print SVG. The remaining issues are medium/low but compound
maintainability debt significantly. This plan addresses all 11 in priority order with
exact file locations, minimal diffs, and acceptance criteria.

---

## Issue Registry

| # | Title | Severity | File(s) |
|---|-------|----------|---------|
| 1 | Batch perception alignment bug | **Critical** | `pbn_node.py` |
| 2 | Double-rendering & `quantized` mode semantics | **High** | `pbn_pipeline.py`, `pbn_node.py` |
| 3 | SVG/raster outline mode mismatch | **High** | `pbn_pipeline.py` |
| 4 | Disjoint island border erasure in print SVG | **Medium** | `backend/segmentation/segmenter.py` |
| 5 | Polylabel CPU freeze / timeout bypass | **Medium** | `backend/labeling/label_placer.py` |
| 6 | `output_mode` variable shadowing in batch loop | **High** | `pbn_node.py` |
| 7 | `kwargs`→`params`→`ProcessingParameters` triple-representation | **High** | `pbn_node.py` |
| 8 | Duplicate `portrait` preset override | Low | `pbn_node.py` |
| 9 | `use_clahe` and `invert_lineart` dead parameters | Low | `pbn_node.py`, `backend/models.py` |
| 10 | `ciede2000_merge_thresh` default mismatch (8.0 vs 10.0) | Low | `pbn_node.py`, `backend/models.py` |
| 11 | `RegionSegmenter` / `Vectorizer` re-instantiated per call | Low | `pbn_pipeline.py` |

---

## Phase 1 — Critical & High Severity (do first)

### Issue 1 — Batch Perception Alignment Bug

**Severity:** Critical
**File:** `pbn_node.py` — `execute()` and `_prepare_perception_inputs()`

**Root cause confirmed:** `_prepare_perception_inputs(kwargs, params)` is called
once, before the batch loop. Inside `_decode_lineart` and `_decode_segmentation`,
the code always grabs `t[0]` — the first frame of the tensor — regardless of the
current batch index `i`. Every image after the first receives the wrong perception
inputs.

**Fix — move perception decoding inside the batch loop:**

```python
# BEFORE (broken — called once outside the loop):
perception = cls._prepare_perception_inputs(kwargs, params)

for i in range(batch_size):
    img_tensor = image[i]
    ...
    result = processor.process_array(img_bgr, proc_params, api=api)

# AFTER (correct — decode per-frame inside the loop):
for i in range(batch_size):
    img_tensor = image[i]

    # Slice the batch tensors to the i-th frame before decoding
    lineart_i = kwargs["lineart"][i:i+1] if kwargs.get("lineart") is not None else None
    seg_i = kwargs["segmentation"][i:i+1] if kwargs.get("segmentation") is not None else None

    perception_i = cls._prepare_perception_inputs_for_frame(
        lineart_i, seg_i, kwargs, params
    )
    # Build proc_params_i with perception_i (dataclasses.replace):
    proc_params_i = dataclasses.replace(proc_params, perception=perception_i)

    img_bgr = cls._torch_to_bgr(img_tensor)
    result = processor.process_array(img_bgr, proc_params_i, api=api)
```

Extract a new static helper `_prepare_perception_inputs_for_frame(lineart_t, seg_t, kwargs, params)`
that accepts pre-sliced single-frame tensors instead of the full batch.

**Acceptance criteria:**
- Processing a batch of 3 images with 3 different segmentation masks applies each
  mask only to its corresponding image.
- `_decode_lineart` and `_decode_segmentation` no longer contain `t[0]` when called
  from the per-frame path.

---

### Issue 2 — Double-Rendering & `quantized` Output Mode Semantics

**Severity:** High
**Files:** `pbn_pipeline.py` (lines ~110–125), `pbn_node.py` (batch loop render block)

**Root cause confirmed:** `pbn_pipeline.py` calls `self.renderer.render(... mode="colored" ...)`
at the end of `process_array()` and stores the result as `SVGResult.quantized`. Then
`pbn_node.py` checks `if output_mode == "quantized": result_bgr = result.quantized`
— meaning "quantized" mode outputs a fully-rendered colored+labeled preview, not
the raw posterized raster. For `output_mode == "colored"`, the node calls `renderer.render`
a second time, duplicating the most expensive rendering step.

**Fix — two parts:**

**Part A:** In `pbn_pipeline.py`, store the actual flat posterized raster (the direct
output of the color quantizer, before any vectorization) as `SVGResult.quantized`,
and remove the final `renderer.render()` call from `process_array()`:

```python
# In process_array(), REMOVE the final renderer.render() call.
# Store the raw quantized BGR raster instead:
return SVGResult(
    ...
    quantized=quantized,   # ← the flat BGR array from Stage 3, not a rendered preview
    ...
)
```

**Part B:** In `pbn_node.py`, let the node always call `renderer.render()` exactly
once, keyed on `output_mode`. The `"quantized"` branch now correctly returns the
flat raster:

```python
output_mode_resolved = params.get("output_mode", "colored")

if output_mode_resolved == "quantized":
    result_bgr = result.quantized          # flat posterized raster — correct
else:
    result_bgr = renderer.render(
        result.cleaned_regions,
        result.label_data,
        result.color_palette,
        w, h,
        mode=output_mode_resolved,
        region_colors=result.region_colors,
        shared_borders=result.shared_borders,
        use_shared_borders=params.get("use_shared_borders", True),
    )
```

**Acceptance criteria:**
- `output_mode = "quantized"` outputs a flat posterized image with no labels, no
  outlines — only palette-snapped pixel colors.
- `output_mode = "colored"` calls `renderer.render()` exactly once per image.
- `pbn_pipeline.py` no longer imports or calls `PBNRenderer` directly.

---

### Issue 3 — SVG/Raster Outline Mode Mismatch

**Severity:** High
**File:** `pbn_pipeline.py` — `process_array()` SVG generation call

**Root cause confirmed:** `print_mode` is passed to `svg_generator.generate_svg()` as
`(p.output_mode == "print_svg")`. When `output_mode = "outline"`, `print_mode` is
`False`, so the SVG generator produces a fully-colored SVG with filled regions while
the raster preview correctly shows black outlines on white — direct contradiction.

**Fix — one line:**

```python
# BEFORE:
svg_content = self.svg_generator.generate_svg(
    ...
    print_mode=(p.output_mode == "print_svg"),
)

# AFTER:
svg_content = self.svg_generator.generate_svg(
    ...
    print_mode=(p.output_mode in ("outline", "print_svg")),
)
```

**Acceptance criteria:**
- `output_mode = "outline"` produces an SVG with black strokes and white/no fills,
  matching the raster preview.
- `output_mode = "print_svg"` behavior is unchanged.

---

### Issue 6 — `output_mode` Variable Shadowing in Batch Loop

**Severity:** High
**File:** `pbn_node.py` — `execute()`, batch loop

**Root cause confirmed:** The function parameter `output_mode` is overwritten inside
the loop with `output_mode = params.get("output_mode", "colored")` on each iteration.
This shadows the outer parameter name and makes the code fragile.

**Fix:** Use a distinct local name (e.g., `output_mode_resolved`) for the resolved
value throughout `execute()` and the batch loop. See also Issue 2 fix above which
resolves this simultaneously.

---

### Issue 7 — `kwargs`→`params`→`ProcessingParameters` Triple Representation

**Severity:** High
**File:** `pbn_node.py` — `execute()`

**Root cause confirmed:** All inputs are packed into `kwargs`, passed to
`_resolve_presets(kwargs)` which produces a second dict `params`, which is then
unpacked via `.get()` calls to construct `ProcessingParameters`. Three representations
of the same data exist simultaneously, and the `ProcessingParameters.__post_init__`
validation is bypassed for any preset-overridden value that goes through the dict path.

**Fix — collapse to a single flow:**

```python
@classmethod
def execute(cls, image, num_colors=24, ..., preset="balanced"):

    # 1. Build ProcessingParameters directly from declared args
    raw_params = ProcessingParameters(
        num_colors=num_colors if num_colors > 0 else None,
        simplification=simplification,
        ...
    )

    # 2. Apply preset overrides via dataclasses.replace (stays typed + validated)
    proc_params = cls._apply_preset(raw_params, preset)

    # 3. Decode perception inputs (per-frame inside the loop — see Issue 1 fix)
    ...
```

Rename `_resolve_presets` → `_apply_preset(params: ProcessingParameters, preset: str) -> ProcessingParameters`
and have it return a `dataclasses.replace(params, **PRESETS[preset])` when applicable.
This ensures every override passes through `__post_init__` validation.

**Acceptance criteria:**
- `execute()` contains no `kwargs` dict.
- `_resolve_presets` is replaced by `_apply_preset` that accepts and returns
  `ProcessingParameters`.
- All 11+ fields are validated by `__post_init__` regardless of preset.

---

## Phase 2 — Medium Severity

### Issue 4 — Disjoint Island Border Erasure Bug

**Severity:** Medium
**File:** `backend/segmentation/segmenter.py` — `segment()`

**Root cause confirmed:** In `segment()`, `shared_border_segmentation(segmented)` is
called on the intermediate integer matrix before the contour extraction loop. Inside
the contour loop, multi-contour regions (disjoint islands) are assigned new IDs
starting from `next_id = np.max(region_ids) + 1`. These new IDs are never added to
`shared_borders`. In `print_svg` mode the SVG generator draws region fills with
`stroke="none"` and uses `shared_borders` exclusively for visible outlines — so
island regions have invisible borders in the final printable SVG.

**Fix — reorder: run `shared_border_segmentation` after the contour loop on the
final, fully-split region map:**

```python
def segment(self, quantized, colors) -> RegionData:
    ...
    # Stage A: build initial raster regions
    segmented, region_colors = self._get_regions_pbnify(smoothed)

    # Stage B: extract polygons, splitting multi-contour regions into islands
    regions, region_colors = self._extract_polygons(segmented, region_colors)

    # Stage C: rebuild the final integer matrix from the polygon set
    final_matrix = self._regions_to_matrix(regions, quantized.shape[:2])

    # Stage D: NOW build shared borders on the final, complete region map
    shared_borders = self.shared_border_segmentation(final_matrix)
    adjacency_graph = self.build_adjacency_graph(final_matrix)

    return RegionData(regions=regions, region_colors=region_colors,
                      shared_borders=shared_borders,
                      adjacency_graph=adjacency_graph,
                      segmented_matrix=final_matrix)
```

Extract the contour loop from `segment()` into a private `_extract_polygons(segmented, region_colors)`
method that returns `(regions_dict, region_colors_dict)`. Add a helper
`_regions_to_matrix(regions, shape)` that rasterizes the final polygon dict back to
an integer matrix for the border scan.

**Acceptance criteria:**
- Every region ID present in the `regions` dict has a corresponding entry (possibly
  empty list) in `shared_borders`.
- `output_mode = "print_svg"` on an image with fragmented regions (e.g., a sky band
  split by a foreground subject) shows complete outlines on all island fragments.

---

### Issue 5 — Polylabel CPU Freeze / Timeout Bypass

**Severity:** Medium
**File:** `backend/labeling/label_placer.py` — `polylabel_placement()`

**Root cause confirmed:** The 100 ms timeout is checked *before* calling `polylabel()`,
but `polylabel()` itself is a blocking call. A high-vertex polygon from a detailed
region can take several seconds inside a single `polylabel()` invocation, completely
bypassing the guard. The timeout is effectively non-functional.

**Fix — pre-simplify the polygon before the polylabel call:**

```python
def polylabel_placement(self, polygon: Polygon, precision: float = 1.0) -> Point:
    # Pre-simplify to bound vertex count; tolerance=1.0px retains visual shape
    # while reducing a 10,000-vertex polygon to ~50 vertices
    working_polygon = polygon.simplify(tolerance=1.0, preserve_topology=True)
    if not working_polygon.is_valid or working_polygon.is_empty:
        working_polygon = polygon  # fall back to original if simplification breaks it

    start_time = time.time()
    current_precision = precision

    while current_precision >= self.min_precision:
        try:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.timeout_ms:
                return working_polygon.centroid

            label_point = polylabel(working_polygon, tolerance=current_precision)
            return label_point
        except Exception:
            current_precision /= 2

    return working_polygon.centroid
```

Also apply the same pre-simplification in `inscribed_circle_radius()` which calls
`polylabel` directly without any timeout guard.

**Acceptance criteria:**
- No label placement call blocks for more than ~200 ms on any real-world input.
- Pre-simplified polygon produces a label point indistinguishable from the
  full-resolution result at normal zoom levels (tolerance 1.0px is sub-pixel at
  print resolutions).

---

## Phase 3 — Low Severity / Cleanup

### Issue 8 — Duplicate `portrait` Preset Override

**File:** `pbn_node.py` — `_resolve_presets()` / `_apply_preset()`

The `PRESETS["portrait"]` dict already sets `use_auto_albedo: True`. The function
body also contains an explicit `if preset == "portrait": params["use_auto_albedo"] = True`
block, which is redundant and will silently diverge from the dict in future edits.

**Fix:** Delete the `if preset == "portrait":` block. The dict is the single source
of truth.

---

### Issue 9 — Dead Parameters: `use_clahe` and `invert_lineart`

**Files:** `pbn_node.py`, `backend/models.py`

- `use_clahe` is present in `ProcessingParameters` and plumbed through the pipeline
  to `preprocessor.preprocess()`, but has no UI input in `define_schema()` and is
  permanently `False`.
- `invert_lineart` is declared as a parameter in `execute()` but has no UI input;
  it is always `False`.

**Fix options:**
- **Wire them:** Add `io.Boolean.Input("use_clahe", ...)` and `io.Boolean.Input("invert_lineart", ...)`
  to `define_schema()` with `advanced=True`. CLAHE is a legitimate quality lever
  worth exposing.
- **Remove them:** Delete both parameters from the signature, model, and pipeline
  if they are not intended for users.

Either choice is fine; the current state (present but unreachable) is the worst option.

---

### Issue 10 — `ciede2000_merge_thresh` Default Mismatch

**Files:** `pbn_node.py` (default `10.0`), `backend/models.py` (default `8.0`)

When the node is called via the ComfyUI UI, the UI default of `10.0` takes effect.
When `ProcessingParameters` is constructed directly in tests or scripts, `8.0` is
used. This creates a silent behavior difference between UI and programmatic usage.

**Fix:** Align to `10.0` everywhere:
- `ProcessingParameters.ciede2000_merge_thresh: float = 10.0` in `models.py`
- Keep `io.Float.Input("ciede2000_merge_thresh", default=10.0, ...)` in `pbn_node.py`
- Update the `__post_init__` validator accordingly if needed

---

### Issue 11 — `RegionSegmenter` and `Vectorizer` Re-instantiated Per Call

**File:** `pbn_pipeline.py` — `process_array()`

Both `RegionSegmenter(...)` and `Vectorizer(...)` carry no inter-call state. They
are constructed fresh inside `process_array()` on every call, but the constructor
parameters come directly from `p` (the `ProcessingParameters`). This is minor
overhead but creates inconsistency: `ColorQuantizer`, `Preprocessor`, and
`SVGGenerator` are stored as instance attributes on `ImageProcessor`, while these
two are not.

**Fix:** Either store them as instance attributes (updating parameters before each
call) or convert their core logic to `@staticmethod` / module-level functions
accepting params explicitly. The latter is cleaner given they have no accumulated
state.

---

## Execution Order

Complete Phase 1 as a single atomic PR — these issues interact (Issues 1, 2, 6, 7
all touch the `execute()` method and the `proc_params` construction). Mixing partial
fixes risks introducing new regressions.

| Phase | Issues | Suggested branch | Notes |
|-------|--------|-----------------|-------|
| 1 | #1, #2, #3, #6, #7 | `fix/critical-pipeline-hardening` | All touch `pbn_node.py` / `pbn_pipeline.py` — do atomically |
| 2 | #4, #5 | `fix/segmentation-label-hardening` | Independent of Phase 1 |
| 3 | #8, #9, #10, #11 | `chore/cleanup` | Pure cleanup — no behavior change for wired paths |

---

## Test Coverage Requirements

Each Phase 1 fix must be accompanied by a test:

- **Issue 1:** Add an integration test in `tests/integration/` that passes a batch
  of 2 synthetic images with 2 distinct segmentation masks and asserts that
  `SVGResult.color_palette.color_count` differs between them (impossible if both use
  mask[0]).
- **Issue 2:** Add a unit test asserting `SVGResult.quantized` for a known input
  contains only colors from the palette (i.e., is a flat quantized raster, not a
  labeled preview).
- **Issue 3:** Add a test that calls `process_array` with `output_mode="outline"` and
  asserts `svg_content` does not contain `fill="#` (i.e., no colored fills).
- **Issues 4 & 5:** Add unit tests in `tests/unit/` targeting `segmenter.segment()`
  with a known multi-island input and `label_placer.polylabel_placement()` with a
  high-vertex polygon, asserting sub-200ms completion.
