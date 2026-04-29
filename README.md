# ComfyUI paint by number

ComfyUI paint by number transforms your digital images into high-quality,
printable paint-by-number templates. It lets you generate vector-aligned regions
with accurate labels directly within your ComfyUI workflows.

## Features

This node provides several advanced tools for creating paint-by-number assets.
It's built on a modular backend that ensures high performance and reliable
results.

- **V3 API support:** Built on the modern ComfyUI V3 API for better performance
  and compatibility.
- **Fast smoothing:** A vectorized majority filter provides real-time region
  smoothing even on high-resolution images.
- **Multiple output modes:** You can choose between colored previews,
  print-ready outlines, or raw quantized views.
- **Automatic color detection:** The node automatically selects the optimal
  number of colors for your image if you don't specify a count.
- **SVG export:** Generates raw SVG content that you can use in vector editing
  software like Inkscape or Adobe Illustrator.
- **Vector preview:** Real-time, resolution-independent SVG preview directly
  within the ComfyUI node body.
- **Perception stack:** Support for albedo, segmentation masks, lineart, and
  normal maps to guide color quantization and boundary detection.
- **Auto-albedo:** Integrated Retinex-based shadow removal to estimate clean
  material colors when external maps are missing.
- **Normal-Map-Guided SLIC:** Leverages 3D surface geometry to ensure region
  boundaries follow physical creases and anatomical features.
- **Sapiens Adaptive Priority:** Automatically prioritizes faces, hands, and
  subject regions for higher detail preservation.

## Installation

To install the node, follow these steps. Note that this project requires Python
3.10 or higher.

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/tazztone/pbn_node
    ```
3.  Install the required Python dependencies:
    ```bash
    pip install -r pbn_node/requirements.txt
    ```

<!-- prettier-ignore -->
> [!NOTE]
> You must restart ComfyUI after installation for the node to appear in the
> menu.

## Usage

You can find the node in the ComfyUI menu under **image** > **process** >
**Paint By Number**. It integrates seamlessly with other image processing
nodes.

### Inputs

The node accepts several configuration parameters to fine-tune the result.

- **image:** The input image tensor you want to process.
- **albedo:** Optional albedo image used to guide color clustering. This helps
  isolate material color from lighting conditions.
- **segmentation:** Optional segmentation or mask image. When provided, the node
  allocates color budgets proportionally to the detected regions.
- **lineart:** Optional edge map (e.g., from Canny or HED). This prevents color
  regions from bleeding across semantic boundaries.
- **num_colors:** The number of color clusters to use. Set this to `0` to enable
  automatic detection.
- **simplification:** Controls how much the region contours are simplified.
  Accepted values range from `0.5` to `2.0`.
- **use_watershed:** When enabled, uses the watershed transform for
  segmentation. This is more accurate to the original specification but slower.
- **output_mode:** Selects the visual style of the output image (**colored**,
  **outline**, or **quantized**).
- **subject_priority:** A multiplier for color allocation to non-background
  segments when using a segmentation mask.
- **material_weight:** Controls the influence of the albedo map over the
  original photo during quantization.
- **edge_influence:** Controls how much the lineart edge map biases color
  quantization. High values ensure color boundaries follow edges.
- **normals:** Optional surface normal map (from Sapiens or Depth-to-Normal).
  Causes superpixel boundaries to align with 3D surface creases even when
  colors are similar.
- **normal_strength:** How strongly the normal map influences superpixel shapes.
- **segmentation_format:** How to interpret the segmentation map (`auto`,
  `grayscale`, or `rgb_packed`).
- **use_auto_albedo:** When enabled, automatically estimates a shadow-free
  albedo map if no external albedo is provided.
- **use_painterly_preprocess:** Applies a stylization filter before processing
  to simplify complex textures and produce cleaner shapes.

### Outputs

The node returns three values that you can use in your workflow.

- **IMAGE:** The rendered raster template based on your selected output mode.
- **SVG:** A string containing the raw SVG data for the generated template. This
  result is also previewed as a vector graphic within the node UI.
- **COLOR_COUNT:** An integer representing the total number of colors used in
  the final palette.

## Development

We use modern static analysis tools to maintain code quality. If you want to
contribute to the project, we recommend using `uv` to run these tools without
manual installation.

### Linting and formatting

We use `ruff` for linting and code formatting. You can run these checks using
the following commands:

```bash
uvx ruff check .
uvx ruff format .
```

### Type checking

We use `mypy` for static type checking to catch potential logic errors early.
Run it with this command:

```bash
uvx mypy .
```

### Pre-commit hooks

We use `pre-commit` to automatically run linting, formatting, and type checking
before each commit. To set up the hooks, install `pre-commit` and run the
installation command:

```bash
pip install pre-commit
pre-commit install
```

Once installed, the hooks will run automatically on every commit. You can also
run them manually on all files at any time:

```bash
pre-commit run --all-files
```

### Automated testing

We use `pytest` for unit and integration testing. To ensure all mocks load
correctly, you must run tests using the provided wrapper script:

```bash
../../venv/bin/python tests/run_tests.py
```

For more details on the testing infrastructure, see
[tests/TESTING.md](tests/TESTING.md).

## Next steps

Once you've installed the node, there are several ways to extend your workflow.

- Try connecting the **SVG** output to a file-saving node to export vector
  templates.
- Use the **outline** mode to generate templates ready for physical printing.
- Experiment with the **simplification** setting to balance detail and ease of
  painting.

## Roadmap

For the full list of planned features and their implementation priority, see
[ROADMAP.md](ROADMAP.md).

## License

This project is licensed under the MIT License. See the `LICENSE` file for
details.
