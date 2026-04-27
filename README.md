# ComfyUI paint by number

ComfyUI paint by number is a custom node for ComfyUI that transforms your digital
images into high-quality, printable paint-by-number templates. It lets you
generate vector-aligned regions with accurate labels directly within your
ComfyUI workflows.

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
- **num_colors:** The number of color clusters to use. Set this to `0` to enable
  automatic detection.
- **simplification:** Controls how much the region contours are simplified.
  Accepted values range from `0.5` to `2.0`.
- **use_watershed:** When enabled, uses the watershed transform for
  segmentation. This is more accurate to the original specification but slower.
- **output_mode:** Selects the visual style of the output image (**colored**,
  **outline**, or **quantized**).

### Outputs

The node returns three values that you can use in your workflow.

- **IMAGE:** The rendered raster template based on your selected output mode.
- **SVG:** A string containing the raw SVG data for the generated template.
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

## Next steps

Once you've installed the node, there are several ways to extend your workflow.

- Try connecting the **SVG** output to a file-saving node to export vector
  templates.
- Use the **outline** mode to generate templates ready for physical printing.
- Experiment with the **simplification** setting to balance detail and ease of
  painting.

## Roadmap

To reach professional-grade output, we are planning to integrate the following "best of breed" features from other PBN research:

- [ ] **Shared Border Segmentation**: Porting the "Facet Border Segmenter" logic from `paintbynumbersgenerator` to eliminate SVG gaps caused by independent shape simplification.
- [ ] **Content-Aware Preprocessing**: Adding face and object detection (from `my-paint-by-numbers`) to protect high-frequency details in portraits and pets.
- [ ] **CIEDE2000 Color Distance**: Moving from Euclidean LAB to the industry-standard CIEDE2000 formula (from `paintr`) for superior perceptual color matching.
- [ ] **Interactive Palette Controls**: Adding ComfyUI widget support for manual color merging and splitting (from `paintbynumbers`).

### AI / Deep Learning Candidates

The following advanced techniques are under consideration to further improve segmentation quality and region semantics:

- [ ] **SAM 3.1 (Segment Anything with Concepts)**: Replace or augment `direct_color_segmentation()` with Meta's [SAM 3.1](https://huggingface.co/facebook/sam3.1) auto-mask generator. SAM 3.1 extends SAM 2 with open-vocabulary concept segmentation (text prompts) and Object Multiplex for ~7x faster multi-object tracking. Would be exposed as an optional `use_sam` toggle alongside the existing `use_watershed` flag.
- [ ] **Depth Anything 3 (DA3)**: Add a depth-map preprocessing stage using [DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE-1.1) (0.35B params, ByteDance Seed) to split the color budget between foreground and background. DA3 outperforms both Depth Anything 2 and VGGT for monocular depth, and also provides confidence maps that can weight region importance during quantization — particularly beneficial for portraits and landscapes.
- [ ] **SLIC Superpixels**: Insert a superpixel pre-segmentation step (scikit-image `slic`) between preprocessing and color quantization. Superpixels constrain KMeans to perceptually uniform regions, dramatically reducing speckle without the cost of a full neural inference pass.
- [ ] **Learned Edge Detection (DexiNed / HED)**: Replace the morphological smoothing in `segmenter.py` with a learned edge detector to produce crisper, more artistically natural contour boundaries and cleaner SVG paths.

## License

This project is licensed under the MIT License. See the `LICENSE` file for
details.
