# ComfyUI paint by number

ComfyUI paint by number is a custom node for ComfyUI that transforms your digital
images into high-quality, printable paint-by-number templates. It lets you
generate vector-aligned regions with accurate labels directly within your
ComfyUI workflows.

## Features

This node provides several advanced tools for creating paint-by-number assets:

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

To install the node, follow these steps:

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
**Paint By Number**.

### Inputs

The node accepts the following configuration parameters:

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

The node returns three values:

- **IMAGE:** The rendered raster template based on your selected output mode.
- **SVG:** A string containing the raw SVG data for the generated template.
- **COLOR_COUNT:** An integer representing the total number of colors used in
  the final palette.

## Next steps

- Try connecting the **SVG** output to a file-saving node to export vector
  templates.
- Use the **outline** mode to generate templates ready for physical printing.
- Experiment with the **simplification** setting to balance detail and ease of
  painting.

## License

This project is licensed under the MIT License.
