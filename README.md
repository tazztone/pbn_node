# ComfyUI Paint By Number

A custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that transforms digital images into high-quality paint-by-number templates.

## Features

- **V3 API Support**: Built using the modern ComfyUI V3 API.
- **Fast Smoothing**: Vectorized majority filter for real-time region smoothing.
- **Multiple Output Modes**:
  - `colored`: Filled regions with palette colors and labels.
  - `outline`: Print-ready template with black outlines and labels on a white background.
  - `quantized`: Preview of the color-reduced image.
- **Auto Color Detection**: Uses KneeLocator to find the optimal number of colors if not specified.
- **SVG Export**: Outputs raw SVG content for vector editing.

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/tazztone/pbn_node
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Find the node under `image/process -> Paint By Number`.

### Inputs

- **image**: The input image to process.
- **num_colors**: Number of clusters for color quantization (0 for auto-detection).
- **simplification**: Tolerance for contour simplification (0.5 - 2.0).
- **use_watershed**: Whether to use watershed segmentation (accurate but slower).
- **output_mode**: Choose between `colored`, `outline`, or `quantized`.

### Outputs

- **IMAGE**: The rendered PBN template.
- **SVG**: Raw SVG string content.
- **COLOR_COUNT**: Actual number of colors used in the palette.

## License

MIT
