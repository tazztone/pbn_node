import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from pbn_node.pbn_node import PaintByNumberNode


class TestPaintByNumberNode:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp

    def test_execute_returns_svg_preview(self, sample_image_tensor, temp_dir):
        # Mock folder_paths to use our temp directory
        with patch("folder_paths.get_temp_directory", return_value=temp_dir):
            output = PaintByNumberNode.execute(
                image=sample_image_tensor,
                num_colors=4,
                simplification=1.0,
                use_watershed=False,
                output_mode="colored",
            )

            # Check output structure
            assert hasattr(output, "ui")
            assert "pbn_svg" in output.ui
            assert "images" in output.ui

            # Check SVG references
            svg_refs = output.ui["pbn_svg"]
            assert len(svg_refs) == 1
            filename = svg_refs[0]["filename"]
            assert filename.startswith("pbn_")
            assert filename.endswith(".svg")

            # Check if file exists in temp dir
            filepath = os.path.join(temp_dir, filename)
            assert os.path.exists(filepath)

            # Check content (should be a string starting with <svg or <?xml)
            with open(filepath) as f:
                content = f.read()
                assert content.startswith("<?xml") or content.startswith("<svg")

    def test_svg_filename_determinism(self, sample_image_tensor, temp_dir):
        # Mock folder_paths
        with patch("folder_paths.get_temp_directory", return_value=temp_dir):
            # We need to patch the ImageProcessor constructor to return a mock instance
            # with specific attributes needed for rendering and hashing test.
            with patch("pbn_node.pbn_node.ImageProcessor") as mock_proc_cls:
                from pbn_node.backend.models import ColorPalette, LabelData, SVGResult

                mock_proc_inst = mock_proc_cls.return_value
                mock_proc_inst.process_array.return_value = SVGResult(
                    svg_content="<svg>deterministic test</svg>",
                    color_palette=ColorPalette(
                        colors=np.zeros((1, 3)), hex_colors=["#000000"], color_count=1
                    ),
                    processing_time=0.1,
                    region_count=1,
                    label_count=1,
                )
                # Set required attributes for the renderer
                mock_proc_inst.last_cleaned_regions = {}
                mock_proc_inst.last_label_data = LabelData(
                    positions={}, font_sizes={}, skipped_regions=set()
                )
                mock_proc_inst.last_palette = ColorPalette(
                    colors=np.zeros((1, 3)), hex_colors=["#000000"], color_count=1
                )
                mock_proc_inst.last_quantized = np.zeros((128, 128, 3), dtype=np.uint8)

                # Execute twice with same input
                out1 = PaintByNumberNode.execute(
                    image=sample_image_tensor,
                    num_colors=4,
                    simplification=1.0,
                    use_watershed=False,
                    output_mode="colored",
                )
                out2 = PaintByNumberNode.execute(
                    image=sample_image_tensor,
                    num_colors=4,
                    simplification=1.0,
                    use_watershed=False,
                    output_mode="colored",
                )

                fname1 = out1.ui["pbn_svg"][0]["filename"]
                fname2 = out2.ui["pbn_svg"][0]["filename"]

                # Should be exactly the same due to hashing
                assert fname1 == fname2
                # Hash of "<svg>deterministic test</svg>"
                assert fname1 == "pbn_87c57ef9264f7600.svg"

            # Verify only one file exists despite two "writes"
            files = os.listdir(temp_dir)
            svg_files = [f for f in files if f.startswith("pbn_") and f.endswith(".svg")]
            assert len(svg_files) == 1
