import pytest
import torch

from pbn_node.pbn_node import PaintByNumberNode


@pytest.mark.integration
def test_pbn_node_execute(sample_image_tensor):
    # sample_image_tensor is [1, 128, 128, 3]
    node = PaintByNumberNode()

    # Run execute
    result = node.execute(
        image=sample_image_tensor,
        num_colors=5,
        simplification=1.0,
        use_watershed=False,
        output_mode="colored",
    )

    # Verify outputs
    # io.NodeOutput is mocked as a tuple-like object in conftest
    image_out, svg_out, count_out = result

    assert isinstance(image_out, torch.Tensor)
    assert image_out.shape == sample_image_tensor.shape
    assert isinstance(svg_out, list)
    assert isinstance(svg_out[0], str)
    assert "<svg" in svg_out[0]
    assert isinstance(count_out, list)
    assert isinstance(count_out[0], int)
    assert count_out[0] > 0


@pytest.mark.integration
def test_pbn_node_output_modes(sample_image_tensor):
    node = PaintByNumberNode()

    for mode in ["colored", "outline", "quantized"]:
        result = node.execute(
            image=sample_image_tensor,
            num_colors=5,
            simplification=1.0,
            use_watershed=False,
            output_mode=mode,
        )
        assert result[0] is not None
