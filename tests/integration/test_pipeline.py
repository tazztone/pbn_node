import pytest

from pbn_node.backend.models import ProcessingParameters
from pbn_node.pbn_pipeline import ImageProcessor


@pytest.mark.integration
def test_image_processor_full_run(sample_image_np):
    processor = ImageProcessor()
    params = ProcessingParameters(num_colors=5, simplification=1.0, use_watershed=False)

    result = processor.process_array(sample_image_np, params)

    assert "<svg" in result.svg_content
    assert result.color_palette.color_count <= 5
    assert result.region_count > 0
    assert result.processing_time > 0

    # Check that all required data is in the result
    assert result.color_palette is not None
    assert result.cleaned_regions is not None
    assert result.label_data is not None
    assert result.quantized is not None
    assert result.region_colors and len(result.region_colors) == result.region_count


@pytest.mark.integration
def test_image_processor_batch_simulation(sample_image_np):
    """Ensure processor can be reused for multiple images."""
    processor = ImageProcessor()
    params = ProcessingParameters(num_colors=4)

    # Process twice
    res1 = processor.process_array(sample_image_np, params)
    res2 = processor.process_array(sample_image_np, params)

    assert res1.svg_content is not None
    assert res2.svg_content is not None
    assert res1.region_count == res2.region_count
