import pytest
from pbn_node.backend.preprocessing.preprocessor import Preprocessor


@pytest.mark.unit
def test_bilateral_filter(sample_image_np):
    """Test that bilateral_filter returns an array of the same shape and dtype."""
    preprocessor = Preprocessor()
    filtered = preprocessor.bilateral_filter(sample_image_np)

    assert filtered.shape == sample_image_np.shape
    assert filtered.dtype == sample_image_np.dtype


@pytest.mark.unit
def test_preprocess_output_shape(sample_image_np):
    preprocessor = Preprocessor()
    processed = preprocessor.preprocess(sample_image_np)

    assert processed.shape == sample_image_np.shape
    assert processed.dtype == sample_image_np.dtype
