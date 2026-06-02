import cv2
import numpy as np
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


@pytest.mark.unit
def test_histogram_equalization_output_shape(sample_image_np):
    """Test that histogram_equalization maintains shape and dtype."""
    preprocessor = Preprocessor()
    processed = preprocessor.histogram_equalization(sample_image_np)

    assert processed.shape == sample_image_np.shape
    assert processed.dtype == sample_image_np.dtype


@pytest.mark.unit
def test_histogram_equalization_invalid_channels():
    """Test that histogram_equalization fails gracefully on single channel images."""
    preprocessor = Preprocessor()
    # Create a 1-channel image instead of 3-channel
    grayscale_img = np.zeros((128, 128), dtype=np.uint8)

    # OpenCV cvtColor should fail since it expects BGR for BGR2LAB conversion
    with pytest.raises(cv2.error):
        preprocessor.histogram_equalization(grayscale_img)
