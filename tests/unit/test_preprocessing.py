import pytest
import numpy as np
from pbn_node.backend.preprocessing.preprocessor import Preprocessor

@pytest.mark.unit
def test_detect_image_type():
    preprocessor = Preprocessor()
    
    # Landscape
    img_landscape = np.zeros((100, 200, 3), dtype=np.uint8)
    assert preprocessor.detect_image_type(img_landscape) == "landscape"
    
    # Portrait
    img_portrait = np.zeros((200, 100, 3), dtype=np.uint8)
    assert preprocessor.detect_image_type(img_portrait) == "portrait"

@pytest.mark.unit
def test_preprocess_metadata(sample_image_np):
    preprocessor = Preprocessor()
    _, metadata = preprocessor.preprocess(sample_image_np)
    
    assert metadata.width == 128
    assert metadata.height == 128
    assert metadata.channels == 3
    assert metadata.image_type == "landscape" # 128x128 is landscape in this logic

@pytest.mark.unit
def test_preprocess_output_shape(sample_image_np):
    preprocessor = Preprocessor()
    processed, _ = preprocessor.preprocess(sample_image_np)
    
    assert processed.shape == sample_image_np.shape
    assert processed.dtype == sample_image_np.dtype
