import pytest
from pbn_node.backend.preprocessing.preprocessor import Preprocessor


@pytest.mark.unit
def test_preprocess_output_shape(sample_image_np):
    preprocessor = Preprocessor()
    processed = preprocessor.preprocess(sample_image_np)

    assert processed.shape == sample_image_np.shape
    assert processed.dtype == sample_image_np.dtype
