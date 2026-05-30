import pytest
from pbn_node.backend.models import ProcessingParameters


@pytest.mark.unit
def test_processing_parameters_simplification_validation():
    # Test valid simplification
    ProcessingParameters(simplification=0.5)
    ProcessingParameters(simplification=1.0)

    # Test invalid simplification
    with pytest.raises(ValueError, match="Simplification must be at least 0.5"):
        ProcessingParameters(simplification=0.49)


@pytest.mark.unit
def test_processing_parameters_ciede2000_merge_thresh_validation():
    # Test valid ciede2000_merge_thresh
    ProcessingParameters(ciede2000_merge_thresh=2.0)
    ProcessingParameters(ciede2000_merge_thresh=10.0)
    ProcessingParameters(ciede2000_merge_thresh=20.0)

    # Test invalid ciede2000_merge_thresh
    with pytest.raises(ValueError, match="ciede2000_merge_thresh must be between 2.0 and 20.0"):
        ProcessingParameters(ciede2000_merge_thresh=1.99)
    with pytest.raises(ValueError, match="ciede2000_merge_thresh must be between 2.0 and 20.0"):
        ProcessingParameters(ciede2000_merge_thresh=20.01)


@pytest.mark.unit
def test_processing_parameters_min_region_width_validation():
    # Test valid min_region_width
    ProcessingParameters(min_region_width=2)
    ProcessingParameters(min_region_width=5)
    ProcessingParameters(min_region_width=20)

    # Test invalid min_region_width
    with pytest.raises(ValueError, match="min_region_width must be between 2 and 20"):
        ProcessingParameters(min_region_width=1)
    with pytest.raises(ValueError, match="min_region_width must be between 2 and 20"):
        ProcessingParameters(min_region_width=21)


@pytest.mark.unit
def test_processing_parameters_num_colors_validation():
    # Test valid num_colors
    ProcessingParameters(num_colors=None)
    ProcessingParameters(num_colors=2)
    ProcessingParameters(num_colors=10)

    # Test invalid num_colors
    with pytest.raises(ValueError, match="num_colors must be at least 2 if provided"):
        ProcessingParameters(num_colors=1)
    with pytest.raises(ValueError, match="num_colors must be at least 2 if provided"):
        ProcessingParameters(num_colors=0)
    with pytest.raises(ValueError, match="num_colors must be at least 2 if provided"):
        ProcessingParameters(num_colors=-1)
