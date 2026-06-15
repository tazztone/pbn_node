import pytest
from pbn_node.backend.models import ProcessingParameters, PerceptionInputs


@pytest.mark.unit
def test_perception_inputs_material_weight_validation():
    # Test valid material_weight
    PerceptionInputs(material_weight=0.0)
    PerceptionInputs(material_weight=0.5)
    PerceptionInputs(material_weight=1.0)

    # Test invalid material_weight
    with pytest.raises(ValueError, match="material_weight must be between 0.0 and 1.0"):
        PerceptionInputs(material_weight=-0.01)
    with pytest.raises(ValueError, match="material_weight must be between 0.0 and 1.0"):
        PerceptionInputs(material_weight=1.01)

@pytest.mark.unit
def test_perception_inputs_subject_priority_validation():
    # Test valid subject_priority
    PerceptionInputs(subject_priority=1.0)
    PerceptionInputs(subject_priority=2.5)
    PerceptionInputs(subject_priority=5.0)

    # Test invalid subject_priority
    with pytest.raises(ValueError, match="subject_priority must be between 1.0 and 5.0"):
        PerceptionInputs(subject_priority=0.99)
    with pytest.raises(ValueError, match="subject_priority must be between 1.0 and 5.0"):
        PerceptionInputs(subject_priority=5.01)

@pytest.mark.unit
def test_perception_inputs_edge_influence_validation():
    # Test valid edge_influence
    PerceptionInputs(edge_influence=0.0)
    PerceptionInputs(edge_influence=0.5)
    PerceptionInputs(edge_influence=1.0)

    # Test invalid edge_influence
    with pytest.raises(ValueError, match="edge_influence must be between 0.0 and 1.0"):
        PerceptionInputs(edge_influence=-0.01)
    with pytest.raises(ValueError, match="edge_influence must be between 0.0 and 1.0"):
        PerceptionInputs(edge_influence=1.01)

@pytest.mark.unit
def test_perception_inputs_lineart_strength_validation():
    # Test valid lineart_strength
    PerceptionInputs(lineart_strength=0.0)
    PerceptionInputs(lineart_strength=0.5)
    PerceptionInputs(lineart_strength=1.0)

    # Test invalid lineart_strength
    with pytest.raises(ValueError, match="lineart_strength must be between 0.0 and 1.0"):
        PerceptionInputs(lineart_strength=-0.01)
    with pytest.raises(ValueError, match="lineart_strength must be between 0.0 and 1.0"):
        PerceptionInputs(lineart_strength=1.01)

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
    with pytest.raises(
        ValueError, match="ciede2000_merge_thresh must be between 2.0 and 20.0"
    ):
        ProcessingParameters(ciede2000_merge_thresh=1.99)
    with pytest.raises(
        ValueError, match="ciede2000_merge_thresh must be between 2.0 and 20.0"
    ):
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
