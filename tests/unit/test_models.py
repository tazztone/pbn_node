import pytest
<<<<<<< HEAD
from pbn_node.backend.models import PerceptionInputs, ProcessingParameters


@pytest.mark.unit
class TestPerceptionInputs:
    def test_valid_inputs(self):
        """Test that valid inputs do not raise exceptions."""
        inputs = PerceptionInputs(
            material_weight=0.5,
            subject_priority=2.0,
            edge_influence=0.3,
            lineart_strength=0.7,
        )
        assert inputs.material_weight == 0.5
        assert inputs.subject_priority == 2.0
        assert inputs.edge_influence == 0.3
        assert inputs.lineart_strength == 0.7

    @pytest.mark.parametrize("invalid_weight", [-0.1, 1.1])
    def test_invalid_material_weight(self, invalid_weight):
        """Test that invalid material_weight raises ValueError."""
        with pytest.raises(ValueError, match="material_weight must be between 0.0 and 1.0"):
            PerceptionInputs(material_weight=invalid_weight)

    @pytest.mark.parametrize("invalid_priority", [0.9, 5.1])
    def test_invalid_subject_priority(self, invalid_priority):
        """Test that invalid subject_priority raises ValueError."""
        with pytest.raises(ValueError, match="subject_priority must be between 1.0 and 5.0"):
            PerceptionInputs(subject_priority=invalid_priority)

    @pytest.mark.parametrize("invalid_influence", [-0.1, 1.1])
    def test_invalid_edge_influence(self, invalid_influence):
        """Test that invalid edge_influence raises ValueError."""
        with pytest.raises(ValueError, match="edge_influence must be between 0.0 and 1.0"):
            PerceptionInputs(edge_influence=invalid_influence)

    @pytest.mark.parametrize("invalid_strength", [-0.1, 1.1])
    def test_invalid_lineart_strength(self, invalid_strength):
        """Test that invalid lineart_strength raises ValueError."""
        with pytest.raises(ValueError, match="lineart_strength must be between 0.0 and 1.0"):
            PerceptionInputs(lineart_strength=invalid_strength)


@pytest.mark.unit
class TestProcessingParameters:
    def test_valid_inputs(self):
        """Test that valid inputs do not raise exceptions."""
        params = ProcessingParameters(
            simplification=1.0,
            ciede2000_merge_thresh=10.0,
            min_region_width=5,
            num_colors=5,
        )
        assert params.simplification == 1.0
        assert params.ciede2000_merge_thresh == 10.0
        assert params.min_region_width == 5
        assert params.num_colors == 5

    def test_invalid_simplification(self):
        """Test that invalid simplification raises ValueError."""
        with pytest.raises(ValueError, match="Simplification must be at least 0.5"):
            ProcessingParameters(simplification=0.4)

    @pytest.mark.parametrize("invalid_thresh", [1.9, 20.1])
    def test_invalid_ciede2000_merge_thresh(self, invalid_thresh):
        """Test that invalid ciede2000_merge_thresh raises ValueError."""
        with pytest.raises(ValueError, match="ciede2000_merge_thresh must be between 2.0 and 20.0"):
            ProcessingParameters(ciede2000_merge_thresh=invalid_thresh)

    @pytest.mark.parametrize("invalid_width", [1, 21])
    def test_invalid_min_region_width(self, invalid_width):
        """Test that invalid min_region_width raises ValueError."""
        with pytest.raises(ValueError, match="min_region_width must be between 2 and 20"):
            ProcessingParameters(min_region_width=invalid_width)

    def test_invalid_num_colors(self):
        """Test that invalid num_colors raises ValueError."""
        with pytest.raises(ValueError, match="num_colors must be at least 2 if provided"):
            ProcessingParameters(num_colors=1)
=======
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
>>>>>>> origin/master
