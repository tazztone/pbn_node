import torch

from pbn_node.pbn_node import PaintByNumberNode


class TestPerceptionIntegration:
    def test_normals_integration(self, sample_image_tensor):
        # Create a dummy normal map (all pointing up/Z)
        # RGB [0.5, 0.5, 1.0] -> XYZ [0, 0, 1]
        normals = torch.ones_like(sample_image_tensor) * 0.5
        normals[..., 2] = 1.0

        output = PaintByNumberNode.execute(
            image=sample_image_tensor,
            num_colors=4,
            simplification=1.0,
            use_watershed=False,
            output_mode="colored",
            normals=normals,
            normal_strength=0.8,
        )

        assert output is not None
        assert isinstance(output[0], torch.Tensor)

    def test_lineart_integration(self, sample_image_tensor):
        # Create a dummy lineart (white background, black cross in middle)
        lineart = torch.ones((1, 128, 128, 1), dtype=torch.float32)
        lineart[0, 60:68, :] = 0.0
        lineart[0, :, 60:68] = 0.0

        output = PaintByNumberNode.execute(
            image=sample_image_tensor,
            num_colors=4,
            simplification=1.0,
            use_watershed=False,
            output_mode="colored",
            lineart=lineart,
            lineart_strength=0.9,
        )

        assert output is not None
        assert isinstance(output[0], torch.Tensor)

    def test_combined_perception(self, sample_image_tensor):
        # Albedo + Segmentation + Normals + Lineart
        segmentation = torch.zeros((1, 128, 128, 1), dtype=torch.float32)
        segmentation[0, 10:50, 10:50] = 1.0 / 255.0  # Class 1

        normals = torch.ones_like(sample_image_tensor) * 0.5
        lineart = torch.ones((1, 128, 128, 1), dtype=torch.float32)

        output = PaintByNumberNode.execute(
            image=sample_image_tensor,
            num_colors=8,
            simplification=1.0,
            use_watershed=False,
            output_mode="colored",
            segmentation=segmentation,
            normals=normals,
            lineart=lineart,
            use_content_protect=True,
        )

        assert output is not None
        assert output[2] > 0  # color_count
