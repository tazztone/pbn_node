"""
Maps Sapiens body-part class IDs to PBN priority weights.
Sapiens uses a fixed 28-class vocabulary.
"""

import numpy as np

# Sapiens body part class → priority weight (1.0 = neutral)
# Face and hands get highest priority (more colors, less merge)
# Background gets lowest (heavy simplification)
SAPIENS_PRIORITY = {
    0: 0.2,  # background
    1: 3.0,  # face skin
    2: 3.0,  # left eye
    3: 3.0,  # right eye
    4: 2.5,  # nose
    5: 2.5,  # mouth
    6: 2.0,  # left ear
    7: 2.0,  # right ear
    8: 2.0,  # neck
    9: 1.5,  # hair
    10: 1.0,  # upper clothing
    11: 1.0,  # lower clothing
    12: 2.5,  # left hand
    13: 2.5,  # right hand
    14: 1.0,  # left arm
    15: 1.0,  # right arm
    16: 1.0,  # left leg
    17: 1.0,  # right leg
    18: 0.8,  # shoes
    19: 0.8,  # socks / accessories
    # Remaining classes default to 1.0
}


def build_priority_map(
    segmentation_np: np.ndarray,
    class_map: dict[int, float] | None = None,
) -> np.ndarray:
    """
    Convert a Sapiens class label map to a float priority weight map.

    Args:
        segmentation_np: [H, W] int32 class labels
        class_map: override mapping (uses SAPIENS_PRIORITY by default)

    Returns:
        [H, W] float32 priority weights
    """
    mapping = class_map or SAPIENS_PRIORITY
    priority = np.ones(segmentation_np.shape, dtype=np.float32)
    for class_id, weight in mapping.items():
        priority[segmentation_np == class_id] = weight
    return priority
