import cv2
import numpy as np


class Protector:
    """
    Implements content-aware preprocessing using Mediapipe for face detection.
    Generates a weight map that oversamples detected regions during KMeans fitting.
    """

    def __init__(self):
        import mediapipe as mp

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def generate_protection_map(self, image: np.ndarray, weight_multiplier: float = 5.0) -> np.ndarray:
        """
        Detect faces in the image and generate a weight map.

        Args:
            image: Input image (BGR format)
            weight_multiplier: Factor by which to multiply weights of detected regions

        Returns:
            2D numpy array of shape (H, W) with pixel weights (base weight is 1.0)
        """
        h, w = image.shape[:2]
        weight_map = np.ones((h, w), dtype=np.float32)

        # Convert BGR to RGB for Mediapipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = self.face_detection.process(rgb_image)

        if not results.detections:
            return weight_map

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            # Convert relative coordinates to absolute pixels
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Apply padding (expand box by 20%)
            pad_w = int(width * 0.2)
            pad_h = int(height * 0.2)

            x_start = max(0, x_min - pad_w)
            y_start = max(0, y_min - pad_h)
            x_end = min(w, x_min + width + pad_w)
            y_end = min(h, y_min + height + pad_h)

            # Apply weight to detected region
            weight_map[y_start:y_end, x_start:x_end] = weight_multiplier

        # Apply slight blur to smooth transitions
        weight_map = cv2.GaussianBlur(weight_map, (15, 15), 0)

        return weight_map
