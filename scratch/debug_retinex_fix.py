import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def multiscale_retinex(
    image_bgr: np.ndarray,
    scales=(15, 80, 250),
) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    l_channel = lab[:, :, 0]
    l_channel = np.maximum(l_channel, 1.0)

    retinex_log = np.zeros_like(l_channel)
    for sigma in scales:
        blur = gaussian_filter(l_channel, sigma=sigma)
        blur = np.maximum(blur, 1.0)
        retinex_log += np.log(l_channel) - np.log(blur)

    retinex_log /= len(scales)

    # Linearize
    albedo_lin = np.exp(retinex_log)

    # Restore original mean luminance
    target_mean = np.mean(l_channel)
    current_mean = np.mean(albedo_lin)
    albedo_lin *= target_mean / current_mean

    # Clamp to [0, 255]
    albedo_lin = np.clip(albedo_lin, 0, 255)

    lab[:, :, 0] = albedo_lin
    lab[:, :, 1:] = np.clip(lab[:, :, 1:], 0, 255)
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


def test_chroma():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = 200  # Bright Blue
    image[:, :, 1] = 50
    image[:, :, 2] = 50

    albedo = multiscale_retinex(image)

    lab_in = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_out = cv2.cvtColor(albedo, cv2.COLOR_BGR2LAB)

    shift_a = np.mean(np.abs(lab_in[:, :, 1].astype(np.int16) - lab_out[:, :, 1].astype(np.int16)))
    shift_b = np.mean(np.abs(lab_in[:, :, 2].astype(np.int16) - lab_out[:, :, 2].astype(np.int16)))

    print(f"Shift A: {shift_a}, Shift B: {shift_b}")
    if shift_a < 10 and shift_b < 10:
        print("CHROMA SUCCESS")
    else:
        print("CHROMA FAILURE")


if __name__ == "__main__":
    test_chroma()
