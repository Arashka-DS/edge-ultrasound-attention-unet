import cv2
import numpy as np


class TemporalMaskStabilizer:
    """Stabilizes edge flickering across frames using Exponential Moving Average (EMA)."""

    def __init__(self, alpha: float = 0.65):
        self.alpha = alpha
        self.smoothed_mask = None

    def update(self, current_mask: np.ndarray) -> np.ndarray:
        if self.smoothed_mask is None:
            self.smoothed_mask = current_mask.astype(np.float32)
        else:
            self.smoothed_mask = (self.alpha * self.smoothed_mask) + (
                (1.0 - self.alpha) * current_mask.astype(np.float32)
            )
        return (self.smoothed_mask > 0.5).astype(np.uint8)

    def reset(self):
        self.smoothed_mask = None


class UltrasoundPreprocessor:
    """Preprocesses raw ultrasound frames with Contrast-Limited Adaptive Histogram Equalization."""

    def __init__(self, clip_limit: float = 2.0, tile_size: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    def process_frame(self, frame_bgr: np.ndarray, target_size: int = 256) -> tuple:
        # Convert to grayscale
        if len(frame_bgr.shape) == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr

        # Apply CLAHE to resolve hypoechoic tissue walls
        enhanced = self.clahe.apply(gray)

        # Bilateral filter suppresses acoustic speckle while preserving sharp vessel boundaries
        filtered = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35, sigmaSpace=35)

        # Resize for model input
        resized = cv2.resize(filtered, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0

        return normalized, enhanced


def overlay_mask(base_image_bgr: np.ndarray, binary_mask: np.ndarray, color=(0, 255, 0), alpha=0.45) -> np.ndarray:
    """Alpha-blends the predicted vascular segmentation mask directly over the live feed."""
    h, w = base_image_bgr.shape[:2]
    mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    colored_overlay = base_image_bgr.copy()
    colored_overlay[mask_resized > 0] = color

    return cv2.addWeighted(colored_overlay, alpha, base_image_bgr, 1.0 - alpha, 0)
