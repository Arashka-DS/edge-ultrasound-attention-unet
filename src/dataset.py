import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from src.postprocessing import UltrasoundPreprocessor

class TemporalUltrasoundDataset(Dataset):
    def __init__(self, num_samples: int = 160, img_size: int = 256):
        self.num_samples = num_samples
        self.img_size = img_size
        self.preprocessor = UltrasoundPreprocessor(clip_limit=2.0)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        frames = []
        has_contact = 1.0 if np.random.rand() > 0.15 else 0.0
        has_vessel = 1.0 if (has_contact and np.random.rand() > 0.20) else 0.0

        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        cx, cy = self.img_size // 2, self.img_size // 2
        rx, ry = 28, 20

        for t in range(3):
            if not has_contact:
                frame = np.random.normal(20, 10, (self.img_size, self.img_size))
            else:
                speckle = np.random.rayleigh(scale=85, size=(self.img_size, self.img_size))
                frame = speckle
                if has_vessel:
                    dynamic_rx = rx + int((t - 1) * 2)
                    dynamic_ry = ry + int((t - 1) * 2)
                    y, x = np.ogrid[:self.img_size, :self.img_size]
                    lumen = ((x - cx) ** 2) / (dynamic_rx ** 2) + ((y - cy) ** 2) / (dynamic_ry ** 2) <= 1.0
                    frame[lumen] = frame[lumen] * 0.25 # Dark anechoic fluid
                    
                    if t == 2:
                        mask[lumen] = 1.0

            # Convert to uint8 and pass through the EXACT same CLAHE pipeline as inference
            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            norm_frame, _ = self.preprocessor.process_frame(frame_uint8, target_size=self.img_size)
            frames.append(norm_frame)

        input_tensor = torch.from_numpy(np.stack(frames, axis=0))
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image": input_tensor,
            "mask": mask_tensor,
            "contact_label": torch.tensor([has_contact], dtype=torch.float32),
            "vessel_label": torch.tensor([has_vessel], dtype=torch.float32),
        }
