import collections
import os
import queue
import threading
import time
import cv2
import numpy as np
import onnxruntime as ort
from src.postprocessing import UltrasoundPreprocessor, TemporalMaskStabilizer, overlay_mask

class EdgeStreamEngine:
    def __init__(self, model_path: str = "models/vascular_unet_fp32.onnx", source=None):
        self.source = source
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.stopped = False

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX Model not found at {model_path}. Run export_onnx.py first.")

        # Pointing to CPU FP32 to avoid software emulation latency
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.preprocessor = UltrasoundPreprocessor(clip_limit=2.0)
        self.stabilizer = TemporalMaskStabilizer(alpha=0.7)

    def capture_worker(self):
        """Thread 1: Ingests frames, auto-detects cameras, and buffers the temporal window."""
        cap = None
        
        # 1. Attempt explicit source if provided
        if self.source is not None:
            cap = cv2.VideoCapture(self.source)
        # 2. Attempt default system camera
        else:
            cap = cv2.VideoCapture(0)
            
        if cap is None or not cap.isOpened():
            print("No physical camera detected. Falling back to synthetic ultrasound simulator.")
            cap = None
            
        temporal_buffer = collections.deque(maxlen=3)
        t_step = 0

        while not self.stopped:
            ret = False
            if cap is not None:
                ret, frame = cap.read()

            if not ret:
                # Reverted to hard-edged boolean masks to perfectly match dataset.py distribution
                gray_256 = np.random.rayleigh(scale=85, size=(256, 256))
                cx, cy = 128, 128

                rx = 32 + int(6 * np.sin(t_step * 0.25))
                ry = 24 + int(6 * np.sin(t_step * 0.25))

                y, x = np.ogrid[:256, :256]
                lumen = ((x - cx)**2) / (rx**2) + ((y - cy)**2) / (ry**2) <= 1.0
                gray_256[lumen] = gray_256[lumen] * 0.25

                frame_uint8 = np.clip(gray_256, 0, 255).astype(np.uint8)
                norm_frame, _ = self.preprocessor.process_frame(frame_uint8, target_size=256)

                display_frame = cv2.resize(frame_uint8, (640, 480), interpolation=cv2.INTER_LINEAR)
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

                t_step += 1
                time.sleep(0.04)
            else:
                norm_frame, _ = self.preprocessor.process_frame(frame, target_size=256)
                display_frame = frame

            temporal_buffer.append(norm_frame)

            if len(temporal_buffer) == 3:
                stacked_tensor = np.stack(list(temporal_buffer), axis=0)[np.newaxis, ...]
                if not self.frame_queue.full():
                    self.frame_queue.put((display_frame, stacked_tensor))

        if cap is not None:
            cap.release()

    def inference_worker(self):
        """Thread 2: Executes ONNX inference."""
        while not self.stopped:
            try:
                raw_frame, tensor = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t_start = time.perf_counter()
            
            # Explicitly request output names to prevent ONNX graph alphabetical swapping
            outputs = self.session.run(
                ["contact_logits", "vessel_logits", "seg_logits"], 
                {"temporal_input": tensor}
            )
            contact_logit, vessel_logit, seg_logits = outputs[0][0][0], outputs[1][0][0], outputs[2][0][0]

            contact_prob = 1.0 / (1.0 + np.exp(-contact_logit))
            vessel_prob = 1.0 / (1.0 + np.exp(-vessel_logit))

            display_frame = raw_frame.copy()
            latency_ms = (time.perf_counter() - t_start) * 1000

            if contact_prob < 0.40:
                self.stabilizer.reset()
                cv2.putText(display_frame, f"PROBE STATUS: NO CONTACT ({contact_prob*100:.1f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            elif vessel_prob < 0.40:
                self.stabilizer.reset()
                cv2.putText(display_frame, f"SCANNING: NO VESSEL ({vessel_prob*100:.1f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            else:
                raw_mask = (seg_logits > 0.0).astype(np.uint8)
                smoothed_mask = self.stabilizer.update(raw_mask)
                display_frame = overlay_mask(display_frame, smoothed_mask, color=(0, 255, 0), alpha=0.45)
                cv2.putText(display_frame, f"VESSEL DETECTED ({vessel_prob*100:.1f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.putText(display_frame, f"Latency: {latency_ms:.1f} ms | FPS: {1000/max(latency_ms, 1.0):.1f} | Contact: {contact_prob*100:.0f}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if not self.result_queue.full():
                self.result_queue.put(display_frame)

    def run(self):
        t_cap = threading.Thread(target=self.capture_worker, daemon=True)
        t_inf = threading.Thread(target=self.inference_worker, daemon=True)
        t_cap.start()
        t_inf.start()
        print("Engine initialized. Press 'q' to terminate stream.")

        try:
            while True:
                try:
                    frame = self.result_queue.get(timeout=0.1)
                    cv2.imshow("Vascular-Edge-UNet Ultrasound Feed", frame)
                except queue.Empty:
                    pass

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.stopped = True
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Point explicitly to the FP32 model for pure CPU execution
    engine = EdgeStreamEngine(model_path="models/vascular_unet_fp32.onnx")
    engine.run()
