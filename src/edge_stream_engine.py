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
    # Change default source to None to bypass hardware camera checks completely
    def __init__(self, model_path: str = "models/vascular_unet_int8.onnx", source=None):
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

        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.preprocessor = UltrasoundPreprocessor()
        self.stabilizer = TemporalMaskStabilizer(alpha=0.7)

    def capture_worker(self):
        """Thread 1: Ingests frames and buffers the 3-frame temporal window."""
        # Only initialize VideoCapture if a physical source is explicitly requested
        cap = cv2.VideoCapture(self.source) if self.source is not None else None
        temporal_buffer = collections.deque(maxlen=3)
        t_step = 0

        while not self.stopped:
            ret = False
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                
            if not ret:
                # Synthetic fallback loop matching the EXACT scale=85 training distribution
                gray = np.random.rayleigh(scale=85, size=(480, 640))
                cx, cy = 320, 240
                
                # Simulate cardiac pulsation using a sine wave
                rx = 45 + int(8 * np.sin(t_step * 0.3))
                ry = 35 + int(8 * np.sin(t_step * 0.3))
                
                y, x = np.ogrid[:480, :640]
                lumen = ((x - cx)**2) / (rx**2) + ((y - cy)**2) / (ry**2) <= 1.0
                
                # Match the 0.25 anechoic fluid darkness from dataset.py
                gray[lumen] = gray[lumen] * 0.25
                
                frame = np.clip(gray, 0, 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                t_step += 1
                time.sleep(0.04)  # Enforce ~25 FPS frame rate limit

            norm_frame, enhanced = self.preprocessor.process_frame(frame, target_size=256)
            temporal_buffer.append(norm_frame)

            if len(temporal_buffer) == 3:
                stacked_tensor = np.stack(list(temporal_buffer), axis=0)[np.newaxis, ...]
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, stacked_tensor))

        if cap is not None:
            cap.release()

    def inference_worker(self):
        """Thread 2: Executes conditional ONNX inference and postprocessing."""
        while not self.stopped:
            try:
                raw_frame, tensor = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t_start = time.perf_counter()
            outputs = self.session.run(None, {"temporal_input": tensor})
            contact_logit, vessel_logit, seg_logits = outputs[0][0][0], outputs[1][0][0], outputs[2][0][0]

            contact_prob = 1.0 / (1.0 + np.exp(-contact_logit))
            vessel_prob = 1.0 / (1.0 + np.exp(-vessel_logit))

            display_frame = raw_frame.copy()
            latency_ms = (time.perf_counter() - t_start) * 1000

            if contact_prob < 0.50:
                self.stabilizer.reset()
                cv2.putText(display_frame, "PROBE STATUS: NO ACOUSTIC CONTACT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif vessel_prob < 0.50:
                self.stabilizer.reset()
                cv2.putText(display_frame, "SCANNING: NO VESSEL IN PLANE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                raw_mask = (seg_logits > 0.0).astype(np.uint8)
                smoothed_mask = self.stabilizer.update(raw_mask)
                display_frame = overlay_mask(display_frame, smoothed_mask, color=(0, 255, 0), alpha=0.45)
                cv2.putText(display_frame, f"VESSEL DETECTED ({vessel_prob*100:.1f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display_frame, f"Latency: {latency_ms:.1f} ms | FPS: {1000/max(latency_ms, 1.0):.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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
    # Explicitly set source=None to default to the synthetic pulsatile stream
    engine = EdgeStreamEngine(model_path="models/vascular_unet_int8.onnx", source=None)
    engine.run()
