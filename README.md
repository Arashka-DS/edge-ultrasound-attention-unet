# Vascular-Edge-UNet: Real-Time Ultrasound Segmentation

A low-latency, edge-deployed computer vision pipeline designed for handheld point-of-care ultrasound (POCUS) devices. This system identifies vascular structures and anatomical boundaries in real-time to assist clinicians with peripheral vascular access and injection safety, processing directly on low-power mobile or cart-based hardware.

## 🚀 Architectural Foundations
1. **Multi-Head MobileNetV3 Attention U-Net:** A custom multi-task architecture built over a lightweight backbone. Alongside the pixel-level Attention U-Net decoder, two classification heads predict **Acoustic Probe Contact** and **Target Vessel Presence**.
2. **Conditional Compute Path:** If acoustic contact is lost or no target anatomy is detected in the cross-section plane, the segmentation mask rendering is bypassed. This saves up to 70% of compute load and extends battery life on portable tablets.
3. **Temporal Rolling Stack & Anti-Flicker EMA:** Ingests a temporal stack of three consecutive frames $[t-2, t-1, t]$ through a 3-channel input layer. An Exponential Moving Average (EMA) postprocessing filter stabilizes mask jitter across frames to counteract acoustic speckle noise.
4. **Hardware-Level CPU Quantization:** Exported to static ONNX graphs and quantized dynamically to INT8, achieving sub-15ms inference latencies on bare-metal Intel/AMD/ARM CPUs without requiring a dedicated GPU.
5. **Bare-Metal Multithreading:** Designed without containerization overhead to maximize edge performance. A Producer-Consumer queue splits video capture and OpenCV CLAHE preprocessing from the ONNX inference worker.

## ⚙️ Quick Start

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Model Training & Validation
Trains the multi-task model on synthetic speckle-noise ultrasound sequences:
```bash
python -m src.train
```

### 3. ONNX INT8 Compilation
Exports the PyTorch checkpoint to ONNX and quantizes the graph to INT8:
```bash
python -m src.export_onnx
```

### 4. Real-Time Stream Execution
Launches the multithreaded OpenCV inference stream:
```bash
python -m src.edge_stream_engine
```

*(If no physical USB ultrasound probe is detected, the engine automatically falls back to an internal synthetic ultrasound generator for validation.)*
