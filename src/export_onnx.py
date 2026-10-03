import os
import torch
import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, shape_inference, CalibrationDataReader
from src.model import VascularAttentionUNet
from src.dataset import TemporalUltrasoundDataset

class UltrasoundCalibrationReader(CalibrationDataReader):
    """Provides representative input data to calibrate static INT8 activation scales."""
    def __init__(self, batch_size=1, num_samples=16):
        # Generate exact representative ultrasound frames for hardware calibration
        dataset = TemporalUltrasoundDataset(num_samples=num_samples, img_size=256)
        self.data = iter([
            {"temporal_input": dataset[i]["image"].unsqueeze(0).numpy()}
            for i in range(num_samples)
        ])

    def get_next(self):
        return next(self.data, None)

def export_and_quantize():
    os.makedirs("models", exist_ok=True)
    pytorch_weights = "models/vascular_attention_unet.pth"
    fp32_onnx_path = "models/vascular_unet_fp32.onnx"
    prepped_onnx_path = "models/vascular_unet_prep.onnx"
    int8_onnx_path = "models/vascular_unet_int8.onnx"

    model = VascularAttentionUNet(pretrained=False)
    if os.path.exists(pytorch_weights):
        model.load_state_dict(torch.load(pytorch_weights, map_location="cpu"))
        print(f"Loaded weights from {pytorch_weights}")

    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    print("Exporting PyTorch model to ONNX FP32...")
    torch.onnx.export(
        model,
        dummy_input,
        fp32_onnx_path,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=["temporal_input"],
        output_names=["contact_logits", "vessel_logits", "seg_logits"],
    )

    print("Pre-processing ONNX graph for shape inference...")
    shape_inference.quant_pre_process(
        input_model_path=fp32_onnx_path,
        output_model_path=prepped_onnx_path,
        skip_symbolic_shape=False,
    )

    print("Performing INT8 Static Quantization for CNNs on bare-metal CPU...")
    calibrator = UltrasoundCalibrationReader()
    quantize_static(
        model_input=prepped_onnx_path,
        model_output=int8_onnx_path,
        calibration_data_reader=calibrator,
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )

    print(f"Static quantized edge model ready: {int8_onnx_path}")


if __name__ == "__main__":
    export_and_quantize()
