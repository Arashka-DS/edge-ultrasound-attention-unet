import os
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from src.model import VascularAttentionUNet


def export_and_quantize():
    os.makedirs("models", exist_ok=True)
    pytorch_weights = "models/vascular_attention_unet.pth"
    fp32_onnx_path = "models/vascular_unet_fp32.onnx"
    int8_onnx_path = "models/vascular_unet_int8.onnx"

    model = VascularAttentionUNet(pretrained=False)
    if os.path.exists(pytorch_weights):
        model.load_state_dict(torch.load(pytorch_weights, map_location="cpu"))
        print(f"Loaded weights from {pytorch_weights}")
    else:
        print("No weights found. Exporting base architecture...")

    model.eval()

    # Fixed spatial shape: [1, 3, 256, 256] guarantees SIMD cache alignment on edge CPUs
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    print("Exporting PyTorch model to ONNX FP32...")
    torch.onnx.export(
        model,
        dummy_input,
        fp32_onnx_path,
        export_params=True,
        opset_version >= 18,
        do_constant_folding=True,
        input_names=["temporal_input"],
        output_names=["contact_logits", "vessel_logits", "seg_logits"],
    )

    # Validate ONNX model
    onnx_model = onnx.load(fp32_onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX FP32 export verified: {fp32_onnx_path}")

    # INT8 Dynamic Quantization for edge CPU execution
    print("Performing INT8 Dynamic Quantization for bare-metal CPU...")
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8,
    )

    fp32_size = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    print(f"FP32 Model Size: {fp32_size:.2f} MB")
    print(f"INT8 Model Size: {int8_size:.2f} MB")
    print(f"Quantized edge model ready: {int8_onnx_path}")


if __name__ == "__main__":
    export_and_quantize()
