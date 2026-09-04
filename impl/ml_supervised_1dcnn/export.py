"""
Model export for edge and MCU deployment.

Exports a trained BearingFaultCNN1D checkpoint to:
    1. ONNX          — cross-platform edge deployment via ONNX Runtime
    2. TF Lite INT8  — microcontroller deployment (arXiv 2304.09100)

Also profiles CPU/GPU inference latency and validates ONNX output consistency.

Usage::

    python export.py --checkpoint ./runs/best_model.pt \\
        --output-dir ./exported \\
        --calibration-data /data/cwru
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import FAULT_CLASSES, NUM_CLASSES, BearingFaultCNN1D

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BearingFaultCNN1D to ONNX / TF Lite INT8")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", default="./exported", type=Path)
    parser.add_argument("--calibration-data", type=Path, default=None,
                        help="Dataset root for INT8 calibration (optional)")
    parser.add_argument("--n-warmup", default=50, type=int, help="Warmup iterations for latency profiling")
    parser.add_argument("--n-bench", default=500, type=int, help="Benchmark iterations")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size for profiling / export")
    parser.add_argument("--onnx-opset", default=17, type=int)
    parser.add_argument("--atol", default=1e-5, type=float, help="Absolute tolerance for ONNX validation")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> BearingFaultCNN1D:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    model = BearingFaultCNN1D(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    logger.info(
        "Loaded checkpoint from %s (epoch %d, val F1 %.4f)",
        checkpoint_path, ckpt.get("epoch", -1), ckpt.get("val_f1", float("nan")),
    )
    return model


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(
    model: BearingFaultCNN1D,
    output_path: Path,
    batch_size: int,
    opset: int,
) -> Path:
    """
    Export model to ONNX format with dynamic batch axis.

    Args:
        model:       Trained model in eval mode.
        output_path: Destination .onnx file path.
        batch_size:  Representative batch size for shape inference.
        opset:       ONNX opset version.

    Returns:
        Path to the written .onnx file.
    """
    dummy_input = torch.randn(batch_size, 1, 2048, device=next(model.parameters()).device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["fft_spectrum"],
        output_names=["class_probabilities"],
        dynamic_axes={
            "fft_spectrum": {0: "batch_size"},
            "class_probabilities": {0: "batch_size"},
        },
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )

    # Validate the ONNX model graph
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX export validated OK: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# ONNX validation against PyTorch
# ---------------------------------------------------------------------------

def validate_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    n_samples: int = 20,
    atol: float = 1e-5,
    device: torch.device = torch.device("cpu"),
) -> bool:
    """
    Run N random inputs through both PyTorch and ONNX Runtime; assert outputs match.

    Args:
        model:     PyTorch model (eval mode).
        onnx_path: Path to exported .onnx file.
        n_samples: Number of random test inputs.
        atol:      Absolute tolerance for output comparison.
        device:    Device for PyTorch inference.

    Returns:
        True if all outputs match within tolerance, raises AssertionError otherwise.
    """
    ort_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cuda"
        else ["CPUExecutionProvider"],
    )
    input_name = ort_session.get_inputs()[0].name

    max_diff = 0.0
    for i in range(n_samples):
        x_np = np.random.randn(1, 1, 2048).astype(np.float32)
        x_pt = torch.from_numpy(x_np).to(device)

        with torch.no_grad():
            pt_out = model(x_pt).cpu().numpy()

        ort_out = ort_session.run(None, {input_name: x_np})[0]
        diff = np.abs(pt_out - ort_out).max()
        max_diff = max(max_diff, diff)

    assert max_diff < atol, (
        f"ONNX output differs from PyTorch by {max_diff:.2e} (atol={atol})"
    )
    logger.info("ONNX validation passed — max output diff: %.2e (atol=%.2e)", max_diff, atol)
    return True


# ---------------------------------------------------------------------------
# TF Lite INT8 Export  (arXiv 2304.09100)
# ---------------------------------------------------------------------------

def export_tflite_int8(
    onnx_path: Path,
    output_path: Path,
    calibration_data: np.ndarray | None = None,
) -> Path:
    """
    Convert the ONNX model to TF Lite INT8 via the onnx-tf bridge.

    This produces a fully integer-quantised model suitable for deployment on
    Arm Cortex-M microcontrollers using TensorFlow Lite Micro. Post-training
    INT8 quantisation reduces model size by ~4x and speeds up inference ~2-3x
    on MCUs without dedicated float hardware.

    Reference: arXiv 2304.09100

    Args:
        onnx_path:         Path to the .onnx model.
        output_path:       Destination .tflite file path.
        calibration_data:  (N, 1, 2048) float32 representative dataset for
                           INT8 calibration. If None, uses random data
                           (accuracy may degrade slightly).

    Returns:
        Path to the written .tflite file.
    """
    try:
        import tensorflow as tf
        from onnx_tf.backend import prepare
    except ImportError as e:
        logger.error(
            "TF Lite export requires tensorflow and onnx-tf. "
            "Install with: pip install tensorflow onnx-tf. Error: %s", e
        )
        raise

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert ONNX → TF SavedModel
    saved_model_dir = output_path.with_suffix("") / "saved_model"
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(saved_model_dir))
    logger.info("ONNX → TF SavedModel: %s", saved_model_dir)

    # TF Lite INT8 quantisation
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    if calibration_data is None:
        logger.warning("No calibration data provided — using random data for INT8 quantisation.")
        calibration_data = np.random.randn(256, 1, 2048).astype(np.float32)

    # NHWC for TF: (N, L, 1) — transpose from (N, 1, L)
    cal_nhwc = calibration_data.transpose(0, 2, 1)

    def representative_dataset():
        for sample in cal_nhwc:
            yield [sample[np.newaxis].astype(np.float32)]

    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    logger.info("TF Lite INT8 model saved: %s (%.1f KB)", output_path, size_kb)
    return output_path


# ---------------------------------------------------------------------------
# Inference latency profiling
# ---------------------------------------------------------------------------

def profile_latency(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 1,
    n_warmup: int = 50,
    n_bench: int = 500,
) -> dict[str, float]:
    """
    Measure mean and P99 inference latency for the PyTorch model.

    Args:
        model:      Trained model (eval mode).
        device:     Target device.
        batch_size: Inference batch size.
        n_warmup:   Number of warm-up iterations (not timed).
        n_bench:    Number of timed iterations.

    Returns:
        dict with keys: mean_ms, std_ms, p50_ms, p99_ms, throughput_samples_per_s
    """
    model.eval()
    dummy = torch.randn(batch_size, 1, 2048, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_bench):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies_arr = np.array(latencies)
    results = {
        "mean_ms": float(latencies_arr.mean()),
        "std_ms": float(latencies_arr.std()),
        "p50_ms": float(np.percentile(latencies_arr, 50)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
        "throughput_samples_per_s": float(batch_size * 1000 / latencies_arr.mean()),
    }
    logger.info(
        "Latency [%s] batch=%d: mean=%.2fms p50=%.2fms p99=%.2fms throughput=%.0f samples/s",
        device, batch_size, results["mean_ms"], results["p50_ms"],
        results["p99_ms"], results["throughput_samples_per_s"],
    )
    return results


def profile_onnx_latency(
    onnx_path: Path,
    batch_size: int = 1,
    n_warmup: int = 50,
    n_bench: int = 500,
    use_gpu: bool = False,
) -> dict[str, float]:
    """Measure ONNX Runtime inference latency."""
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(batch_size, 1, 2048).astype(np.float32)

    for _ in range(n_warmup):
        sess.run(None, {input_name: dummy})

    latencies = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies_arr = np.array(latencies)
    results = {
        "mean_ms": float(latencies_arr.mean()),
        "std_ms": float(latencies_arr.std()),
        "p50_ms": float(np.percentile(latencies_arr, 50)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
        "throughput_samples_per_s": float(batch_size * 1000 / latencies_arr.mean()),
    }
    label = "ONNX+GPU" if use_gpu else "ONNX+CPU"
    logger.info(
        "Latency [%s] batch=%d: mean=%.2fms p50=%.2fms p99=%.2fms throughput=%.0f samples/s",
        label, batch_size, results["mean_ms"], results["p50_ms"],
        results["p99_ms"], results["throughput_samples_per_s"],
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda") if torch.cuda.is_available() else None

    # Load model
    model = load_model(args.checkpoint, cpu_device)

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------
    onnx_path = args.output_dir / "bearing_fault_cnn1d.onnx"
    export_onnx(model, onnx_path, batch_size=args.batch_size, opset=args.onnx_opset)
    validate_onnx(model, onnx_path, n_samples=20, atol=args.atol, device=cpu_device)

    # ------------------------------------------------------------------
    # Latency profiling — CPU (PyTorch)
    # ------------------------------------------------------------------
    logger.info("--- CPU latency (PyTorch) ---")
    profile_latency(model, cpu_device, batch_size=args.batch_size,
                    n_warmup=args.n_warmup, n_bench=args.n_bench)

    # ------------------------------------------------------------------
    # Latency profiling — GPU (PyTorch) if available
    # ------------------------------------------------------------------
    if gpu_device is not None:
        logger.info("--- GPU latency (PyTorch) ---")
        model_gpu = load_model(args.checkpoint, gpu_device)
        profile_latency(model_gpu, gpu_device, batch_size=args.batch_size,
                        n_warmup=args.n_warmup, n_bench=args.n_bench)

    # ------------------------------------------------------------------
    # Latency profiling — ONNX Runtime (CPU)
    # ------------------------------------------------------------------
    logger.info("--- CPU latency (ONNX Runtime) ---")
    profile_onnx_latency(onnx_path, batch_size=args.batch_size,
                         n_warmup=args.n_warmup, n_bench=args.n_bench, use_gpu=False)

    # ------------------------------------------------------------------
    # TF Lite INT8 export
    # ------------------------------------------------------------------
    calibration_data = None
    if args.calibration_data is not None:
        try:
            from dataset import CWRUBearingDataset
            ds = CWRUBearingDataset(data_root=args.calibration_data)
            indices = np.random.choice(len(ds), size=min(512, len(ds)), replace=False)
            calibration_data = np.stack([ds[int(i)][0].numpy() for i in indices])
            logger.info("Collected %d calibration samples from CWRU dataset", len(calibration_data))
        except Exception as exc:
            logger.warning("Could not load calibration data: %s — using random data", exc)

    tflite_path = args.output_dir / "bearing_fault_cnn1d_int8.tflite"
    try:
        export_tflite_int8(onnx_path, tflite_path, calibration_data=calibration_data)
    except ImportError:
        logger.warning(
            "Skipping TF Lite export — install tensorflow and onnx-tf to enable. "
            "See arXiv 2304.09100 for MCU deployment details."
        )

    logger.info("Export complete. Output directory: %s", args.output_dir)
    logger.info("  ONNX model:      %s", onnx_path)
    logger.info("  TF Lite INT8:    %s", tflite_path if tflite_path.exists() else "skipped")


if __name__ == "__main__":
    main()
