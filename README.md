# CIFAR-10 ResNet Compression

Abubakar Bilal | 22I-0450 | Section A

A reproducible study of quantization, pruning, and CPU deployment for image classification.
Prepared for Edge Computing and Intelligent Systems, Assignment 3. It trains a CIFAR-10
ResNet-18 and produces FP32, PTQ INT8, QAT INT8, unstructured-pruned FP32, and
physically channel-pruned FP32 variants. Source code is Python.

## Installation

Training and the reported CPU benchmarks used Kaggle Python 3.13.16 with PyTorch 2.11.0 and torchvision 0.26.0. Windows inference was additionally verified with Python 3.12 and PyTorch 2.11.0 CPU. GPU training is
recommended. Kaggle already provides the matching CUDA packages. For an existing
compatible PyTorch environment, install the remaining dependencies with:

```text
python -m pip install -r requirements.txt
```

On a Windows CPU-only machine, install the CPU wheels first:

```text
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The quantized PyTorch backend is detected from the installed build, preferring
FBGEMM, x86, oneDNN, then QNNPACK. Linux Kaggle and Windows builds can provide
different engines; benchmark results identify the engine actually used. ONNX
Runtime inference uses CPUExecutionProvider. No local CUDA GPU is required for
inference. Training uses CUDA automatically when available.

## Run the supplied models

Run commands from the repository root. The example is official
CIFAR-10 test image zero, whose ground-truth label is cat (class 3).

```text
python src/infer.py --model models/baseline_fp32.pt --image examples/cifar10_test_0.png
python src/infer.py --model models/ptq_int8.pt --image examples/cifar10_test_0.png
python src/infer.py --model models/qat_int8.pt --image examples/cifar10_test_0.png
python src/infer.py --model models/unstructured_fp32.pt --image examples/cifar10_test_0.png
python src/infer.py --model models/structured_fp32.pt --image examples/cifar10_test_0.png
python src/infer_onnx.py --model models/qat_int8.onnx --image examples/cifar10_test_0.png
```

The unstructured model is supplied as an ordinary .pt checkpoint. Both INT8 .pt files are executable TorchScript
checkpoints. Other checkpoints store state dictionaries and architecture widths;
models.py reconstructs the executable network. Only load trusted checkpoints.

## Reproduce everything

```text
python src/run_all.py --data data --output reproduction/models --results reproduction/results
```

Defaults: seed 450, 50 baseline epochs, 6 QAT epochs, three unstructured pruning
rounds with 3 recovery epochs each, and 8 structured pruning recovery epochs.
Use --resume to skip stages with complete final artifacts. This is stage-level
recovery, not optimizer-exact recovery partway through an epoch. To change an
experiment, use a new output folder.

The first execution downloads the official CIFAR-10 Python archive and verifies
torchvision's checksums. The 50,000 training images are split into 45,000 fitting
images and 5,000 validation images with exactly 500 validation images per class.
Calibration uses a seeded shuffle of fitting images only. The official 10,000
test images are reserved for final evaluation. Split and calibration indices are
provided in results for auditing.

For independent stage execution:

```text
python src/train.py --data data --output artifacts --epochs 50
python src/calibrate.py --data data --baseline artifacts/baseline_fp32.pt --output artifacts --calibration 1024
python src/ptq.py --data data --baseline artifacts/baseline_fp32.pt --output artifacts
python src/qat.py --data data --baseline artifacts/baseline_fp32.pt --output artifacts --epochs 6
python src/prune_unstructured.py --data data --baseline artifacts/baseline_fp32.pt --output artifacts --epochs 3
python src/prune_structured.py --data data --baseline artifacts/baseline_fp32.pt --output artifacts --fraction 0.375 --epochs 8
python src/export_onnx.py --models artifacts
python src/verify_models.py --models artifacts --output verification
python src/benchmark.py --models artifacts --data data --output measured_results
```

calibrate.py and ptq.py are alternative entry points for the same complete PTQ
stage; running both is unnecessary. finetune.py provides another explicit entry
point: `python src/finetune.py qat|unstructured|structured [options]`.

## Optimization implementation

The CIFAR-10 adaptation keeps the ResNet-18 2-2-2-2 BasicBlock layout with stage
widths 64, 128, 256, and 512. It uses a 3x3 stride-one stem without max pooling,
global average pooling, and a 10-class linear head. Random padded crops and
horizontal flips augment training images. Fixed normalization is documented in
data.py; validation, calibration, and test data have no random augmentation.

PTQ fuses Conv-BN-ReLU and Conv-BN layers, inserts histogram observers, calibrates
on 1,024 shuffled training images, and converts convolution, linear, and residual
addition operations to quantized equivalents. Weights are symmetric INT8 per
output channel. Activations use affine UINT8 storage, with a reduced observer
range to avoid problematic accumulator saturation on some x86 kernels. Thus
"INT8" denotes 8-bit integer execution; activations are unsigned, weights signed.

QAT starts with the same baseline. It inserts activation and per-channel weight
fake quantizers, fine-tunes with SGD, freezes fused BN statistics from epoch 3,
and disables observers from epoch 4. Conversion removes the fake-quantization
training modules and produces a real quantized inference model.

Unstructured pruning globally ranks magnitudes across convolutional and linear
weights. Cumulative targets are 30%, 50%, and 70%. Masks remain active during
recovery so deleted weights do not regrow. prune.remove makes pruning permanent
after the last recovery stage. Dense parameter dimensions and dense MAC counts
remain unchanged. The original experiment measured GZIP storage savings; the repository instead supplies
an ordinary .pt checkpoint with identical weights. Speedup is measured rather
than inferred from the zero count.

Structured pruning ranks each BasicBlock's first convolution's output channels by
mean absolute filter magnitude and removes 37.5%. It slices corresponding BN
parameters and second-convolution input columns. Widths become 40, 80, 160, and
320 within the four stages. Block outputs and skip interfaces retain their
original widths. The reconstructed network has fewer physical parameters and
MACs and no channel masks.

## ONNX deployment

All five variants have ONNX exports. FP32 and pruning exports use PyTorch's
opset-17 exporter. Quantized exports explicitly unpack the converted PyTorch
weights and trained scales into standard QuantizeLinear/DequantizeLinear graphs.
This preserves the PTQ and QAT quantization parameters without recalibrating QAT
as PTQ. INT8 initializers are real integer tensors. ONNX Runtime optimizes
supported QDQ patterns for its CPU backend. Floating point input/output boundaries
and some graph operations remain; this is not a claim that every graph operation
is an integer instruction. Native and ONNX kernels can differ in rounding;
onnx_parity.csv measures their full-test-set agreement and logit differences.

## Results and benchmarking

All benchmark tables were measured on the Kaggle Linux CPU host with FBGEMM, using the same host for all ten comparisons. Additional Windows oneDNN checks establish model portability and are recorded separately.

results/benchmark_summary.csv contains test accuracy, file size, batch-one mean,
p50 and p99 latency, and throughput for every model under both runtimes. Timing
uses one CPU thread, 30 warmups per repeat, and 300 measured calls per repeat over
three repeats. Preprocessing, data loading, checkpoint loading and session setup
are outside the timed region. Throughput is measured loop calls divided by wall
time. RSS columns describe observed process memory after loading, not a measured
peak activation-memory requirement.

results/test_predictions.csv has one record per model, runtime, and test image:
100,000 records in total. It includes ground truth, prediction, confidence,
correctness, and evaluation-batch-amortized execution time. That last column is
not a separately timed batch-one latency; the batch-one measurements are in
latency_samples.csv and benchmark_summary.csv.

Training histories and optimization summaries record the settings and observed
validation recovery. architecture_layers.csv records physical weights and MACs.
model_verification.json records hashes, graph inspection, and loading checks.
The PDF report discusses the measured results, limitations, and screenshots.

GPU runs use cuDNN benchmarking and mixed precision for FP32/pruning training.
The seed and data splits are fixed, but different hardware or library kernels
can produce small numerical differences. Exact bitwise retraining is not claimed.

## Deliverables

Submit this repository link and the PDF report separately. No compressed folder
is needed or included in this repository.

- Repository: https://github.com/Abubakar4757/cifar10-resnet-compression
- Documentation: [PDF report](docs/CIFAR10_ResNet_Compression_22I-0450_A.pdf)
- Python implementation: src/
- Five trained checkpoints and five ONNX exports: models/
- Full predictions, timing samples, training histories and environment metadata: results/
- Execution evidence: screenshots/
- Requirement coverage: SUBMISSION_CHECKLIST.md

Datasets, environments, caches, installers, and executable programs are excluded.
The standard PyTorch and ONNX model files are supplied directly in models/.

## Measured CPU outcomes

| Variant | Test accuracy | Native p50 latency |
|---|---:|---:|
| FP32 baseline | 93.42% | 19.615 ms |
| PTQ INT8 | 93.30% | 9.349 ms |
| QAT INT8 | 93.22% | 9.273 ms |
| Unstructured 70% sparse | 93.23% | 19.894 ms |
| Physically channel-pruned | 92.98% | 14.559 ms |

All values above come from the saved full CIFAR-10 test experiment on the same
Kaggle Linux CPU host. They are not laptop or microcontroller measurements.

### Storage-format provenance

Original benchmark and verification files are retained unchanged as experiment
evidence. Their unstructured checkpoint filename, hash, and 17,022,848-byte size
describe the GZIP representation used during that run. For the revised submission,
the GZIP stream was decompressed byte-for-byte into models/unstructured_fp32.pt.
Weights, predictions, pruning, and measured inference latencies are unchanged.
The current ordinary checkpoint is approximately 44.8 MB; it does not claim the
historical compressed-file storage reduction. Current filenames, sizes, hashes,
and conversion verification are in results/repository_model_manifest.json.
SHA256SUMS.txt covers the current repository files. split_indices.npz is a NumPy
data array, not a submission folder archive.

## References

- PyTorch static quantization and QAT: https://pytorch.org/blog/introduction-to-quantization-on-pytorch/
- PyTorch pruning tutorial: https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html
- PyTorch quantization API: https://docs.pytorch.org/docs/main/quantization-support.html
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- ResNet paper: https://arxiv.org/abs/1512.03385

## Verification and report regeneration

The original experiment was audited against the assignment requirements. results/results_audit.json independently reconciles all 100,000 prediction rows and 9,000 latency samples with the summaries, verifies split isolation and calibration membership, and checks model hashes. results/windows_model_verification.json and windows_inference_verification.json record additional local loading and CLI checks.

```text
python src/test_pruning.py
python src/smoke_test.py
python src/make_report.py
```

The first two commands use synthetic inputs for implementation checks; their generated files are not experimental CIFAR-10 results. make_report.py regenerates an editable Word report from the supplied actual results and screenshots. It writes temporary figure assets into report_work/.
