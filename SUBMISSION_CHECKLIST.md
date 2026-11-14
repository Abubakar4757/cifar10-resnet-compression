# Assignment 3 submission checklist

Abubakar Bilal | 22I-0450 | Section A

All paths below are relative to the repository root. This package
contains Assignment 3 only.

| Requirement | Submitted evidence |
|---|---|
| Q1 trained CIFAR-10 baseline | models/baseline_fp32.pt; src/train.py; src/models.py; results/baseline_history.csv |
| PTQ compatible layer fusion | src/ptq.py invokes ResNet18.fuse in src/models.py |
| Quantization backend configuration | src/engine.py initialize; results/ptq_summary.json and benchmark_environment.json |
| Training-only PTQ calibration | src/calibrate.py; src/ptq.py; results/calibration_indices.npy; results/ptq_summary.json |
| Real PTQ INT8 conversion and inference | models/ptq_int8.pt; models/ptq_int8.onnx; src/infer.py |
| QAT starts with pretrained FP32 weights | src/qat.py loads the same baseline_fp32.pt |
| Fake quantization insertion | src/qat.py prepare_qat; src/engine.py quant_config; results/qat_summary.json records 51 fake-quant modules |
| QAT fine-tuning and INT8 conversion | src/qat.py; src/finetune.py; results/qat_history.csv; models/qat_int8.pt |
| Three executable Q1 checkpoints | baseline_fp32.pt, ptq_int8.pt, qat_int8.pt; loading/reconstruction in src/engine.py and models.py |
| Separate PTQ and QAT implementation scripts | src/ptq.py and src/qat.py contain their respective optimization implementations |
| Q1 automated inference and benchmarking | src/infer.py; src/infer_onnx.py; src/benchmark.py |
| Test predictions and performance CSV | results/test_predictions.csv; benchmark_summary.csv; latency_samples.csv |
| Reproducible execution pipeline | src/run_all.py; requirements.txt; README.md |
| Q2 pretrained CIFAR-10 ResNet-18 | The common baseline_fp32.pt and src/models.py |
| Magnitude pruning of convolutional and FC weights | src/prune_unstructured.py uses global L1 pruning over Conv2d and Linear |
| Iterative pruning with multiple rounds | Three cumulative targets of 30, 50, and 70 percent; results/unstructured_summary.json and three recovery histories |
| Weight masking during recovery | src/prune_unstructured.py retains masks throughout fine-tuning |
| Permanent unstructured pruning | prune.remove after recovery; model verification checks absence of masks and weight_orig |
| Permanent sparse model | models/unstructured_fp32.pt; ordinary checkpoint reconstruction in src/engine.py; original GZIP size retained as historical evidence only |
| Structured channel selection | Mean absolute conv1 filter magnitude in src/models.py structured_prune |
| Valid connected tensor dimensions | Matching conv1 output rows, BN1 entries and conv2 input columns; skip interfaces retained |
| Physical architecture reconstruction | models/structured_fp32.pt has widths 40, 40, 80, 80, 160, 160, 320, 320; no masks |
| Recovery fine-tuning for both pruning methods | src/prune_unstructured.py; src/prune_structured.py; src/finetune.py; corresponding history CSVs |
| Deploy both pruning variants | Both native checkpoints and both ONNX exports are supplied |
| Benchmark original and both pruned models | All five variants are evaluated under both runtimes in benchmark_summary.csv |
| Inference for all five models | src/infer.py and src/infer_onnx.py; supplied example image; ten verified CLI executions |
| ONNX deployment and CPU scripts | Five models/*.onnx files; src/export_onnx.py; src/infer_onnx.py |
| Technical report with details, results and screenshots | docs/CIFAR10_ResNet_Compression_22I-0450_A.pdf; screenshots/ |
| Installation and execution documentation | README.md; requirements.txt |
| Revised submission format | New GitHub repository plus standalone PDF; no folder archives |

## Verification and provenance

- The original complete Kaggle experiment and its full-test predictions are preserved.
- All five checkpoints and ONNX models were previously reloaded and executed.
- Train/validation separation, calibration isolation, 100,000 prediction records,
  and 9,000 latency samples were independently audited.
- The unstructured checkpoint has been decompressed without changing the serialized
  payload or trained weights. Current hashes are recorded separately from the original
  experiment metadata.
- Source entry points now produce and use the ordinary unstructured .pt checkpoint.
- The PDF is the submission documentation; its pages are visually checked.
- GitHub repository contains source code, trained models, and experimental results.
- Submit the repository URL and PDF separately, following the revised announcement.

Original experiment verification JSON files describe the original artifacts.
results/repository_model_manifest.json and SHA256SUMS.txt identify current files.
No Classroom submission is performed by this workflow.
