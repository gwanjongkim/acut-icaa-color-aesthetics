# Benchmark Summary — A-CUT ICAA Color Aesthetics (TF-Native DAT Port)

## Overview
- **Repository**: `acut-icaa-color-aesthetics`
- **Role in A-CUT**: Evaluated Color Aesthetics Specialist & TF-Native Engineering Port (Excluded from on-device active ensemble due to 167.7 MB footprint)
- **Task**: Image Color Aesthetics Assessment (ICAA17K) — Joint `[MOS, Color]` prediction

## Evaluated Metrics

| Dataset | Head / Metric | SRCC | PLCC | MAE | Parity vs PyTorch | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| ICAA17K (1,617 test) | **Color Head (`index 1`)** | **0.8791** | **0.8958** | **0.0214** | Exact (`~1e-5`) | **Verified TF-Native Port** |
| ICAA17K (1,617 test) | **MOS Head (`index 0`)** | **0.9058** | **0.9142** | 0.0198 | Exact (`~1e-5`) | **Verified TF-Native Port** |

## Key Engineering & Audit Findings

- **Why TF-Native Reimplementation Was Necessary**: Exporting the official PyTorch DAT model via ONNX (`onnx2tf`) failed due to dynamic `grid_sample` / deformable offset ops and `roll` / `einsum` shape incompatibilities. Rebuilding every block natively in TensorFlow (`tf_models/`) and mapping all 487 PyTorch state_dict tensors (`scripts/tf_native/`) achieved full numerical parity without `SELECT_TF_OPS` (Flex).
- **Contract Clarification (2026-07-11 Audit)**: The exported model outputs `[MOS, Color]`. For pure color aesthetics, use `score_index: 1` and `imagenet` input normalization.
- **Deployment Decision**: Kept as offline/research color subscore rather than active mobile ensemble member due to its 167.7 MB FP16 size and per-image latency.

## Evidence Files in `results/`
All metric tables, per-sample predictions, parity checks, and audit reports backing the numbers above are stored directly in the [`results/`](../results/) directory of this repository.
