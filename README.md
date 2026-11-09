# A-CUT ICAA Color Aesthetics (TF-Native DAT Port)

> **TensorFlow-Native Reimplementation & TFLite Conversion of ICAA17K Deformable Attention Transformer (DAT) for Image Color Aesthetics**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Project: A-CUT](https://img.shields.io/badge/Project-A--CUT%20Capstone-success.svg)](https://github.com/gwanjongkim/photo_score_project)

## 1. Overview

- **Role in A-CUT**: Evaluated Color Aesthetics Specialist & TF-Native Engineering Port (Excluded from on-device active ensemble due to 167.7 MB footprint)
- **Task**: Image Color Aesthetics Assessment (ICAA17K) — Joint `[MOS, Color]` prediction
- **Architecture**: Full TensorFlow-native port (`tf_models/tf_dat_model.py`) of the official ICAA17K Deformable Attention Transformer (DAT) with `LocalAttentionBlock`, `DeformableAttention2D`, `SoftHistogram`, and 487-key strict PyTorch state-dict weight mapping (`weight_maps/`).

## 2. Input / Output Contract & Model Artifacts

- **Input Contract**: RGB float32 `[1, 224, 224, 3]`, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Output Contract**: Float32 vector `[1, 2]` -> `[MOS, Color]`; index `0` is overall MOS (`SRCC 0.9058`), index `1` is Color Aesthetics (`SRCC 0.8791`)
- **Model Files (`models/model_files/` & Releases)**:
  - `icaa_dat_tf_native_fp16.tflite (167.74 MB, SHA256: 10ae68a3283b — attached via GitHub Releases)`
  - `icaa_dat_tf_native_full_integer_int8_float_io.tflite (87.36 MB, SHA256: 8137583cf7ab — attached via GitHub Releases)`

## 3. Benchmark Results

| Dataset | Head / Metric | SRCC | PLCC | MAE | Parity vs PyTorch | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| ICAA17K (1,617 test) | **Color Head (`index 1`)** | **0.8791** | **0.8958** | **0.0214** | Exact (`~1e-5`) | **Verified TF-Native Port** |
| ICAA17K (1,617 test) | **MOS Head (`index 0`)** | **0.9058** | **0.9142** | 0.0198 | Exact (`~1e-5`) | **Verified TF-Native Port** |

## 4. Key Engineering Notes & 2026 Audit Findings

- **Why TF-Native Reimplementation Was Necessary**: Exporting the official PyTorch DAT model via ONNX (`onnx2tf`) failed due to dynamic `grid_sample` / deformable offset ops and `roll` / `einsum` shape incompatibilities. Rebuilding every block natively in TensorFlow (`tf_models/`) and mapping all 487 PyTorch state_dict tensors (`scripts/tf_native/`) achieved full numerical parity without `SELECT_TF_OPS` (Flex).
- **Contract Clarification (2026-07-11 Audit)**: The exported model outputs `[MOS, Color]`. For pure color aesthetics, use `score_index: 1` and `imagenet` input normalization.
- **Deployment Decision**: Kept as offline/research color subscore rather than active mobile ensemble member due to its 167.7 MB FP16 size and per-image latency.

## 5. Quick Start

```bash
pip install -r requirements.txt
python scripts/tf_native/stage_h_full_dat_parity.py --help
python scripts/tf_native/stage_k_convert_tflite_fp16.py --help
```

For full documentation, see:
- [`docs/model_card.md`](./docs/model_card.md) — Architecture, tensor contracts, and limitations
- [`docs/benchmark_summary.md`](./docs/benchmark_summary.md) — Detailed metrics and dataset evaluations
- [`docs/reproduction_notes.md`](./docs/reproduction_notes.md) — Environment and reproduction guide
- [`models/download.md`](./models/download.md) — Model binary hashes and download notes

## 6. A-CUT Ecosystem Repositories

| Repository | Dimension | Role / Status | Key Metric |
| :--- | :---: | :--- | :--- |
| [acut-nima-mobile](https://github.com/gwanjongkim/acut-nima-mobile) | Aesthetic (IAA) | **Production Ensemble (0.40)** | AVA SRCC `0.5894`, AUC `0.8018` |
| [acut-rgnet-mobile](https://github.com/gwanjongkim/acut-rgnet-mobile) | Aesthetic (IAA) | **Production Ensemble (0.45)** | AADB SRCC `0.6819` (FP16 `0.6648`) |
| [acut-mobile-alamp](https://github.com/gwanjongkim/acut-mobile-alamp) | Aesthetic (IAA) | **Production Ensemble (0.15)** | AVA Acc `0.7633`, AUC `0.7877` |
| [acut-icaa-color-aesthetics](https://github.com/gwanjongkim/acut-icaa-color-aesthetics) | Color Aesthetics | **Verified TF-Native Port** | ICAA17K Color SRCC `0.8791`, MOS `0.9058` |
| [acut-koniq-flive-technical-iqa](https://github.com/gwanjongkim/acut-koniq-flive-technical-iqa) | Technical (IQA) | **Production Baseline (4/7 + 3/7)** | Official SRCC `0.7168` (`0.7958` recon), `0` Severe FP |
| [acut-technical-iqa](https://github.com/gwanjongkim/acut-technical-iqa) | Technical (IQA) | **Umbrella Benchmark** | 4,967-Image Official Protocol Suite |
| [acut-topiq-mixed112](https://github.com/gwanjongkim/acut-topiq-mixed112) | Technical (IQA) | **Evaluated Candidate** | SPAQ SRCC `0.8990`, KonIQ `0.8630` |
| [acut-topiq-flive-hybrid](https://github.com/gwanjongkim/acut-topiq-flive-hybrid) | Technical (IQA) | **Best Log-Only Shadow (60/40)** | Official SRCC `0.8069`, MAE `5.04` |
| [acut-c6-topiq-koniq](https://github.com/gwanjongkim/acut-c6-topiq-koniq) | Technical (IQA) | **Negative Result (Rejected)** | Official SRCC `0.7314`, MAE `15.39` |
| [acut-mdiqa-benchmark](https://github.com/gwanjongkim/acut-mdiqa-benchmark) | Technical (IQA) | **Teacher Ceiling Reference** | KonIQ SRCC `0.9464`, SPAQ `0.9136` |
| [acut-musiq-probe](https://github.com/gwanjongkim/acut-musiq-probe) | Technical (IQA) | **Reference & Probe Study** | Official SPAQ `0.9460`, KonIQ `0.9275` |
| [acut-student-iqa](https://github.com/gwanjongkim/acut-student-iqa) | Technical (IQA) | **Distillation Study (No-Go)** | 14k Student SRCC `0.7436`, SPAQ `0.8984` |
| [photo_score_project](https://github.com/gwanjongkim/photo_score_project) | Full Monorepo | **Core Research Workspace** | Complete Training, Audit & Pipeline Code |

## 7. References & Citation

- Xia, Z., Pan, X., Song, S., Li, L. E., & Huang, G. (2022). *Vision Transformer with Deformable Attention*. CVPR, 4794–4803.
- ICAA17K Official Repository & Benchmark Protocol for Image Color Aesthetics Assessment.

```bibtex
@misc{acut_icaa_color_aesthetics_2026,
  title        = {A-CUT ICAA Color Aesthetics (TF-Native DAT Port): TensorFlow-Native Reimplementation & TFLite Conversion of ICAA17K Deformable Attention Transformer (DAT) for Image Color Aesthetics},
  author       = {Kim, Gwanjung},
  year         = {2026},
  howpublished = {\url{https://github.com/gwanjongkim/acut-icaa-color-aesthetics}}
}
```
