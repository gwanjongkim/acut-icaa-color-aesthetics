# A-CUT ICAA Color Aesthetics

#### A-CUT Capstone Project — Mobile Image Quality and Aesthetic Assessment

> TF-native and TFLite adaptation of ICAA (Image Color Aesthetics Assessment) using Delegate Transformer.

## Introduction

TF-native and TFLite adaptation of ICAA (Image Color Aesthetics Assessment) using Delegate Transformer. This model specializes in assessing the color-related aesthetic quality of images.

- **Role**: Converted/evaluated color aesthetics model.
- **Task**: Image Quality and Aesthetic Assessment.

## Model / Method

Ported from TorchScript to TF-Native for TFLite conversion. While it achieves high correlation on the ICAA17K dataset, concerns regarding mobile latency were noted during evaluation.

- **Artifacts**: Inferred from scripts

## Dataset and Benchmark

- Benchmark results are summarized below. See `docs/benchmark_summary.md` for detailed per-dataset analysis.
- Note: Large datasets are excluded due to size/license constraints.

## Model Weights / Artifacts

- Due to size constraints, artifacts larger than 50MB are not included in this repository.
- Recommended hosting: GitHub Releases.
- See `models/download.md`.

## Repository Structure

```
.
├── configs/
├── docs/
├── models/
├── results/
├── scripts/
└── requirements.txt
```

## Requirements

Please refer to `requirements.txt`. Minimal dependencies include `torch`, `tensorflow`, `numpy`, and `pillow`.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/smoke_test_icaa.py --model_path models/model_files/icaa_dat_tf_native_fp16.tflite
```
*(Verify path assumptions in `docs/reproduction_notes.md`)*

## Results

| Dataset | Metric | Value | Evidence Path | Decision |
|---------|--------|-------|---------------|----------|
| ICAA17K | SRCC | 0.8791 | outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv | Evaluated |
| ICAA17K | PLCC | 0.8958 | outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv | Evaluated |

## Limitations

- Results are based on local reproduction and may not exactly match original papers.
- See `docs/model_card.md` for detailed limitations.

## Citation

```bibtex
@misc{acut_icaa_2026,
  title        = {A-CUT ICAA Color Aesthetics: Mobile Image Quality and Aesthetic Assessment Model},
  author       = {Kim, Gwanjung},
  year         = {2026},
  note         = {Capstone project repository}
}
```
### Original References

- Original ICAA / color aesthetics reference will be added after citation verification.

## Related Repositories

- [A-CUT Public Model Repositories](..)
