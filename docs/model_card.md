# Model Card — A-CUT ICAA Color Aesthetics (TF-Native DAT Port)

## Model Details
- **Model Name**: A-CUT ICAA Color Aesthetics (TF-Native DAT Port) (`acut-icaa-color-aesthetics`)
- **Developer**: Gwanjung Kim (A-CUT Mobile Photo Evaluation & Curation Capstone)
- **Architecture**: Full TensorFlow-native port (`tf_models/tf_dat_model.py`) of the official ICAA17K Deformable Attention Transformer (DAT) with `LocalAttentionBlock`, `DeformableAttention2D`, `SoftHistogram`, and 487-key strict PyTorch state-dict weight mapping (`weight_maps/`).
- **Deployment Role**: Evaluated Color Aesthetics Specialist & TF-Native Engineering Port (Excluded from on-device active ensemble due to 167.7 MB footprint)

## Tensor Input / Output Contract
- **Input Contract**: RGB float32 `[1, 224, 224, 3]`, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Output Contract**: Float32 vector `[1, 2]` -> `[MOS, Color]`; index `0` is overall MOS (`SRCC 0.9058`), index `1` is Color Aesthetics (`SRCC 0.8791`)
- **Included Model Artifacts**:
  - `icaa_dat_tf_native_fp16.tflite (167.74 MB, SHA256: 10ae68a3283b — attached via GitHub Releases)`
  - `icaa_dat_tf_native_full_integer_int8_float_io.tflite (87.36 MB, SHA256: 8137583cf7ab — attached via GitHub Releases)`

## Intended Use
- Designed for on-device or offline photo quality and aesthetics evaluation within the A-CUT smartphone photo selection pipeline.
- Not intended as an uncalibrated standalone medical, forensic, or safety-critical classifier.

## Limitations & Audit Qualifications
- **Why TF-Native Reimplementation Was Necessary**: Exporting the official PyTorch DAT model via ONNX (`onnx2tf`) failed due to dynamic `grid_sample` / deformable offset ops and `roll` / `einsum` shape incompatibilities. Rebuilding every block natively in TensorFlow (`tf_models/`) and mapping all 487 PyTorch state_dict tensors (`scripts/tf_native/`) achieved full numerical parity without `SELECT_TF_OPS` (Flex).
- **Contract Clarification (2026-07-11 Audit)**: The exported model outputs `[MOS, Color]`. For pure color aesthetics, use `score_index: 1` and `imagenet` input normalization.
- **Deployment Decision**: Kept as offline/research color subscore rather than active mobile ensemble member due to its 167.7 MB FP16 size and per-image latency.
