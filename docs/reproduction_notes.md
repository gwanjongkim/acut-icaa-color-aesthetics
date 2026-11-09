# Reproduction & Execution Notes — A-CUT ICAA Color Aesthetics (TF-Native DAT Port)

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Available Scripts (`scripts/`)
```bash
python scripts/tf_native/stage_h_full_dat_parity.py --help
python scripts/tf_native/stage_k_convert_tflite_fp16.py --help
```

## Preprocessing & Parity Verification
- **Input Specification**: RGB float32 `[1, 224, 224, 3]`, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Output Specification**: Float32 vector `[1, 2]` -> `[MOS, Color]`; index `0` is overall MOS (`SRCC 0.9058`), index `1` is Color Aesthetics (`SRCC 0.8791`)
- Ensure EXIF orientation transposition (`ImageOps.exif_transpose`) is applied prior to resizing so portrait smartphone photos are not evaluated rotated.
