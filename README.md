# CASPEr-GRL Data Processing & Burst Detection Pipeline

This repository provides the pipeline for converting SPS telemetry files into NPY arrays, optionally preprocessing the signals, and performing Radon-based burst detection. The system is configurable through a single `config.py` and designed for reproducible scientific analysis.

<img width="1363" height="394" alt="Screenshot 2025-11-25 at 1 19 21 AM" src="https://github.com/user-attachments/assets/2d1cfb01-f9b6-4066-b6b3-0a09d9e544eb" />

---

## 1. Repository Structure

```
CASPEr-GRL/
├── README.md
├── requirements.txt
├── _sps_data/         # Raw SPS files
├── _raw_npy/          # SPS to NPY converted files
├── _cleaned_npy/      # Preprocessed output
├── _detected_srb/     # Radon-based detection output
└── source/
    ├── config.py
    ├── convert_SPS.py
    ├── preprocess_data.py
    ├── radon_analysis.py
    ├── dbx_sync.py
    └── preprocess/
```

---

## 2. Pipeline Summary

### (1) SPS → NPY Conversion
`convert_SPS.py` reads `.sps` telemetry files and outputs `.npy` arrays.

### (2) Preprocessing (optional)
`preprocess_data.py` performs gap-filling, filtering, detrending, and other signal preparation steps.

### (3) Radon Transform Burst Identification
`radon_analysis.py` applies Radon-transform feature extraction and produces detection maps and diagnostic plots.

### (4) Optional Dropbox Synchronization
`dbx_sync.py` acquires SPS files from cloud storage if enabled.

---

## 3. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Running the Full Pipeline

```bash
bash sample_data.sh
```

Or run steps manually:

```bash
python3 source/convert_SPS.py
python3 source/preprocess_data.py
python3 source/radon_analysis.py
```

---

# 5. Configuration Guide (`config.py`)

All behavior is controlled in:

```
source/config.py
```

---

## 5.1 Dropbox Sync Settings

```python
LOG_DBX         = True
SAMPLE_RATE_DBX = 0.01
DRY_RUN_DBX     = False
WANT_DBX        = ["sps"]
DESTINATION_PATH_DBX = "./_sps_data"
```

Controls remote synchronization behavior.

---

## 5.2 SPS Conversion (SPS to NPY)

```python
SPS_DUMP_STN = DESTINATION_PATH_DBX
NPY_DIR_STN  = "./_raw_npy"
MAKE_NPY_STN = True
SHOW_SPS_STN = False
```

Ensures SPS files are converted into NPY format.

---

## 5.3 Preprocessing Options

```python
NPY_DUMP_PRP    = NPY_DIR_STN
FILL_ZERO_PRP   = True
PREPROCESS_PRP  = False
CLEANED_NPY_PRP = "./_cleaned_npy"
```

To enable full filtering and preprocessing:

```python
PREPROCESS_PRP = True
```

---

## 5.4 Radon Transform & Burst Detection

```python
INPUT_DIR_RA  = CLEANED_NPY_PRP
OUTPUT_DIR_RA = "./_detected_srb"
CREATE_IMG_RA = True
```

Controls Radon analysis input/output and image creation.

---

## 5.5 Directory Creation

Run:

```bash
python3 source/config.py
```

to auto-create all required output folders.

---
