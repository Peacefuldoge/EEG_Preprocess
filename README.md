# EEG Preprocessing and Cross-Dataset Consistency Analysis

This repository contains the EEG preprocessing pipelines and cross-dataset consistency analysis used to prepare multiple resting-state EEG datasets for downstream self-supervised learning and depression detection experiments.

The current workflow supports four datasets:

- **PRED+CT**
- **MODMA**
- **HBN**
- **TDBRAIN**

The main goal is to reduce dataset-specific differences as much as possible before model training while preserving meaningful EEG characteristics.

---

## Project Structure

```text
.
├── .vscode/
├── __pycache__/
├── figures/
│
├── PREDCT_FINAL.py
├── MODMA_FINAL.py
├── HBN_FINAL.py
├── TDBRAIN_FINAL.py
│
├── Dataset_consistency_analysis.ipynb
├── TDBRAIN_MDD_HC.csv
│
├── dataset_summary.csv
├── preprocessing_metadata.csv
├── channel_overlap.csv
├── band_power_relative_pct.csv
├── psd_js_divergence.csv
├── fc_validation.csv
├── compatibility_report.txt
│
└── README.md
```

### Main preprocessing scripts

| File | Description |
|---|---|
| `PREDCT_FINAL.py` | Preprocessing pipeline for the PRED+CT dataset |
| `MODMA_FINAL.py` | Preprocessing pipeline for the MODMA dataset |
| `HBN_FINAL.py` | Preprocessing pipeline for the HBN dataset |
| `TDBRAIN_FINAL.py` | Preprocessing pipeline for the TDBRAIN dataset |
| `Dataset_consistency_analysis.ipynb` | Cross-dataset quality and consistency analysis |

---

## Preprocessing Pipeline

The preprocessing scripts aim to make EEG signals from different datasets as comparable as possible.

The general pipeline includes:

1. Loading the original EEG recording
2. Selecting EEG channels
3. Band-pass filtering
4. Power-line noise removal
5. Resampling
6. Bad-channel detection
7. Bad-channel interpolation
8. Common average reference
9. ICA-based artifact removal
10. Epoch/window generation
11. Data validation
12. Saving the processed EEG as `.npz`

The preprocessing configuration is designed around a common target sampling frequency of **250 Hz**.

For datasets with sufficiently dense electrode layouts, preprocessing is performed using a common channel configuration to improve compatibility between datasets.

TDBRAIN has a smaller EEG montage and is therefore retained with its available EEG channels rather than artificially expanding it to the higher-density montage used by the pretraining datasets.

---

## Dataset Roles

The datasets are intended for the following experimental setup:

| Dataset | Intended role |
|---|---|
| PRED+CT | Self-supervised pretraining |
| MODMA | Self-supervised pretraining |
| HBN | Self-supervised pretraining |
| TDBRAIN | Downstream depression classification / fine-tuning |

This separation allows the representation model to learn from heterogeneous EEG data during pretraining and then evaluate transferability on TDBRAIN.

---

## Output Format

Processed recordings are stored as compressed NumPy files:

```text
*.npz
```

Depending on the dataset and preprocessing stage, an `.npz` file may contain fields such as:

```python
data
label
subject_id
dataset
channels
sfreq
window_start
```

The main EEG tensor is expected to follow:

```text
(N, C, T)
```

where:

- `N` = number of EEG windows
- `C` = number of EEG channels
- `T` = number of time samples per window

For example, with 1-second windows at 250 Hz:

```text
T = 250
```

---

## Cross-Dataset Consistency Analysis

`Dataset_consistency_analysis.ipynb` is used to compare the processed datasets and identify remaining domain differences before model training.

The analysis includes several complementary checks.

### 1. Dataset summary

Basic properties such as:

- number of samples/windows
- number of channels
- sampling frequency
- signal amplitude statistics
- RMS amplitude

Output:

```text
dataset_summary.csv
```

---

### 2. Preprocessing metadata

Checks whether preprocessing configurations are aligned across datasets.

Output:

```text
preprocessing_metadata.csv
```

---

### 3. Channel overlap

Measures the electrode/channel overlap between datasets.

Output:

```text
channel_overlap.csv
```

This is particularly important when transferring representations learned from high-density datasets to TDBRAIN.

---

### 4. Power spectral density

Power spectral density (PSD) is used to verify whether the frequency characteristics of different datasets are reasonably aligned after preprocessing.

Typical EEG bands include:

| Band | Frequency |
|---|---:|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

Relative band-power statistics are saved to:

```text
band_power_relative_pct.csv
```

---

### 5. PSD distribution divergence

Jensen-Shannon divergence is used to quantify the difference between PSD distributions from different datasets.

Output:

```text
psd_js_divergence.csv
```

Lower divergence generally indicates more similar spectral distributions.

---

### 6. Functional connectivity

Functional connectivity patterns are compared across datasets to ensure preprocessing does not introduce obviously abnormal spatial relationships.

Output:

```text
fc_validation.csv
```

---

### 7. Topographic comparison

EEG topographies are visualized to inspect spatial signal distributions across datasets.

The notebook arranges dataset-level visualizations into matched multi-panel figures whenever possible so that differences can be inspected directly.

Generated figures are stored in:

```text
figures/
```

---

## Compatibility Report

A final human-readable summary of the consistency analysis is saved as:

```text
compatibility_report.txt
```

This report is intended to help determine whether the datasets are sufficiently compatible for joint self-supervised pretraining.

Cross-dataset differences are expected because the original datasets were recorded using different:

- EEG systems
- electrode montages
- reference schemes
- acquisition environments
- participant populations
- recording protocols

The purpose of preprocessing is therefore **not to make the datasets numerically identical**, but to remove avoidable acquisition-related differences while retaining meaningful physiological variation.

---

## Installation

A typical Python environment requires:

```bash
pip install numpy scipy pandas matplotlib scikit-learn mne jupyter
```

Using a dedicated Conda/Mamba environment is recommended.

Example:

```bash
mamba create -n eeg python=3.12
mamba activate eeg

pip install numpy scipy pandas matplotlib scikit-learn mne jupyter
```

Depending on the source EEG format, additional MNE-related dependencies may be required.

---

## Running the Preprocessing

Configure the source and output paths inside the corresponding preprocessing script and run:

```bash
python PREDCT_FINAL.py
```

```bash
python MODMA_FINAL.py
```

```bash
python HBN_FINAL.py
```

```bash
python TDBRAIN_FINAL.py
```

After preprocessing all datasets, run the consistency analysis notebook:

```bash
jupyter notebook Dataset_consistency_analysis.ipynb
```

or:

```bash
jupyter lab
```

---

## Recommended Data Organization

A possible directory structure is:

```text
dataset/
└── preprocessed/
    ├── PRED_CT/
    ├── MODMA/
    ├── HBN/
    └── TDBRAIN/
```

Keeping each dataset in a separate directory makes dataset-level sampling, normalization, validation, and debugging easier.

---

## Normalization for Model Training

The preprocessing pipeline standardizes acquisition-related properties, but model-level normalization should generally remain part of the training pipeline.

A recommended strategy is to calculate normalization statistics using **training data only** and apply the same transformation to validation and test data.

This avoids information leakage and makes evaluation more realistic.

In other words:

```text
Raw EEG
   ↓
Offline preprocessing
   ↓
Saved .npz
   ↓
Train/validation/test split
   ↓
Training-set normalization statistics
   ↓
Model input
```

Do not independently normalize the validation or test sets using statistics calculated from those sets if the normalization method depends on dataset-level statistics.

---

## Intended Research Workflow

```text
PRED+CT ─┐
MODMA   ─┼──> Self-Supervised Pretraining
HBN     ─┘              │
                        ↓
                 Pretrained Encoder
                        │
                        ↓
                    TDBRAIN
                        │
                        ↓
             MDD vs. Healthy Control
```

The broader objective is to investigate whether self-supervised EEG representations learned from multiple heterogeneous datasets can transfer effectively to EEG-based depression detection.

---

## Notes

- Large raw EEG datasets and generated `.npz` files should normally not be committed directly to GitHub.
- Consider adding dataset directories, cache files, and generated outputs to `.gitignore`.
- Always perform subject-level splitting before downstream evaluation when conducting subject-independent experiments.
- Dataset compatibility should be evaluated using multiple signal characteristics rather than a single metric such as RMS.

---

## License

This repository contains research code only.

The EEG datasets used by this project are distributed by their respective owners and remain subject to their original licenses and data-use agreements.

---

## Author

**ZHU ZHENZUO**

Research project on self-supervised representation learning for EEG-based depression detection.
