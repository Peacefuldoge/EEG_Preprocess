#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
MODMA 128-channel resting-state EEG preprocessing (.mat version).

Dataset
-------
EEG_128channels_resting_lanzhou_2015

Expected input example
----------------------
E:\Workspace\dataset\modma\
    EEG_128channels_resting_lanzhou_2015\
    EEG_128channels_resting_lanzhou_2015\
    data\
        02010002.mat
        ...

MODMA resting-state .mat structure
----------------------------------
The official MODMA description states that EEG.data contains 129 signals:
    EEG.data[0:128] -> electrodes E1 ... E128
    EEG.data[128]   -> Cz reference channel

This script maps the HydroCel E1-E128 + Cz layout onto the same fixed
COMMON_64 channel configuration used by PRED+CT. Manufacturer 10-10
equivalences are used where available; six additional PRED+CT positions
(M1/M2/PO5/PO6/CB1/CB2) use fixed nearest-position HydroCel matches.

Recording condition
-------------------
- Eyes-closed resting state
- Approximately 5 minutes
- Native sampling rate: 250 Hz
- HydroCel Geodesic Sensor Net, 128 electrodes
- Original reference: Cz

Diagnosis from original filename
--------------------------------
0201xxxx.mat -> MDD
0203xxxx.mat -> HEALTHY

Output
------
One compressed NPZ per recording:
    02010002_EC.npz

Saved arrays
------------
data : (N, 64, 250), float32, unit = microvolts (uV)
fc   : (N, 64, 64), float32, Pearson correlation

N is the number of complete non-overlapping 1-s windows.

Preprocessing
-------------
MAT -> MNE Raw (internally volts)
-> 50-Hz notch
-> 0.5-45 Hz Butterworth IIR
-> resample to 250 Hz if needed
-> robust-z bad-channel detection
-> 3-nearest-electrode interpolation
-> common-average reference
-> conservative FastICA ocular removal using frontal EEG proxy channels
-> complete non-overlapping 1-s EC windows
-> whole-window mean baseline
-> convert final saved EEG to uV
-> Pearson FC for each window

Notes
-----
1. Net Station / EEGLAB-style EEG.data is assumed to be stored in uV.
   Change MAT_INPUT_UNIT if inspection of your files proves otherwise.
2. MODMA resting files do not provide dedicated EOG channels in the published
   129-signal layout. Therefore ocular ICA uses frontal EEG proxies and is
   deliberately conservative (max two removed components).
"""

from __future__ import annotations

import gc
import json
import re
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from scipy.io import loadmat
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

DATA_DIR = Path(
    r"E:\Workspace\dataset\modma"
    r"\EEG_128channels_resting_lanzhou_2015"
    r"\EEG_128channels_resting_lanzhou_2015"
    r"\data"
)

OUTPUT_DIR = Path(
    r"E:\Workspace\dataset\preprocessed\MODMA"
)

# The example supplied by the user.
SINGLE_TEST_FILE = DATA_DIR / "02010002.mat"

# False = batch-process every *.mat in DATA_DIR.
# True  = process only SINGLE_TEST_FILE.
PROCESS_ONLY_SINGLE_TEST_FILE = False

DATASET_NAME = "MODMA"

# Native MODMA sampling rate is 250 Hz.
TARGET_SFREQ = 250.0

BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0
DEFAULT_LINE_FREQ = 50.0
FILTER_ORDER = 4

BAD_Z_THRESHOLD = 6.0
NEIGHBOR_COUNT = 3

ICA_MAX_COMPONENTS = 10
ICA_MAX_OCULAR_COMPONENTS = 2
ICA_FIT_SECONDS = 30.0
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = 300
ICA_EOG_THRESHOLD = 2.5
ICA_MIN_ABS_EOG_SCORE = 0.6

# MODMA resting-state MAT files do not contain dedicated EOG in the published
# E1-E128 + Cz layout. These HydroCel electrodes are near the frontal pole and
# are used only as conservative ocular-reference proxies for ICA scoring.
ICA_FRONTAL_PROXY_CHANNELS = ["Fp1", "Fpz", "Fp2"]
USE_FRONTAL_PROXY_ICA = True

WINDOW_SECONDS = 1.0
APPLY_BASELINE = True

# Net Station / EEGLAB-style numeric EEG.data is normally represented in uV.
# Accepted values: "uV" or "V".
MAT_INPUT_UNIT = "uV"

# MNE operates internally in volts; exported windows are converted to uV.
OUTPUT_EEG_SCALE = 1e6
OUTPUT_EEG_UNIT = "uV"

# -------------------------------------------------------------------------
# Final window-level artifact rejection (saved EEG is in uV)
# -------------------------------------------------------------------------
WINDOW_QC_ENABLED = True

# Hard rejection limits for resting scalp EEG.
WINDOW_QC_MAX_ABS_UV = 200.0
WINDOW_QC_MAX_P2P_UV = 400.0
WINDOW_QC_MIN_RMS_UV = 0.05

# Subject/run-adaptive rejection. These are deliberately conservative.
WINDOW_QC_ROBUST_Z_THRESHOLD = 8.0
WINDOW_QC_MIN_WINDOWS_FOR_ROBUST = 20

# If QC removes a large fraction, keep the recording out of the dataset
# rather than silently accepting a severely contaminated recording.
WINDOW_QC_WARN_REMOVAL_FRACTION = 0.20
WINDOW_QC_MIN_KEEP_FRACTION = 0.50

OVERWRITE = True


# =============================================================================
# 2. FIXED 64-CHANNEL HARMONIZED CONFIGURATION
# =============================================================================

# Exact final order used by PRED+CT. MODMA must match this order so that the
# pretraining datasets share the same channel semantics and tensor shape.
COMMON_64 = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "M1", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "TP8", "M2",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
    "CB1", "O1", "Oz", "O2", "CB2",
]

HYDROCEL_128 = [f"E{i}" for i in range(1, 129)]
HYDROCEL_129 = HYDROCEL_128 + ["Cz"]
MONTAGE_NAME = "GSN-HydroCel-129"

# Fixed mapping from the PRED+CT COMMON_64 semantic positions to the MODMA
# HydroCel-129 sensor layout.
#
# Most entries are manufacturer-published approximate 10-10 equivalents for
# the 128-channel HydroCel GSN. The six positions not present in that table
# (M1/M2/PO5/PO6/CB1/CB2) use fixed nearest-position matches derived from the
# HydroCel-129 and PRED+CT electrode coordinates. Keeping this dictionary fixed
# ensures every MODMA participant uses exactly the same channel mapping.
COMMON64_TO_HYDROCEL = {
    "Fp1": "E22",
    "Fpz": "E14",
    "Fp2": "E9",
    "AF3": "E23",
    "AF4": "E3",

    "F7": "E33",
    "F5": "E27",
    "F3": "E24",
    "F1": "E19",
    "Fz": "E11",
    "F2": "E4",
    "F4": "E124",
    "F6": "E123",
    "F8": "E122",

    "FT7": "E34",
    "FC5": "E28",
    "FC3": "E29",
    "FC1": "E13",
    "FCz": "E6",
    "FC2": "E112",
    "FC4": "E111",
    "FC6": "E117",
    "FT8": "E116",

    "T7": "E45",
    "C5": "E41",
    "C3": "E36",
    "C1": "E30",
    "Cz": "Cz",
    "C2": "E105",
    "C4": "E104",
    "C6": "E103",
    "T8": "E108",

    # Fixed nearest-position additions required by PRED+CT COMMON_64.
    "M1": "E56",

    "TP7": "E46",
    "CP5": "E47",
    "CP3": "E42",
    "CP1": "E37",
    "CPz": "E55",
    "CP2": "E87",
    "CP4": "E93",
    "CP6": "E98",
    "TP8": "E102",

    # Fixed nearest-position addition.
    "M2": "E107",

    "P7": "E58",
    "P5": "E51",
    "P3": "E52",
    "P1": "E60",
    "Pz": "E62",
    "P2": "E85",
    "P4": "E92",
    "P6": "E97",
    "P8": "E96",

    "PO7": "E65",
    # Fixed nearest-position addition.
    "PO5": "E66",
    "PO3": "E67",
    "POz": "E72",
    "PO4": "E77",
    # Fixed nearest-position addition.
    "PO6": "E84",
    "PO8": "E90",

    # Fixed nearest-position additions for the PRED+CT cerebellar positions.
    "CB1": "E69",
    "O1": "E70",
    "Oz": "E75",
    "O2": "E83",
    "CB2": "E89",
}

if list(COMMON64_TO_HYDROCEL.keys()) != COMMON_64:
    raise RuntimeError(
        "COMMON64_TO_HYDROCEL keys must exactly match COMMON_64 order."
    )

if len(set(COMMON64_TO_HYDROCEL.values())) != len(COMMON_64):
    raise RuntimeError(
        "COMMON64_TO_HYDROCEL must be one-to-one; duplicate HydroCel sensors found."
    )

# MODMA is recorded online with Cz as the reference. When Cz is stored as a
# zero-valued reference trace (or absent from a 128-row export), it must NOT be
# treated as a bad/flat EEG channel before common-average re-referencing.
ONLINE_REFERENCE_TARGET = "Cz"

# =============================================================================
# 3. MAT LOADING
# =============================================================================

def _get_struct_field(obj, name: str):
    """Read one field from scipy-loaded MATLAB structs robustly."""
    if hasattr(obj, name):
        return getattr(obj, name)

    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        value = obj[name]
        while isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
        return value

    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]

    raise KeyError(f"MATLAB EEG struct has no field '{name}'.")


def _extract_chanloc_labels(chanlocs) -> list[str]:
    """Best-effort extraction of EEGLAB chanlocs.labels."""
    labels: list[str] = []

    if chanlocs is None:
        return labels

    arr = np.atleast_1d(chanlocs).ravel()

    for item in arr:
        value = None

        if hasattr(item, "labels"):
            value = getattr(item, "labels")
        elif isinstance(item, np.void) and item.dtype.names and "labels" in item.dtype.names:
            value = item["labels"]

        if value is None:
            continue

        while isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()

        labels.append(str(value).strip())

    return labels


def load_modma_mat(
    mat_path: Path,
) -> tuple[np.ndarray, float, list[str], dict]:
    """
    Load one original MODMA resting-state MAT file.

    Returns
    -------
    data_129
        Shape (129, T): E1-E128 plus Cz reference.
        If an export contains only 128 rows, a zero Cz reference trace is
        appended because MODMA was acquired online referenced to Cz.
    sfreq
        Sampling frequency.
    source_channel_names
        E1 ... E128, Cz.
    mat_info
        Validation metadata from the MAT structure.
    """
    mat_path = Path(mat_path)

    if not mat_path.exists():
        raise FileNotFoundError(
            f"MAT file not found: {mat_path}"
        )

    try:
        mat = loadmat(
            mat_path,
            squeeze_me=True,
            struct_as_record=False,
        )
    except NotImplementedError as exc:
        raise RuntimeError(
            f"{mat_path.name} appears to be MATLAB v7.3/HDF5. "
            "This script expects a MAT structure readable by scipy.io.loadmat."
        ) from exc

    if "EEG" not in mat:
        visible_keys = [
            key
            for key in mat.keys()
            if not key.startswith("__")
        ]
        raise KeyError(
            f"{mat_path.name}: expected MATLAB variable 'EEG', "
            f"found {visible_keys}"
        )

    eeg = mat["EEG"]

    data = np.asarray(
        _get_struct_field(
            eeg,
            "data",
        )
    ).squeeze()

    if data.ndim != 2:
        raise ValueError(
            f"{mat_path.name}: EEG.data must be 2-D, "
            f"got shape {data.shape}"
        )

    if data.shape[0] in (128, 129):
        channel_first = data
    elif data.shape[1] in (128, 129):
        channel_first = data.T
    else:
        raise ValueError(
            f"{mat_path.name}: cannot identify 128/129 channels from "
            f"EEG.data shape {data.shape}"
        )

    n_stored_channels = int(
        channel_first.shape[0]
    )

    data_128 = np.asarray(
        channel_first[:128],
        dtype=np.float64,
    )

    if n_stored_channels == 129:
        cz_data = np.asarray(
            channel_first[128:129],
            dtype=np.float64,
        )
        cz_source = "stored_129th_channel"
    else:
        # Because the acquisition reference is Cz, a zero reference trace can
        # be appended before average rereferencing if a 128-row export omitted
        # the reference channel.
        cz_data = np.zeros(
            (1, data_128.shape[1]),
            dtype=np.float64,
        )
        cz_source = "zero_reference_reconstructed"

    data_129 = np.concatenate(
        [
            data_128,
            cz_data,
        ],
        axis=0,
    )

    try:
        sfreq = float(
            np.asarray(
                _get_struct_field(
                    eeg,
                    "srate",
                )
            ).squeeze()
        )
    except Exception:
        sfreq = 250.0

    if (
        not np.isfinite(sfreq)
        or sfreq <= 0
    ):
        raise ValueError(
            f"{mat_path.name}: invalid EEG.srate={sfreq}"
        )

    chanloc_labels = []

    try:
        chanlocs = _get_struct_field(
            eeg,
            "chanlocs",
        )
        chanloc_labels = (
            _extract_chanloc_labels(
                chanlocs
            )
        )
    except Exception:
        pass

    if len(chanloc_labels) >= 128:
        first_128 = chanloc_labels[:128]

        channel_labels_match = (
            [
                str(x).upper()
                for x in first_128
            ]
            == [
                str(x).upper()
                for x in HYDROCEL_128
            ]
        )
    else:
        channel_labels_match = None

    mat_info = {
        "mat_variable": "EEG",
        "original_eeg_data_shape": list(
            channel_first.shape
        ),
        "stored_channel_count": n_stored_channels,
        "source_layout": "E1-E128 + Cz reference",
        "cz_reference_source": cz_source,
        "mat_sfreq": sfreq,
        "chanloc_labels_found": len(
            chanloc_labels
        ),
        "first_128_chanlocs_match_E1_E128": (
            channel_labels_match
        ),
        "final_channel_count": 64,
        "final_channel_order": COMMON_64,
        "common64_to_hydrocel": (
            COMMON64_TO_HYDROCEL
        ),
    }

    return (
        data_129,
        sfreq,
        HYDROCEL_129.copy(),
        mat_info,
    )


def convert_input_to_volts(
    data: np.ndarray,
) -> np.ndarray:
    """Convert MAT numeric samples to volts for MNE processing."""
    unit = MAT_INPUT_UNIT.strip().lower()

    if unit in {
        "uv",
        "µv",
        "μv",
        "microvolt",
        "microvolts",
    }:
        return (
            np.asarray(
                data,
                dtype=np.float64,
            )
            * 1e-6
        )

    if unit in {
        "v",
        "volt",
        "volts",
    }:
        return np.asarray(
            data,
            dtype=np.float64,
        )

    raise ValueError(
        f"Unsupported MAT_INPUT_UNIT={MAT_INPUT_UNIT!r}; "
        "use 'uV' or 'V'."
    )


def make_raw_from_mat(
    data_129: np.ndarray,
    sfreq: float,
) -> mne.io.RawArray:
    """
    Select and rename the MODMA HydroCel sensors to PRED+CT COMMON_64.

    The selected data are still in the original Cz-referenced potential
    representation at this stage. Common-average re-referencing occurs later.
    """
    if data_129.shape[0] != len(
        HYDROCEL_129
    ):
        raise ValueError(
            f"Expected 129 source channels, got {data_129.shape}"
        )

    source_index = {
        name: i
        for i, name in enumerate(
            HYDROCEL_129
        )
    }

    selected_source_names = [
        COMMON64_TO_HYDROCEL[
            target
        ]
        for target in COMMON_64
    ]

    selected_indices = [
        source_index[
            source
        ]
        for source in selected_source_names
    ]

    selected_data = np.asarray(
        data_129[
            selected_indices,
            :
        ],
        dtype=np.float64,
    )

    data_v = convert_input_to_volts(
        selected_data
    )

    info = mne.create_info(
        ch_names=COMMON_64,
        sfreq=float(
            sfreq
        ),
        ch_types=[
            "eeg"
        ] * len(
            COMMON_64
        ),
    )

    raw = mne.io.RawArray(
        data_v,
        info,
        verbose="ERROR",
    )

    # Use the actual HydroCel coordinates of the selected source sensors while
    # exposing the harmonized PRED+CT semantic channel names to downstream code.
    hydrocel_montage = (
        mne.channels.make_standard_montage(
            MONTAGE_NAME
        )
    )

    hydrocel_positions = (
        hydrocel_montage
        .get_positions()[
            "ch_pos"
        ]
    )

    common_positions = {
        target: np.asarray(
            hydrocel_positions[
                COMMON64_TO_HYDROCEL[
                    target
                ]
            ],
            dtype=float,
        )
        for target in COMMON_64
    }

    common_montage = (
        mne.channels.make_dig_montage(
            ch_pos=common_positions,
            coord_frame="head",
        )
    )

    raw.set_montage(
        common_montage,
        on_missing="raise",
        verbose="ERROR",
    )

    return raw


# =============================================================================
# 4. FILE / LABEL HELPERS
# =============================================================================

def parse_modma_subject_id(path: Path) -> str:
    """Use the original numeric filename as the subject identifier."""
    return Path(path).stem


def diagnosis_from_filename(path: Path) -> tuple[str, int]:
    """
    Official MODMA original filename convention:
      0201... -> MDD
      0203... -> HEALTHY
    """
    name = Path(path).stem

    if name.startswith("0201"):
        return "MDD", 1

    if name.startswith("0203"):
        return "HEALTHY", 0

    raise ValueError(
        f"{path.name}: cannot infer diagnosis from filename. "
        "Expected prefix 0201 (MDD) or 0203 (HEALTHY)."
    )


# =============================================================================
# 5. BAD-CHANNEL DETECTION + INTERPOLATION
# =============================================================================

def detect_bad_eeg_channels(
    raw: mne.io.BaseRaw,
) -> tuple[list[str], dict[str, float]]:
    """
    Robust-z bad-channel detection matched to the PRED+CT/TDBRAIN pipeline.

    Uses log10(channel SD) from the first <=30 s.
    """
    eeg = raw.copy().pick(COMMON_64)

    if eeg.times[-1] > 30.0:
        eeg.crop(
            tmin=0.0,
            tmax=30.0,
            include_tmax=False,
        )

    data = eeg.get_data()

    channel_std = np.std(
        data,
        axis=1,
    )

    log_std = np.log10(
        channel_std + np.finfo(float).eps
    )

    median = np.median(log_std)
    mad = np.median(
        np.abs(log_std - median)
    )

    if mad < np.finfo(float).eps:
        robust_z = np.zeros_like(log_std)
    else:
        robust_z = (
            0.67448975
            * (log_std - median)
            / mad
        )

    flat = channel_std < 1e-12

    bad_mask = (
        flat
        | ~np.isfinite(channel_std)
        | (np.abs(robust_z) > BAD_Z_THRESHOLD)
    )

    # Cz is the online reference in MODMA and may be stored as an all-zero
    # reference trace before CAR. Do not classify that expected reference trace
    # as a bad channel. After CAR it becomes a valid average-referenced signal.
    if ONLINE_REFERENCE_TARGET in eeg.ch_names:
        cz_idx = eeg.ch_names.index(
            ONLINE_REFERENCE_TARGET
        )
        bad_mask[cz_idx] = False
        robust_z[cz_idx] = 0.0

    bads = [
        eeg.ch_names[i]
        for i in np.where(bad_mask)[0]
    ]

    z_scores = {
        eeg.ch_names[i]: float(robust_z[i])
        for i in range(len(eeg.ch_names))
    }

    return bads, z_scores


def get_eeg_positions(
    raw: mne.io.BaseRaw,
) -> dict[str, np.ndarray]:
    montage = raw.get_montage()

    if montage is None:
        raise RuntimeError(
            "No HydroCel montage is available for interpolation."
        )

    montage_positions = montage.get_positions()["ch_pos"]

    positions = {}

    for ch in COMMON_64:
        if ch not in montage_positions:
            continue

        xyz = np.asarray(
            montage_positions[ch],
            dtype=float,
        )

        if np.all(np.isfinite(xyz)):
            positions[ch] = xyz

    missing = [
        ch for ch in COMMON_64
        if ch not in positions
    ]

    if missing:
        raise RuntimeError(
            f"Missing selected-channel coordinates for channels: {missing}"
        )

    return positions


def interpolate_nearest_electrodes(
    raw: mne.io.BaseRaw,
    bad_channels: list[str],
    positions: dict[str, np.ndarray],
    k: int = NEIGHBOR_COUNT,
) -> tuple[mne.io.BaseRaw, dict[str, list[str]]]:
    """Inverse-distance interpolation from k nearest good HydroCel electrodes."""
    if not bad_channels:
        return raw, {}

    interpolation_log = {}

    good_channels = [
        ch for ch in COMMON_64
        if ch not in bad_channels
    ]

    for bad in bad_channels:
        candidates = []

        for good in good_channels:
            distance = float(
                np.linalg.norm(
                    positions[good]
                    - positions[bad]
                )
            )

            if np.isfinite(distance) and distance > 0:
                candidates.append(
                    (good, distance)
                )

        candidates.sort(
            key=lambda item: item[1]
        )

        selected = candidates[:k]

        if len(selected) < k:
            raise RuntimeError(
                f"Cannot interpolate {bad}: only "
                f"{len(selected)} valid neighbours."
            )

        neighbours = [
            name for name, _ in selected
        ]

        distances = np.asarray(
            [distance for _, distance in selected],
            dtype=float,
        )

        weights = 1.0 / np.maximum(
            distances,
            1e-12,
        )

        weights /= weights.sum()

        bad_idx = raw.ch_names.index(bad)

        neighbour_idx = [
            raw.ch_names.index(ch)
            for ch in neighbours
        ]

        raw._data[bad_idx, :] = np.average(
            raw._data[neighbour_idx, :],
            axis=0,
            weights=weights,
        )

        interpolation_log[bad] = neighbours

    return raw, interpolation_log


# =============================================================================
# 6. ICA
# =============================================================================

def run_ica_ocular_removal(
    raw: mne.io.BaseRaw,
) -> tuple[mne.io.BaseRaw, list[int], bool, list[str]]:
    """
    Conservative FastICA ocular removal using frontal EEG proxy channels.

    MODMA's published resting MAT layout does not provide dedicated EOG
    channels, so E25/E14/E8 are used only to score components. At most two
    components are removed.
    """
    if not USE_FRONTAL_PROXY_ICA:
        return raw, [], True, []

    proxy_channels = [
        ch for ch in ICA_FRONTAL_PROXY_CHANNELS
        if ch in raw.ch_names
    ]

    if not proxy_channels:
        return raw, [], True, []

    fit_raw = raw.copy()

    if fit_raw.times[-1] > ICA_FIT_SECONDS:
        fit_raw.crop(
            tmin=0.0,
            tmax=ICA_FIT_SECONDS,
            include_tmax=False,
        )

    n_components = min(
        ICA_MAX_COMPONENTS,
        len(COMMON_64) - 1,
    )

    ica = ICA(
        n_components=n_components,
        method="fastica",
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
        verbose="ERROR",
    )

    converged = True

    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter("always")

        ica.fit(
            fit_raw,
            picks="eeg",
            verbose="ERROR",
        )

        if any(
            issubclass(
                warning.category,
                ConvergenceWarning,
            )
            for warning in caught
        ):
            converged = False

    if not converged:
        return raw, [], False, proxy_channels

    candidate_scores: dict[int, float] = {}

    for proxy_name in proxy_channels:
        try:
            inds, scores = ica.find_bads_eog(
                fit_raw,
                ch_name=proxy_name,
                threshold=ICA_EOG_THRESHOLD,
                verbose="ERROR",
            )

            scores = np.asarray(scores)

            for idx in inds:
                idx = int(idx)
                score = (
                    float(abs(scores[idx]))
                    if idx < len(scores)
                    else 0.0
                )

                candidate_scores[idx] = max(
                    candidate_scores.get(idx, 0.0),
                    score,
                )

        except Exception:
            continue

    # ranked = sorted(
    #     candidate_scores,
    #     key=candidate_scores.get,
    #     reverse=True,
    # )

    # exclude = ranked[
    #     :ICA_MAX_OCULAR_COMPONENTS
    # ]
    
    ranked = sorted(
        candidate_scores,
        key=candidate_scores.get,
        reverse=True,
    )

    exclude = [
        int(idx)
        for idx in ranked
        if float(candidate_scores[idx]) >= ICA_MIN_ABS_EOG_SCORE
    ][:ICA_MAX_OCULAR_COMPONENTS]

    if exclude:
        ica.exclude = exclude

        raw = ica.apply(
            raw.copy(),
            verbose="ERROR",
        )

    return raw, exclude, True, proxy_channels


# =============================================================================
# 7. WINDOWING + FC
# =============================================================================

def make_one_second_ec_windows(
    raw: mne.io.BaseRaw,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    The full MODMA resting recording is eyes-closed.

    Keep only complete, non-overlapping 1-s windows.
    """
    eeg = raw.copy().pick(COMMON_64)
    eeg.reorder_channels(COMMON_64)

    sfreq = float(
        eeg.info["sfreq"]
    )

    samples_per_window = int(
        round(
            WINDOW_SECONDS * sfreq
        )
    )

    if samples_per_window <= 0:
        raise ValueError(
            "Invalid window length."
        )

    n_windows = (
        eeg.n_times
        // samples_per_window
    )

    if n_windows <= 0:
        raise ValueError(
            "Recording contains no complete 1-s window."
        )

    usable_samples = (
        n_windows
        * samples_per_window
    )

    continuous = eeg.get_data(
        start=0,
        stop=usable_samples,
    )

    data = continuous.reshape(
        len(COMMON_64),
        n_windows,
        samples_per_window,
    ).transpose(
        1,
        0,
        2,
    )

    if APPLY_BASELINE:
        data = (
            data
            - data.mean(
                axis=2,
                keepdims=True,
            )
        )

    # MNE internal unit is V. Export uV.
    data = (
        data
        * OUTPUT_EEG_SCALE
    )

    data = np.asarray(
        data,
        dtype=np.float32,
    )

    starts = (
        np.arange(
            n_windows,
            dtype=np.float64,
        )
        * WINDOW_SECONDS
    )

    ends = (
        starts
        + WINDOW_SECONDS
    )

    labels = np.asarray(
        ["Eyes Closed"] * n_windows,
        dtype=np.str_,
    )

    event_codes = np.zeros(
        n_windows,
        dtype=np.int16,
    )

    block_ids = np.asarray(
        ["EC_full_recording"] * n_windows,
        dtype=np.str_,
    )

    block_start = np.zeros(
        n_windows,
        dtype=np.float64,
    )

    block_end_value = float(
        n_windows
        * WINDOW_SECONDS
    )

    block_end = np.full(
        n_windows,
        block_end_value,
        dtype=np.float64,
    )

    return (
        data,
        starts,
        ends,
        labels,
        event_codes,
        block_ids,
        block_start,
        block_end,
    )



def _window_qc_robust_z(values: np.ndarray) -> np.ndarray:
    """One-sided robust z-score used for unusually large window metrics."""
    values = np.asarray(values, dtype=np.float64)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    if not np.isfinite(mad) or mad < np.finfo(float).eps:
        return np.zeros_like(values, dtype=np.float64)

    return (
        0.67448975
        * (values - median)
        / mad
    )


def reject_artifact_windows(
    data_uv: np.ndarray,
    starts: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Detect severe 1-s EEG artifacts after all preprocessing and uV conversion.

    The detector combines:
      1) absolute hard limits:
           max |amplitude| > 200 uV
           worst-channel peak-to-peak > 400 uV
           global RMS < 0.05 uV (near-flat window)
      2) adaptive high-outlier detection using robust z-scores of:
           global RMS
           max |amplitude|
           worst-channel peak-to-peak

    Robust-z rejection is enabled only when enough windows are available.
    The returned boolean mask indexes the *pre-QC* windows.
    """
    if data_uv.ndim != 3:
        raise ValueError(
            f"Window QC expected (N,C,T), got {data_uv.shape}"
        )

    n_windows = int(data_uv.shape[0])

    if n_windows == 0:
        raise ValueError("Window QC received zero windows.")

    if not WINDOW_QC_ENABLED:
        return (
            np.ones(n_windows, dtype=bool),
            {
                "enabled": False,
                "n_before": n_windows,
                "n_kept": n_windows,
                "n_removed": 0,
                "removed_fraction": 0.0,
                "removed_indices": [],
                "removed_window_start_sec": [],
            },
        )

    x = np.asarray(
        data_uv,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(x)):
        # Non-finite windows are never allowed into model training.
        finite_window = np.all(
            np.isfinite(x),
            axis=(1, 2),
        )
    else:
        finite_window = np.ones(
            n_windows,
            dtype=bool,
        )

    # Replace non-finite values only for metric calculation. Such windows are
    # still rejected through finite_window below.
    x_safe = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    rms_uv = np.sqrt(
        np.mean(
            x_safe * x_safe,
            axis=(1, 2),
        )
    )

    max_abs_uv = np.max(
        np.abs(x_safe),
        axis=(1, 2),
    )

    # Maximum channel-wise peak-to-peak value within each 1-s window.
    max_p2p_uv = np.max(
        np.ptp(
            x_safe,
            axis=2,
        ),
        axis=1,
    )

    hard_nonfinite = ~finite_window
    hard_flat = rms_uv < WINDOW_QC_MIN_RMS_UV
    hard_amplitude = max_abs_uv > WINDOW_QC_MAX_ABS_UV
    hard_p2p = max_p2p_uv > WINDOW_QC_MAX_P2P_UV

    if n_windows >= WINDOW_QC_MIN_WINDOWS_FOR_ROBUST:
        rms_z = _window_qc_robust_z(rms_uv)
        max_abs_z = _window_qc_robust_z(max_abs_uv)
        p2p_z = _window_qc_robust_z(max_p2p_uv)

        robust_rms = rms_z > WINDOW_QC_ROBUST_Z_THRESHOLD
        robust_amplitude = (
            max_abs_z > WINDOW_QC_ROBUST_Z_THRESHOLD
        )
        robust_p2p = p2p_z > WINDOW_QC_ROBUST_Z_THRESHOLD
    else:
        rms_z = np.zeros(n_windows, dtype=np.float64)
        max_abs_z = np.zeros(n_windows, dtype=np.float64)
        p2p_z = np.zeros(n_windows, dtype=np.float64)

        robust_rms = np.zeros(n_windows, dtype=bool)
        robust_amplitude = np.zeros(n_windows, dtype=bool)
        robust_p2p = np.zeros(n_windows, dtype=bool)

    reject = (
        hard_nonfinite
        | hard_flat
        | hard_amplitude
        | hard_p2p
        | robust_rms
    )

    keep = ~reject

    n_removed = int(np.sum(reject))
    n_kept = int(np.sum(keep))
    removed_fraction = (
        n_removed / n_windows
    )

    if n_kept == 0:
        raise RuntimeError(
            "Window QC rejected every EEG window."
        )

    if (
        n_kept / n_windows
        < WINDOW_QC_MIN_KEEP_FRACTION
    ):
        raise RuntimeError(
            "Window QC rejected too much of the recording: "
            f"kept {n_kept}/{n_windows} "
            f"({n_kept / n_windows:.1%}), below minimum "
            f"{WINDOW_QC_MIN_KEEP_FRACTION:.1%}. "
            "Review the raw recording instead of training on it."
        )

    if (
        removed_fraction
        > WINDOW_QC_WARN_REMOVAL_FRACTION
    ):
        warnings.warn(
            "Window QC removed a large fraction of this recording: "
            f"{n_removed}/{n_windows} "
            f"({removed_fraction:.1%}).",
            RuntimeWarning,
        )

    removed_indices = np.flatnonzero(
        reject
    )

    if starts is not None:
        starts = np.asarray(
            starts,
            dtype=np.float64,
        )
        if len(starts) != n_windows:
            raise ValueError(
                "starts length does not match the number of windows."
            )
        removed_start_sec = [
            float(x)
            for x in starts[reject]
        ]
    else:
        removed_start_sec = []

    # Keep per-removed-window metrics for auditability. This is normally a
    # small list and is stored only in metadata_json.
    removed_details = []

    for idx in removed_indices:
        reasons = []

        if hard_nonfinite[idx]:
            reasons.append("nonfinite")
        if hard_flat[idx]:
            reasons.append("near_flat")
        if hard_amplitude[idx]:
            reasons.append("max_abs")
        if hard_p2p[idx]:
            reasons.append("peak_to_peak")
        if robust_rms[idx]:
            reasons.append("rms_robust_z")
        if robust_amplitude[idx]:
            reasons.append("max_abs_robust_z")
        if robust_p2p[idx]:
            reasons.append("p2p_robust_z")

        removed_details.append({
            "index_before_qc": int(idx),
            "start_sec": (
                float(starts[idx])
                if starts is not None
                else None
            ),
            "rms_uV": float(rms_uv[idx]),
            "max_abs_uV": float(max_abs_uv[idx]),
            "max_p2p_uV": float(max_p2p_uv[idx]),
            "rms_robust_z": float(rms_z[idx]),
            "max_abs_robust_z": float(max_abs_z[idx]),
            "p2p_robust_z": float(p2p_z[idx]),
            "reasons": reasons,
        })

    qc_info = {
        "enabled": True,
        "n_before": n_windows,
        "n_kept": n_kept,
        "n_removed": n_removed,
        "removed_fraction": float(
            removed_fraction
        ),
        "thresholds": {
            "max_abs_uV": WINDOW_QC_MAX_ABS_UV,
            "max_p2p_uV": WINDOW_QC_MAX_P2P_UV,
            "min_rms_uV": WINDOW_QC_MIN_RMS_UV,
            "robust_z_threshold": WINDOW_QC_ROBUST_Z_THRESHOLD,
            "min_windows_for_robust": WINDOW_QC_MIN_WINDOWS_FOR_ROBUST,
            "minimum_keep_fraction": WINDOW_QC_MIN_KEEP_FRACTION,
        },
        "metric_summary_before_qc": {
            "median_rms_uV": float(np.median(rms_uv)),
            "median_max_abs_uV": float(np.median(max_abs_uv)),
            "median_max_p2p_uV": float(np.median(max_p2p_uv)),
            "max_abs_uV": float(np.max(max_abs_uv)),
            "max_p2p_uV": float(np.max(max_p2p_uv)),
        },
        "reason_counts": {
            "nonfinite": int(np.sum(hard_nonfinite)),
            "near_flat": int(np.sum(hard_flat)),
            "max_abs": int(np.sum(hard_amplitude)),
            "peak_to_peak": int(np.sum(hard_p2p)),
            "rms_robust_z": int(np.sum(robust_rms)),
            "max_abs_robust_z": int(np.sum(robust_amplitude)),
            "p2p_robust_z": int(np.sum(robust_p2p)),
        },
        "removed_indices": [
            int(x)
            for x in removed_indices
        ],
        "removed_window_start_sec": removed_start_sec,
        "removed_details": removed_details,
    }

    return keep, qc_info


def _apply_window_mask(
    keep: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Apply one QC keep mask to all per-window arrays."""
    return tuple(
        np.asarray(array)[keep]
        for array in arrays
    )

def compute_pearson_fc(
    windows: np.ndarray,
) -> np.ndarray:
    """
    Pearson FC per 1-s EEG window.

    windows : (N, 64, 250)
    fc      : (N, 64, 64)
    """
    if windows.ndim != 3:
        raise ValueError(
            f"Expected (N,C,T), got {windows.shape}"
        )

    x = np.asarray(
        windows,
        dtype=np.float32,
    )

    x = (
        x
        - x.mean(
            axis=2,
            keepdims=True,
        )
    )

    numerator = np.einsum(
        "nct,ndt->ncd",
        x,
        x,
        optimize=True,
        dtype=np.float32,
    )

    norms = np.sqrt(
        np.sum(
            x * x,
            axis=2,
            dtype=np.float32,
        )
    )

    denominator = (
        norms[:, :, None]
        * norms[:, None, :]
    )

    # float32.eps is NOT an absolute zero threshold.
    valid_channel = (
        norms
        > np.finfo(np.float32).tiny
    )

    valid_pair = (
        valid_channel[:, :, None]
        & valid_channel[:, None, :]
    )

    fc = np.zeros_like(
        numerator,
        dtype=np.float32,
    )

    np.divide(
        numerator,
        denominator,
        out=fc,
        where=valid_pair,
    )

    fc = np.nan_to_num(
        fc,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    np.clip(
        fc,
        -1.0,
        1.0,
        out=fc,
    )

    diag = np.arange(
        fc.shape[1]
    )

    fc[
        :,
        diag,
        diag,
    ] = 0.0

    zero_fraction = float(
        np.mean(
            np.all(
                fc == 0.0,
                axis=(1, 2),
            )
        )
    )

    if zero_fraction > 0.95:
        raise RuntimeError(
            f"FC sanity check failed: {zero_fraction:.1%} "
            "of FC matrices are completely zero."
        )

    return fc


# =============================================================================
# 8. SAVE
# =============================================================================

def save_npz(
    output_path: Path,
    data: np.ndarray,
    fc: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    labels: np.ndarray,
    event_codes: np.ndarray,
    event_block_ids: np.ndarray,
    event_block_starts: np.ndarray,
    event_block_ends: np.ndarray,
    subject_id: str,
    diagnosis: str,
    diagnosis_id: int,
    source_file: str,
    original_sfreq: float,
    mat_info: dict,
    bad_channels: list[str],
    interpolation_log: dict[str, list[str]],
    ica_excluded: list[int],
    ica_converged: bool,
    ica_proxy_channels: list[str],
    window_qc: dict,
) -> None:
    n_windows = int(
        data.shape[0]
    )

    expected_fc_shape = (
        data.shape[0],
        data.shape[1],
        data.shape[1],
    )

    if fc.shape != expected_fc_shape:
        raise ValueError(
            f"FC shape {fc.shape} != expected {expected_fc_shape}"
        )

    diagnosis_labels = np.asarray(
        [diagnosis] * n_windows,
        dtype=np.str_,
    )

    diagnosis_ids = np.full(
        n_windows,
        diagnosis_id,
        dtype=np.int8,
    )

    metadata = {
        "dataset_name": DATASET_NAME,
        "source_file": source_file,
        "subject_id": subject_id,
        "diagnosis": diagnosis,
        "diagnosis_id": diagnosis_id,
        "diagnosis_mapping": {
            "HEALTHY": 0,
            "MDD": 1,
        },
        "diagnosis_source": (
            "official original MODMA filename prefix: "
            "0201=MDD, 0203=HEALTHY"
        ),
        "physiological_state": "Eyes Closed",
        "shape": list(data.shape),
        "fc_shape": list(fc.shape),
        "channel_order": COMMON_64,
        "channel_mapping_source": (
            "HydroCel manufacturer 10-10 equivalence where available; "
            "fixed nearest-position mapping for M1/M2/PO5/PO6/CB1/CB2"
        ),
        "common64_to_hydrocel": COMMON64_TO_HYDROCEL,
        "online_reference": "Cz",
        "cz_reference_retained_for_CAR": True,
        "mat_info": mat_info,
        "mat_input_unit": MAT_INPUT_UNIT,
        "data_unit": OUTPUT_EEG_UNIT,
        "data_scale_from_mne_volts": OUTPUT_EEG_SCALE,
        "original_sfreq": original_sfreq,
        "target_sfreq": TARGET_SFREQ,
        "window_seconds": WINDOW_SECONDS,
        "windowing": (
            "full eyes-closed resting recording; "
            "complete non-overlapping 1-s windows only"
        ),
        "functional_connectivity": (
            "Pearson correlation computed independently "
            "for each 1-s EEG window"
        ),
        "fc_range": [-1.0, 1.0],
        "fc_diagonal": 0.0,
        "bandpass_hz": [
            BANDPASS_LOW,
            BANDPASS_HIGH,
        ],
        "filter": (
            f"{FILTER_ORDER}th-order Butterworth IIR"
        ),
        "notch_hz": DEFAULT_LINE_FREQ,
        "bad_channel_method": (
            "robust z-score of log(SD), "
            f"|z| > {BAD_Z_THRESHOLD}"
        ),
        "bad_channels": bad_channels,
        "interpolation_method": (
            f"{NEIGHBOR_COUNT}-nearest-electrode "
            "inverse-distance weighted average"
        ),
        "interpolation_neighbours": interpolation_log,
        "montage": MONTAGE_NAME,
        "reference": "common average reference",
        "ica_method": (
            "FastICA with frontal EEG proxy scoring"
            if USE_FRONTAL_PROXY_ICA
            else "disabled"
        ),
        "ica_proxy_channels": ica_proxy_channels,
        "ica_max_components": ICA_MAX_COMPONENTS,
        "ica_max_ocular_components_removed": ICA_MAX_OCULAR_COMPONENTS,
        "ica_excluded_components": ica_excluded,
        "ica_converged": bool(
            ica_converged
        ),
        "baseline_correction": (
            "whole-window mean"
            if APPLY_BASELINE
            else "none"
        ),
        "window_qc": window_qc,
    }

    np.savez(
        output_path,
        data=data,
        fc=fc,
        labels=labels,
        diagnosis_labels=diagnosis_labels,
        diagnosis_ids=diagnosis_ids,
        event_codes=event_codes,
        event_block_ids=event_block_ids,
        event_block_start_sec=event_block_starts,
        event_block_end_sec=event_block_ends,
        window_start_sec=starts,
        window_end_sec=ends,
        window_times_sec=np.column_stack(
            [starts, ends]
        ),
        subject_ids=np.asarray(
            [subject_id] * n_windows,
            dtype=np.str_,
        ),
        run_ids=np.asarray(
            ["restEC"] * n_windows,
            dtype=np.str_,
        ),
        dataset_names=np.asarray(
            [DATASET_NAME] * n_windows,
            dtype=np.str_,
        ),
        channel_names=np.asarray(
            COMMON_64,
            dtype=np.str_,
        ),
        sfreq=np.asarray(
            TARGET_SFREQ,
            dtype=np.float32,
        ),
        data_unit=np.asarray(
            OUTPUT_EEG_UNIT,
            dtype=np.str_,
        ),
        source_file=np.asarray(
            source_file,
            dtype=np.str_,
        ),
        metadata_json=np.asarray(
            json.dumps(
                metadata,
                ensure_ascii=False,
            ),
            dtype=np.str_,
        ),
    )


def count_existing_npz_windows(
    path: Path,
) -> int:
    if not path.exists():
        return 0

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as npz:
            return int(
                npz["data"].shape[0]
            )
    except Exception:
        return 0


# =============================================================================
# 9. PROCESS ONE RECORDING
# =============================================================================

def process_one_recording(
    mat_path: Path,
    output_dir: Path,
) -> dict:
    mat_path = Path(mat_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_id = parse_modma_subject_id(
        mat_path
    )

    diagnosis, diagnosis_id = diagnosis_from_filename(
        mat_path
    )

    output_path = (
        output_dir
        / f"{subject_id}_EC.npz"
    )

    if (
        output_path.exists()
        and not OVERWRITE
    ):
        n_windows = count_existing_npz_windows(
            output_path
        )

        return {
            "subject_id": subject_id,
            "diagnosis": diagnosis,
            "source_file": mat_path.name,
            "output_file": output_path.name,
            "status": "SKIPPED_EXISTS",
            "n_windows": n_windows,
            "n_channels": 64,
            "samples_per_window": 250,
            "fc_shape_per_window": "64x64",
        }

    (
        mat_data,
        original_sfreq,
        channel_names,
        mat_info,
    ) = load_modma_mat(
        mat_path
    )

    raw = make_raw_from_mat(
        mat_data,
        original_sfreq,
    )

    # Basic amplitude sanity check in the original MAT numeric unit.
    original_channel_std = np.std(
        mat_data[:128],
        axis=1,
    )

    median_input_std = float(
        np.median(
            original_channel_std[
                np.isfinite(original_channel_std)
            ]
        )
    )

    # Do not silently continue with obviously inconsistent unit settings.
    if MAT_INPUT_UNIT.lower() == "uv" and median_input_std < 1e-3:
        warnings.warn(
            f"{mat_path.name}: median raw channel SD={median_input_std:.3e} "
            "is extremely small for data declared as uV. Verify MAT_INPUT_UNIT.",
            RuntimeWarning,
        )

    raw._data[
        ~np.isfinite(raw._data)
    ] = 0.0

    # 1) 50-Hz notch.
    if (
        DEFAULT_LINE_FREQ
        < raw.info["sfreq"] / 2
    ):
        raw.notch_filter(
            freqs=[
                DEFAULT_LINE_FREQ
            ],
            method="iir",
            verbose="ERROR",
        )

    # 2) 0.5-45 Hz band-pass.
    raw.filter(
        l_freq=BANDPASS_LOW,
        h_freq=BANDPASS_HIGH,
        method="iir",
        iir_params=dict(
            order=FILTER_ORDER,
            ftype="butter",
        ),
        verbose="ERROR",
    )

    # 3) Resample if required.
    if not np.isclose(
        raw.info["sfreq"],
        TARGET_SFREQ,
    ):
        raw.resample(
            TARGET_SFREQ,
            verbose="ERROR",
        )

    # 4) Bad-channel detection.
    bad_channels, bad_z_scores = (
        detect_bad_eeg_channels(
            raw
        )
    )

    # 5) Nearest-electrode interpolation.
    positions = get_eeg_positions(
        raw
    )

    raw, interpolation_log = (
        interpolate_nearest_electrodes(
            raw,
            bad_channels=bad_channels,
            positions=positions,
            k=NEIGHBOR_COUNT,
        )
    )

    # 6) Common-average reference.
    raw.set_eeg_reference(
        ref_channels="average",
        projection=False,
        verbose="ERROR",
    )

    # 7) Conservative frontal-proxy ICA.
    (
        raw,
        ica_excluded,
        ica_converged,
        ica_proxy_channels,
    ) = run_ica_ocular_removal(
        raw
    )

    # 8) COMMON_64 has already been selected and ordered at MAT import.
    raw.pick(
        COMMON_64
    )

    raw.reorder_channels(
        COMMON_64
    )

    # 9) Whole-recording EC -> 1-s windows, export uV.
    (
        data,
        starts,
        ends,
        labels,
        event_codes,
        event_block_ids,
        event_block_starts,
        event_block_ends,
    ) = make_one_second_ec_windows(
        raw
    )

    # 10) Final 1-s window-level artifact rejection.
    n_windows_before_qc = int(
        data.shape[0]
    )

    keep_windows, window_qc = (
        reject_artifact_windows(
            data,
            starts=starts,
        )
    )

    (
        data,
        starts,
        ends,
        labels,
        event_codes,
        event_block_ids,
        event_block_starts,
        event_block_ends,
    ) = _apply_window_mask(
        keep_windows,
        data,
        starts,
        ends,
        labels,
        event_codes,
        event_block_ids,
        event_block_starts,
        event_block_ends,
    )

    n_windows_removed_qc = (
        n_windows_before_qc
        - int(data.shape[0])
    )

    # 11) FC only for windows that survive QC.
    fc = compute_pearson_fc(
        data
    )

    # 12) Save.
    save_npz(
        output_path=output_path,
        data=data,
        fc=fc,
        starts=starts,
        ends=ends,
        labels=labels,
        event_codes=event_codes,
        event_block_ids=event_block_ids,
        event_block_starts=event_block_starts,
        event_block_ends=event_block_ends,
        subject_id=subject_id,
        diagnosis=diagnosis,
        diagnosis_id=diagnosis_id,
        source_file=mat_path.name,
        original_sfreq=original_sfreq,
        mat_info=mat_info,
        bad_channels=bad_channels,
        interpolation_log=interpolation_log,
        ica_excluded=ica_excluded,
        ica_converged=ica_converged,
        ica_proxy_channels=ica_proxy_channels,
        window_qc=window_qc,
    )

    # Output QC summary.
    max_abs_uv = float(
        np.max(
            np.abs(data)
        )
    )

    mean_abs_fc = float(
        np.mean(
            np.abs(fc)
        )
    )

    zero_fc = int(
        np.sum(
            np.all(
                fc == 0.0,
                axis=(1, 2),
            )
        )
    )

    result = {
        "subject_id": subject_id,
        "diagnosis": diagnosis,
        "diagnosis_id": diagnosis_id,
        "source_file": mat_path.name,
        "output_file": output_path.name,
        "status": "OK",
        "original_sfreq": original_sfreq,
        "final_sfreq": TARGET_SFREQ,
        "n_windows_before_qc": n_windows_before_qc,
        "n_windows_removed_qc": n_windows_removed_qc,
        "n_windows": int(
            data.shape[0]
        ),
        "n_channels": int(
            data.shape[1]
        ),
        "samples_per_window": int(
            data.shape[2]
        ),
        "data_unit": OUTPUT_EEG_UNIT,
        "channel_order": "|".join(COMMON_64),
        "channel_mapping_json": json.dumps(
            COMMON64_TO_HYDROCEL,
            ensure_ascii=False,
        ),
        "max_abs_uV": max_abs_uv,
        "fc_shape_per_window": "64x64",
        "fc_method": (
            "Pearson correlation; diagonal=0"
        ),
        "mean_abs_fc": mean_abs_fc,
        "all_zero_fc_matrices": zero_fc,
        "line_freq": DEFAULT_LINE_FREQ,
        "bad_channels": (
            ",".join(
                bad_channels
            )
            if bad_channels
            else ""
        ),
        "bad_z_scores_json": json.dumps(
            {
                ch: bad_z_scores[ch]
                for ch in bad_channels
            },
            ensure_ascii=False,
        ),
        "interpolation_json": json.dumps(
            interpolation_log,
            ensure_ascii=False,
        ),
        "ica_proxy_channels": ",".join(
            ica_proxy_channels
        ),
        "ica_excluded": ",".join(
            map(
                str,
                ica_excluded,
            )
        ),
        "ica_converged": bool(
            ica_converged
        ),
    }

    del raw, data, fc, mat_data
    gc.collect()

    return result


# =============================================================================
# 10. DISCOVERY + BATCH
# =============================================================================

def discover_recordings(
    data_dir: Path,
) -> list[Path]:
    if PROCESS_ONLY_SINGLE_TEST_FILE:
        if not SINGLE_TEST_FILE.exists():
            raise FileNotFoundError(
                f"Single test file not found: {SINGLE_TEST_FILE}"
            )
        return [SINGLE_TEST_FILE]

    recordings = sorted(
        data_dir.glob("*.mat")
    )

    # Keep only official resting EEG subject files whose names encode diagnosis.
    recordings = [
        path
        for path in recordings
        if path.stem.startswith(("0201", "0203"))
    ]

    return recordings


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recordings = discover_recordings(
        DATA_DIR
    )

    print(
        f"Dataset dir : {DATA_DIR}"
    )
    print(
        f"Output dir  : {OUTPUT_DIR}"
    )
    print(
        f"Input unit  : {MAT_INPUT_UNIT}"
    )
    print(
        f"Output unit : {OUTPUT_EEG_UNIT}"
    )
    print(
        f"Recordings  : {len(recordings)}"
    )

    if not recordings:
        raise RuntimeError(
            "No MODMA resting-state MAT recordings were found."
        )

    results = []
    success_count = 0
    skip_count = 0
    error_count = 0

    progress_bar = tqdm(
        recordings,
        total=len(recordings),
        desc="MODMA preprocessing",
        unit="recording",
        dynamic_ncols=True,
        leave=True,
    )

    for mat_path in progress_bar:
        subject_id = mat_path.stem

        progress_bar.set_postfix(
            subject=subject_id,
            OK=success_count,
            SKIP=skip_count,
            ERR=error_count,
            refresh=True,
        )

        try:
            result = process_one_recording(
                mat_path=mat_path,
                output_dir=OUTPUT_DIR,
            )

            results.append(
                result
            )

            if (
                result.get("status")
                == "SKIPPED_EXISTS"
            ):
                skip_count += 1

                tqdm.write(
                    f"[SKIP] {subject_id} | "
                    f"{result['diagnosis']} | "
                    f"{result.get('n_windows', 0)} windows | "
                    "existing NPZ"
                )

            else:
                success_count += 1

                tqdm.write(
                    f"[OK] {subject_id} | "
                    f"{result['diagnosis']} | "
                    f"{result['n_windows']} windows | "
                    f"removed={result.get('n_windows_removed_qc', 0)} | "
                    "EEG=64x250 uV | FC=64x64 | "
                    f"mean|FC|={result['mean_abs_fc']:.4f}"
                )

        except Exception as error:
            error_count += 1

            tqdm.write(
                f"[ERROR] {subject_id} | {error}"
            )

            results.append({
                "subject_id": subject_id,
                "source_file": mat_path.name,
                "status": "ERROR",
                "error": repr(error),
            })

        progress_bar.set_postfix(
            subject=subject_id,
            OK=success_count,
            SKIP=skip_count,
            ERR=error_count,
            refresh=True,
        )

        pd.DataFrame(
            results
        ).to_csv(
            OUTPUT_DIR
            / "MODMA_preprocessing_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    progress_bar.close()

    summary = pd.DataFrame(
        results
    )

    print(
        "\nBatch finished."
    )
    print(
        f"Successful : {success_count}"
    )
    print(
        f"Skipped    : {skip_count}"
    )
    print(
        f"Failed     : {error_count}"
    )
    print(
        f"Total      : {len(recordings)}"
    )

    if not summary.empty:
        print(
            summary["status"].value_counts(
                dropna=False
            )
        )

    print(
        "Summary saved to: "
        f"{OUTPUT_DIR / 'MODMA_preprocessing_summary.csv'}"
    )


if __name__ == "__main__":
    main()
