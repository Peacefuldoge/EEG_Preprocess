#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
TDBRAIN V3.1 batch EEG preprocessing, adapted directly from the finalized
PRED+CT preprocessing script.

Expected structure
------------------
E:\Downloads\TDBRAIN_Dataset_V3_1_Encr\TDBRAIN_Dataset_V3_1\
    sub-88006477\
        ses-1\
            eeg\
                sub-88006477_ses-1_task-restEC_eeg.bdf

Only MDD / HEALTHY recordings listed in the participant spreadsheet are used.

Output
------
One compressed NPZ per TDBRAIN restEC recording:
sub-88006477_ses-1_task-restEC_eeg_EC.npz

Saved arrays
------------
data : (N, 26, 250)
fc   : (N, 26, 26)

Saved EEG amplitudes are in microvolts (uV).

The core preprocessing is intentionally kept aligned with the supplied
PRED+CT script:
    notch -> 0.5-45 Hz Butterworth IIR -> 250 Hz
    -> robust-z bad-channel detection
    -> 3-nearest-electrode interpolation
    -> common-average reference
    -> FastICA ocular removal
    -> 1-s non-overlapping windows
    -> whole-window mean baseline
    -> Pearson FC

TDBRAIN-specific differences
-----------------------------
1. Input format is BDF.
2. Power-line frequency is 50 Hz.
3. Exactly 26 scalp EEG channels are retained.
4. Files are already task-restEC, so the full recording is one EC block.
5. Disease labels come from the spreadsheet:
       formal_status = HEALTHY / MDD
       diagnosis_ids = 0 / 1
6. For consistency with PRED+CT, ``labels`` retains physiological-state
   semantics and is therefore "Eyes Closed"; disease labels are additionally
   stored in ``diagnosis_labels`` and ``diagnosis_ids``.
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
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

DATASET_ROOT = Path(
    r"E:/Downloads/TDBRAIN_Dataset_V3_1_Encr/TDBRAIN_Dataset_V3_1"
)

OUTPUT_DIR = Path(
    r"E:/Workspace/dataset/preprocessed/TDBRAIN"
)

DATASET_NAME = "TDBRAIN"

# Fixed label table used for all runs.
LABEL_TABLE_PATH = Path("E:/Workspace/EEG_Preprocess/TDBRAIN_MDD_HC.csv")

# Preprocessing -- intentionally matched to the supplied PRED+CT script.
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

WINDOW_SECONDS = 1.0
APPLY_BASELINE = True

# MNE internally uses volts. Keep all preprocessing in volts, then
# convert only the final exported EEG windows to microvolts (uV).
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

# Same resume behavior as the supplied PRED+CT script.
OVERWRITE = True


# =============================================================================
# 2. FIXED TDBRAIN 26-CHANNEL ORDER
# =============================================================================

EEG_26 = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8",
    "CP3", "CPz", "CP4",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

# TDBRAIN auxiliary ocular channels are used only by ICA.
EOG_CHANNELS = ["VPVA", "VNVB", "HPHL", "HNHR"]

# Other non-EEG channels in the BDF.
MISC_CHANNELS = ["Erbs", "Mass"]
STIM_CHANNELS = ["Status"]


# =============================================================================
# 3. LABEL-TABLE HELPERS
# =============================================================================

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported label-table format: {path}")


def discover_label_table(root: Path) -> Path:
    """
    Find a spreadsheet containing the columns required by TDBRAIN:
      TDBRAIN_ID, sessID, formal_status.
    """
    required = {"TDBRAIN_ID", "sessID", "formal_status"}

    candidate_dirs = [root, root.parent]
    candidates: list[Path] = []

    for folder in candidate_dirs:
        if not folder.exists():
            continue
        for pattern in ("*.csv", "*.tsv", "*.xlsx", "*.xls"):
            candidates.extend(folder.glob(pattern))

    for path in candidates:
        try:
            table = read_table(path)
        except Exception:
            continue
        if required.issubset(table.columns):
            return path

    raise FileNotFoundError(
        "Could not automatically find the TDBRAIN label spreadsheet. "
        "Set LABEL_TABLE_PATH to the full CSV/TSV/XLSX path."
    )


def load_label_table(
    label_table_path: Path | None,
    dataset_root: Path,
) -> tuple[pd.DataFrame, Path]:
    if label_table_path is None:
        label_table_path = discover_label_table(dataset_root)

    path = Path(label_table_path)
    if not path.exists():
        raise FileNotFoundError(f"Label table not found: {path}")

    table = read_table(path).copy()

    required = {"TDBRAIN_ID", "sessID", "formal_status"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"Label table is missing columns {sorted(missing)}. "
            f"Available columns: {list(table.columns)}"
        )

    table["TDBRAIN_ID"] = table["TDBRAIN_ID"].astype(str).str.strip()
    table["formal_status"] = (
        table["formal_status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    table["sessID"] = pd.to_numeric(
        table["sessID"],
        errors="coerce",
    ).astype("Int64")

    # The current downstream task is strictly MDD vs HEALTHY.
    table = table[
        table["formal_status"].isin(["MDD", "HEALTHY"])
    ].copy()

    if table.empty:
        raise ValueError(
            "The spreadsheet contains no MDD/HEALTHY rows in formal_status."
        )

    return table, path


def build_label_lookup(
    table: pd.DataFrame,
) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}

    for row in table.itertuples(index=False):
        if pd.isna(row.sessID):
            continue

        key = (str(row.TDBRAIN_ID), int(row.sessID))
        diagnosis = str(row.formal_status)

        previous = lookup.get(key)
        if previous is not None and previous != diagnosis:
            raise ValueError(
                f"Conflicting formal_status labels for {key}: "
                f"{previous} vs {diagnosis}"
            )
        lookup[key] = diagnosis

    return lookup


# =============================================================================
# 4. TDBRAIN FILE / CHANNEL HELPERS
# =============================================================================

def parse_subject_and_session_from_filename(
    path: Path,
) -> tuple[str | None, str | None]:
    match = re.search(
        r"(sub-\d+)_(ses-\d+)_task-restEC_eeg",
        path.name,
    )
    if match:
        return match.group(1), match.group(2)
    return None, None


def parse_subject_and_session(
    path: Path,
    raw: mne.io.BaseRaw | None = None,
) -> tuple[str, str]:
    """
    Normal dataset files are parsed from their BIDS filename.

    The Raw-header fallback is useful for validating uploaded BDF examples
    whose original filename may have been replaced by an opaque upload ID.
    """
    subject_id, session_id = parse_subject_and_session_from_filename(path)
    if subject_id is not None and session_id is not None:
        return subject_id, session_id

    if raw is None:
        raw = mne.io.read_raw_bdf(
            path,
            preload=False,
            verbose="ERROR",
        )

    subject_info = raw.info.get("subject_info") or {}
    his_id = subject_info.get("his_id")

    if his_id is None:
        raise ValueError(
            f"Cannot infer TDBRAIN subject ID from filename or BDF header: {path}"
        )

    text = str(his_id).strip()
    if not text.startswith("sub-"):
        text = f"sub-{text}"

    # Uploaded examples can lose the BIDS filename. TDBRAIN files in the
    # user's actual dataset retain ses-* in their filenames, so ses-1 here is
    # only a conservative validation fallback.
    return text, "ses-1"


def normalize_aux_channel_names(raw: mne.io.BaseRaw) -> None:
    canonical = {
        "ERBS": "Erbs",
        "MASS": "Mass",
        "STATUS": "Status",
    }

    rename = {}
    for ch in raw.ch_names:
        upper = str(ch).upper()
        if upper in canonical and ch != canonical[upper]:
            rename[ch] = canonical[upper]

    if rename:
        raw.rename_channels(rename)


def validate_and_prepare_channel_types(
    raw: mne.io.BaseRaw,
) -> mne.io.BaseRaw:
    normalize_aux_channel_names(raw)

    missing = [ch for ch in EEG_26 if ch not in raw.ch_names]
    if missing:
        raise ValueError(
            f"Missing required TDBRAIN EEG channels: {missing}"
        )

    type_map = {ch: "eeg" for ch in EEG_26}

    for ch in EOG_CHANNELS:
        if ch in raw.ch_names:
            type_map[ch] = "eog"

    for ch in MISC_CHANNELS:
        if ch in raw.ch_names:
            type_map[ch] = "misc"

    for ch in STIM_CHANNELS:
        if ch in raw.ch_names:
            type_map[ch] = "stim"

    raw.set_channel_types(
        type_map,
        verbose="ERROR",
    )

    # All 26 retained EEG channels exist in the standard 10-20 montage.
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(
        montage,
        on_missing="ignore",
        verbose="ERROR",
    )

    return raw


def get_eeg_positions(
    raw: mne.io.BaseRaw,
) -> dict[str, np.ndarray]:
    montage = raw.get_montage()
    if montage is None:
        raise RuntimeError(
            "No montage is available for nearest-electrode interpolation."
        )

    montage_positions = montage.get_positions()["ch_pos"]
    positions: dict[str, np.ndarray] = {}

    for ch in EEG_26:
        if ch not in montage_positions:
            continue
        xyz = np.asarray(
            montage_positions[ch],
            dtype=float,
        )
        if np.all(np.isfinite(xyz)):
            positions[ch] = xyz

    missing = [ch for ch in EEG_26 if ch not in positions]
    if missing:
        raise RuntimeError(
            f"Missing standard coordinates for TDBRAIN channels: {missing}"
        )

    return positions


# =============================================================================
# 5. PREPROCESSING HELPERS -- MATCHED TO PRED+CT
# =============================================================================

def detect_bad_eeg_channels(
    raw: mne.io.BaseRaw,
) -> tuple[list[str], dict[str, float]]:
    """
    Same method as the supplied PRED+CT script:
      robust z-score of log10(channel SD)
      first 30 seconds
      flat channel OR abs(z) > 6
    """
    eeg = raw.copy().pick(EEG_26)

    if eeg.times[-1] > 30.0:
        eeg.crop(
            tmin=0.0,
            tmax=30.0,
            include_tmax=False,
        )

    data = eeg.get_data()

    channel_std = np.std(data, axis=1)
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

    bads = [
        eeg.ch_names[i]
        for i in np.where(bad_mask)[0]
    ]

    z_scores = {
        eeg.ch_names[i]: float(robust_z[i])
        for i in range(len(eeg.ch_names))
    }

    return bads, z_scores


def interpolate_nearest_electrodes(
    raw: mne.io.BaseRaw,
    bad_channels: list[str],
    positions: dict[str, np.ndarray],
    k: int = NEIGHBOR_COUNT,
) -> tuple[mne.io.BaseRaw, dict[str, list[str]]]:
    """
    Same simple interpolation strategy as the PRED+CT script:
    inverse-distance weighted average of k nearest good electrodes.
    """
    if not bad_channels:
        return raw, {}

    interpolation_log: dict[str, list[str]] = {}
    good_channels = [
        ch for ch in EEG_26
        if ch not in bad_channels
    ]

    for bad in bad_channels:
        if bad not in raw.ch_names:
            continue

        candidates = []

        for good in good_channels:
            distance = float(
                np.linalg.norm(
                    positions[good] - positions[bad]
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
                f"{len(selected)} valid neighbouring electrodes."
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

        neighbour_data = raw._data[
            neighbour_idx,
            :
        ]

        raw._data[bad_idx, :] = np.average(
            neighbour_data,
            axis=0,
            weights=weights,
        )

        interpolation_log[bad] = neighbours

    return raw, interpolation_log


def run_ica_ocular_removal(
    raw: mne.io.BaseRaw,
) -> tuple[mne.io.BaseRaw, list[int], bool]:
    """
    Adapted directly from PRED+CT ICA logic.

    TDBRAIN uses VPVA/VNVB/HPHL/HNHR as EOG reference channels.
    Fit on the first <=30 s and remove at most two ocular components.
    """
    eog_channels = [
        ch for ch in EOG_CHANNELS
        if ch in raw.ch_names
    ]

    if not eog_channels:
        return raw, [], True

    fit_raw = raw.copy()

    if fit_raw.times[-1] > ICA_FIT_SECONDS:
        fit_raw.crop(
            tmin=0.0,
            tmax=ICA_FIT_SECONDS,
            include_tmax=False,
        )

    n_components = min(
        ICA_MAX_COMPONENTS,
        len(EEG_26) - 1,
    )

    if n_components < 2:
        return raw, [], True

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
        return raw, [], False

    candidate_scores: dict[int, float] = {}

    for eog_name in eog_channels:
        try:
            inds, scores = ica.find_bads_eog(
                fit_raw,
                ch_name=eog_name,
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
                    candidate_scores.get(
                        idx,
                        0.0,
                    ),
                    score,
                )

        except Exception:
            # Same principle as PRED+CT:
            # one problematic EOG channel should not abort a recording.
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

    return raw, exclude, True


# =============================================================================
# 6. WINDOWING + FUNCTIONAL CONNECTIVITY
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
    TDBRAIN task-restEC BDF is already one Eyes Closed recording.

    We therefore treat the full usable file as one EC block and retain only
    complete non-overlapping 1-s windows. The final <1-s remainder, if any,
    is discarded.

    Return structure intentionally mirrors the PRED+CT event-aware function.
    """
    eeg = raw.copy().pick(EEG_26)
    eeg.reorder_channels(EEG_26)

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
        len(EEG_26),
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

    # MNE data are in volts. Export final windows in microvolts.
    data = data * OUTPUT_EEG_SCALE

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

    # PRED+CT-compatible placeholder event metadata.
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
    Copied conceptually from the supplied PRED+CT script.

    windows : (N, C, T)
    fc      : (N, C, C)

    Pearson r in [-1, 1], diagonal explicitly set to 0.
    Undefined zero-variance correlations are safely replaced with 0.
    """
    if windows.ndim != 3:
        raise ValueError(
            f"Expected windows with shape (N, C, T), "
            f"got {windows.shape}"
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

    # Do not use float32.eps as an absolute zero threshold.
    # A channel is invalid only when its norm is effectively zero.
    valid_channel = (
        norms
        > np.finfo(
            np.float32
        ).tiny
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

    # Prevent silent generation of broken all-zero FC batches.
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
# 7. SAVE -- PRED+CT-COMPATIBLE STRUCTURE + DIAGNOSIS
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
    session_id: str,
    diagnosis: str,
    source_file: str,
    bad_channels: list[str],
    interpolation_log: dict[str, list[str]],
    ica_excluded: list[int],
    ica_converged: bool,
    label_table_path: Path,
    window_qc: dict,
) -> None:
    n_windows = data.shape[0]

    if fc.shape != (
        data.shape[0],
        data.shape[1],
        data.shape[1],
    ):
        raise ValueError(
            f"FC shape {fc.shape} does not match "
            f"EEG data shape {data.shape}."
        )

    diagnosis_id = (
        1 if diagnosis == "MDD"
        else 0
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
        "session_id": session_id,
        "shape": list(data.shape),
        "fc_shape": list(fc.shape),
        "data_unit": OUTPUT_EEG_UNIT,
        "data_scale_from_mne_volts": OUTPUT_EEG_SCALE,
        "functional_connectivity": (
            "Pearson correlation computed independently "
            "for each 1-s EEG window"
        ),
        "fc_range": [-1.0, 1.0],
        "fc_diagonal": 0.0,
        "target_sfreq": TARGET_SFREQ,
        "window_seconds": WINDOW_SECONDS,
        "windowing": (
            "full task-restEC recording; complete "
            "non-overlapping 1-s windows only"
        ),
        "physiological_state": "Eyes Closed",
        "diagnosis": diagnosis,
        "diagnosis_id": diagnosis_id,
        "diagnosis_mapping": {
            "HEALTHY": 0,
            "MDD": 1,
        },
        "label_source_file": str(
            label_table_path
        ),
        "label_source_columns": [
            "TDBRAIN_ID",
            "sessID",
            "formal_status",
        ],
        "bandpass_hz": [
            BANDPASS_LOW,
            BANDPASS_HIGH,
        ],
        "filter": (
            f"{FILTER_ORDER}th-order "
            "Butterworth IIR"
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
        "interpolation_neighbours": (
            interpolation_log
        ),
        "reference": (
            "common average reference"
        ),
        "ica_method": "FastICA",
        "ica_max_components": (
            ICA_MAX_COMPONENTS
        ),
        "ica_max_ocular_components_removed": (
            ICA_MAX_OCULAR_COMPONENTS
        ),
        "ica_eog_channels": EOG_CHANNELS,
        "ica_excluded_components": (
            ica_excluded
        ),
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

    np.savez_compressed(
        output_path,

        # Identical core arrays to PRED+CT.
        data=data,
        fc=fc,
        labels=labels,
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

        # PRED+CT has run_ids; TDBRAIN uses session instead.
        # run_ids is retained for loader compatibility.
        run_ids=np.asarray(
            [session_id] * n_windows,
            dtype=np.str_,
        ),
        session_ids=np.asarray(
            [session_id] * n_windows,
            dtype=np.str_,
        ),

        dataset_names=np.asarray(
            [DATASET_NAME] * n_windows,
            dtype=np.str_,
        ),
        channel_names=np.asarray(
            EEG_26,
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

        # TDBRAIN-specific supervised target.
        diagnosis_labels=diagnosis_labels,
        diagnosis_ids=diagnosis_ids,
    )


def count_existing_npz_windows(
    path: Path,
) -> int:
    """
    Same resume helper used by PRED+CT.
    """
    if not path.exists():
        return 0

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as npz:
            if "data" in npz:
                return int(
                    npz["data"].shape[0]
                )
    except Exception:
        pass

    return 0


# =============================================================================
# 8. PROCESS ONE TDBRAIN RECORDING
# =============================================================================

def process_one_recording(
    bdf_path: Path,
    output_dir: Path,
    label_lookup: dict[tuple[str, int], str],
    label_table_path: Path,
) -> dict:
    bdf_path = Path(bdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Read BDF first; this also enables the validation fallback when an
    # uploaded file has lost its original BIDS filename.
    raw = mne.io.read_raw_bdf(
        bdf_path,
        preload=True,
        verbose="ERROR",
    )

    original_sfreq = float(
        raw.info["sfreq"]
    )

    subject_id, session_id = (
        parse_subject_and_session(
            bdf_path,
            raw=raw,
        )
    )

    session_match = re.fullmatch(
        r"ses-(\d+)",
        session_id,
    )

    if not session_match:
        raise ValueError(
            f"Unexpected TDBRAIN session: {session_id}"
        )

    session_number = int(
        session_match.group(1)
    )

    diagnosis = label_lookup.get(
        (
            subject_id,
            session_number,
        )
    )

    if diagnosis not in {
        "MDD",
        "HEALTHY",
    }:
        raise KeyError(
            "No MDD/HEALTHY formal_status found for "
            f"{subject_id} {session_id}"
        )

    output_path = (
        output_dir
        / (
            f"{subject_id}_{session_id}"
            "_task-restEC_eeg_EC.npz"
        )
    )

    if (
        output_path.exists()
        and not OVERWRITE
    ):
        count = count_existing_npz_windows(
            output_path
        )

        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "diagnosis": diagnosis,
            "source_file": bdf_path.name,
            "output_file": output_path.name,
            "status": "SKIPPED_EXISTS",
            "n_windows": count,
            "n_channels": len(EEG_26),
            "samples_per_window": int(
                round(
                    WINDOW_SECONDS
                    * TARGET_SFREQ
                )
            ),
            "fc_shape_per_window": "26x26",
        }

    raw = validate_and_prepare_channel_types(
        raw
    )

    # Same numerical-safety step as PRED+CT.
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

    # 3) Resample to 250 Hz.
    if not np.isclose(
        raw.info["sfreq"],
        TARGET_SFREQ,
    ):
        raw.resample(
            TARGET_SFREQ,
            verbose="ERROR",
        )

    # 4) Same robust-z bad-channel detection as PRED+CT.
    bad_channels, bad_z_scores = (
        detect_bad_eeg_channels(
            raw
        )
    )

    # 5) Same simple nearest-electrode interpolation.
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

    # 6) Common average reference.
    raw.set_eeg_reference(
        ref_channels="average",
        projection=False,
        verbose="ERROR",
    )

    # 7) FastICA ocular removal.
    raw, ica_excluded, ica_converged = (
        run_ica_ocular_removal(
            raw
        )
    )

    # 8) Final fixed 26 EEG channels only.
    raw.pick(
        EEG_26
    )
    raw.reorder_channels(
        EEG_26
    )

    # 9) Whole-file EC segmentation into complete 1-s windows.
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

    # 11) Pearson FC only for windows that survive QC.
    fc = compute_pearson_fc(
        data
    )

    # 12) Save in a PRED+CT-compatible NPZ structure.
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
        session_id=session_id,
        diagnosis=diagnosis,
        source_file=bdf_path.name,
        bad_channels=bad_channels,
        interpolation_log=interpolation_log,
        ica_excluded=ica_excluded,
        ica_converged=ica_converged,
        label_table_path=label_table_path,
        window_qc=window_qc,
    )

    result = {
        "subject_id": subject_id,
        "session_id": session_id,
        "diagnosis": diagnosis,
        "diagnosis_id": (
            1 if diagnosis == "MDD"
            else 0
        ),
        "source_file": bdf_path.name,
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
        "fc_shape_per_window": (
            f"{data.shape[1]}x"
            f"{data.shape[1]}"
        ),
        "fc_method": (
            "Pearson correlation; "
            "diagonal=0"
        ),
        "line_freq": (
            DEFAULT_LINE_FREQ
        ),
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

    del raw, data, fc
    gc.collect()

    return result


# =============================================================================
# 9. DISCOVER LABELLED TDBRAIN REST-EC RECORDINGS
# =============================================================================

def discover_recordings(
    root: Path,
    label_lookup: dict[
        tuple[str, int],
        str,
    ],
) -> list[Path]:
    """
    Process only subjects/sessions present in the MDD/HEALTHY spreadsheet.
    """
    recordings: list[Path] = []

    for (
        subject_id,
        session_number,
    ), diagnosis in sorted(
        label_lookup.items()
    ):
        if diagnosis not in {
            "MDD",
            "HEALTHY",
        }:
            continue

        session_id = (
            f"ses-{session_number}"
        )

        bdf_path = (
            root
            / subject_id
            / session_id
            / "eeg"
            / (
                f"{subject_id}_"
                f"{session_id}_"
                "task-restEC_eeg.bdf"
            )
        )

        if not bdf_path.exists():
            print(
                "[WARN] Missing restEC BDF: "
                f"{bdf_path}"
            )
            continue

        recordings.append(
            bdf_path
        )

    return recordings


# =============================================================================
# 10. BATCH PROCESSING WITH TQDM
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_table, label_table_path = (
        load_label_table(
            LABEL_TABLE_PATH,
            DATASET_ROOT,
        )
    )

    label_lookup = (
        build_label_lookup(
            label_table
        )
    )

    recordings = discover_recordings(
        DATASET_ROOT,
        label_lookup,
    )

    diagnosis_counts = (
        label_table[
            "formal_status"
        ]
        .value_counts()
        .to_dict()
    )

    print(
        f"Dataset root : {DATASET_ROOT}"
    )
    print(
        f"Output dir   : {OUTPUT_DIR}"
    )
    print(
        f"Label table  : {label_table_path}"
    )
    print(
        f"Labels       : {diagnosis_counts}"
    )
    print(
        f"Recordings   : {len(recordings)}"
    )

    if not recordings:
        raise RuntimeError(
            "No labelled TDBRAIN restEC "
            "recordings were found."
        )

    results = []
    success_count = 0
    skip_count = 0
    error_count = 0

    progress_bar = tqdm(
        recordings,
        total=len(recordings),
        desc="TDBRAIN preprocessing",
        unit="recording",
        dynamic_ncols=True,
        leave=True,
    )

    for bdf_path in progress_bar:
        subject_id, session_id = (
            parse_subject_and_session_from_filename(
                bdf_path
            )
        )

        subject_id = (
            subject_id or "unknown"
        )
        session_id = (
            session_id or "unknown"
        )

        progress_bar.set_postfix(
            subject=subject_id,
            session=session_id,
            OK=success_count,
            SKIP=skip_count,
            ERR=error_count,
            refresh=True,
        )

        try:
            result = (
                process_one_recording(
                    bdf_path=bdf_path,
                    output_dir=OUTPUT_DIR,
                    label_lookup=label_lookup,
                    label_table_path=label_table_path,
                )
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
                    f"[SKIP] "
                    f"{result['subject_id']} "
                    f"{result['session_id']} | "
                    f"{result['diagnosis']} | "
                    f"{result.get('n_windows', 0)} windows | "
                    "existing NPZ"
                )

            else:
                success_count += 1

                tqdm.write(
                    f"[OK] "
                    f"{result['subject_id']} "
                    f"{result['session_id']} | "
                    f"{result['diagnosis']} | "
                    f"{result['n_windows']} windows | "
                    f"removed={result.get('n_windows_removed_qc', 0)} | "
                    "EEG=26x250 uV | FC=26x26"
                )

        except Exception as error:
            error_count += 1

            tqdm.write(
                f"[ERROR] "
                f"{subject_id} "
                f"{session_id} | "
                f"{error}"
            )

            results.append(
                {
                    "subject_id": (
                        subject_id
                    ),
                    "session_id": (
                        session_id
                    ),
                    "source_file": (
                        bdf_path.name
                    ),
                    "status": "ERROR",
                    "error": repr(error),
                }
            )

        progress_bar.set_postfix(
            subject=subject_id,
            session=session_id,
            OK=success_count,
            SKIP=skip_count,
            ERR=error_count,
            refresh=True,
        )

        # Same continuous logging behavior as PRED+CT.
        pd.DataFrame(
            results
        ).to_csv(
            OUTPUT_DIR
            / "TDBRAIN_preprocessing_summary.csv",
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
            summary[
                "status"
            ].value_counts(
                dropna=False
            )
        )

    print(
        "Summary saved to: "
        f"{OUTPUT_DIR / 'TDBRAIN_preprocessing_summary.csv'}"
    )


if __name__ == "__main__":
    main()
