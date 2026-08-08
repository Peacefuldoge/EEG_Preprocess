#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
HBN resting-state EEG preprocessing, adapted from the user's finalized
MODMA preprocessing script.

Dataset example
---------------
E:\Downloads\ds005505\sub-NDARAC904DMU\eeg\
    sub-NDARAC904DMU_task-RestingState_eeg.set
    sub-NDARAC904DMU_task-RestingState_channels.tsv
    sub-NDARAC904DMU_task-RestingState_eeg.json
    sub-NDARAC904DMU_task-RestingState_events.tsv

Input
-----
EEGLAB .set, HydroCel-129 layout:
    E1 ... E128 + Cz

Output
------
Two NPZ files per subject:
    sub-NDARAC904DMU_task-RestingState_eeg_EO.npz
    sub-NDARAC904DMU_task-RestingState_eeg_EC.npz

Each file contains:
    data : (N, 64, 250), float32, uV
    fc   : (N, 64, 64), float32, Pearson r

The public 64-channel order and HydroCel mapping are exactly the same as in
the user's finalized MODMA script.

Preprocessing
-------------
SET -> common-64 HydroCel mapping
-> 60-Hz notch
-> 0.5-45 Hz 4th-order Butterworth IIR
-> 500 -> 250 Hz
-> robust-z bad-channel detection
-> 3-nearest-electrode interpolation
-> common-average reference
-> optional conservative frontal-proxy FastICA
-> event-aware EO/EC blocks from events.tsv
-> complete non-overlapping 1-s windows within each block
-> whole-window mean baseline
-> export uV
-> window-level artifact QC
-> Pearson FC
-> separate EO and EC NPZ files

Important event rule
--------------------
The HBN resting task begins at ``resting_start``. The initial state before the
first ``instructed_toOpenEyes`` marker is treated as Eyes Closed. Subsequent
``instructed_toOpenEyes`` and ``instructed_toCloseEyes`` markers switch the
current physiological state. The resting block ends at the first ``break cnt``
after ``resting_start``. No saved 1-s window is allowed to cross a state boundary.
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

DATASET_ROOT = Path(r"E:/Downloads/ds005505")
OUTPUT_DIR = Path(r"E:/Workspace/dataset/preprocessed/HBN")

DATASET_NAME = "HBN"

TARGET_SFREQ = 250.0
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0
DEFAULT_LINE_FREQ = 60.0
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

# HBN resting-state SET has no dedicated EOG channels in the 129-channel file.
ICA_FRONTAL_PROXY_CHANNELS = ["Fp1", "Fpz", "Fp2"]
USE_FRONTAL_PROXY_ICA = False

WINDOW_SECONDS = 1.0
APPLY_BASELINE = True

# MNE reads EEGLAB EEG internally in volts; save final EEG in uV.
OUTPUT_EEG_SCALE = 1e6
OUTPUT_EEG_UNIT = "uV"

# Window-level artifact QC -- copied from the finalized MODMA script.
WINDOW_QC_ENABLED = True
WINDOW_QC_MAX_ABS_UV = 200.0
WINDOW_QC_MAX_P2P_UV = 400.0
WINDOW_QC_MIN_RMS_UV = 0.05
WINDOW_QC_ROBUST_Z_THRESHOLD = 8.0
WINDOW_QC_MIN_WINDOWS_FOR_ROBUST = 20
WINDOW_QC_WARN_REMOVAL_FRACTION = 0.20
WINDOW_QC_MIN_KEEP_FRACTION = 0.50

OVERWRITE = True


# =============================================================================
# 2. FIXED 64-CHANNEL HARMONIZED CONFIGURATION
# =============================================================================

COMMON_64 = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2']

COMMON64_TO_HYDROCEL = {'Fp1': 'E22', 'Fpz': 'E14', 'Fp2': 'E9', 'AF3': 'E23', 'AF4': 'E3', 'F7': 'E33', 'F5': 'E27', 'F3': 'E24', 'F1': 'E19', 'Fz': 'E11', 'F2': 'E4', 'F4': 'E124', 'F6': 'E123', 'F8': 'E122', 'FT7': 'E34', 'FC5': 'E28', 'FC3': 'E29', 'FC1': 'E13', 'FCz': 'E6', 'FC2': 'E112', 'FC4': 'E111', 'FC6': 'E117', 'FT8': 'E116', 'T7': 'E45', 'C5': 'E41', 'C3': 'E36', 'C1': 'E30', 'Cz': 'Cz', 'C2': 'E105', 'C4': 'E104', 'C6': 'E103', 'T8': 'E108', 'M1': 'E56', 'TP7': 'E46', 'CP5': 'E47', 'CP3': 'E42', 'CP1': 'E37', 'CPz': 'E55', 'CP2': 'E87', 'CP4': 'E93', 'CP6': 'E98', 'TP8': 'E102', 'M2': 'E107', 'P7': 'E58', 'P5': 'E51', 'P3': 'E52', 'P1': 'E60', 'Pz': 'E62', 'P2': 'E85', 'P4': 'E92', 'P6': 'E97', 'P8': 'E96', 'PO7': 'E65', 'PO5': 'E66', 'PO3': 'E67', 'POz': 'E72', 'PO4': 'E77', 'PO6': 'E84', 'PO8': 'E90', 'CB1': 'E69', 'O1': 'E70', 'Oz': 'E75', 'O2': 'E83', 'CB2': 'E89'}

if list(COMMON64_TO_HYDROCEL.keys()) != COMMON_64:
    raise RuntimeError("COMMON64_TO_HYDROCEL keys must match COMMON_64 order.")

if len(set(COMMON64_TO_HYDROCEL.values())) != len(COMMON_64):
    raise RuntimeError("COMMON64_TO_HYDROCEL must be one-to-one.")

MONTAGE_NAME = "GSN-HydroCel-129"
ONLINE_REFERENCE_TARGET = "Cz"


# =============================================================================
# 3. FILE / SIDECAR HELPERS
# =============================================================================

def parse_subject_id(set_path: Path) -> str:
    match = re.search(r"(sub-[A-Za-z0-9]+)", set_path.name)
    return match.group(1) if match else set_path.stem


def sidecar_paths(set_path: Path) -> dict[str, Path]:
    stem = set_path.stem  # sub-..._task-RestingState_eeg
    prefix = stem[:-4] if stem.endswith("_eeg") else stem
    folder = set_path.parent

    return {
        "channels": folder / f"{prefix}_channels.tsv",
        "eeg_json": folder / f"{stem}.json",
        "events": folder / f"{prefix}_events.tsv",
    }


def get_line_frequency(eeg_json_path: Path) -> float:
    if eeg_json_path.exists():
        with open(eeg_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        try:
            return float(meta.get("PowerLineFrequency", DEFAULT_LINE_FREQ))
        except Exception:
            pass
    return float(DEFAULT_LINE_FREQ)


# =============================================================================
# 4. LOAD SET + MAP HYDROCEL-129 TO COMMON-64
# =============================================================================

def load_hbn_set(set_path: Path) -> tuple[mne.io.BaseRaw, float, dict]:
    raw_source = mne.io.read_raw_eeglab(
        set_path,
        preload=True,
        verbose="ERROR",
    )

    original_sfreq = float(raw_source.info["sfreq"])

    required_source = set(COMMON64_TO_HYDROCEL.values())
    missing = sorted(required_source - set(raw_source.ch_names))

    if missing:
        raise ValueError(
            f"Missing HydroCel source channels in {set_path.name}: {missing}"
        )

    selected_source_names = [
        COMMON64_TO_HYDROCEL[target]
        for target in COMMON_64
    ]

    raw_source.pick(selected_source_names)
    raw_source.reorder_channels(selected_source_names)

    selected_data_v = raw_source.get_data().astype(np.float64)

    info = mne.create_info(
        ch_names=COMMON_64,
        sfreq=original_sfreq,
        ch_types=["eeg"] * len(COMMON_64),
    )

    raw = mne.io.RawArray(
        selected_data_v,
        info,
        verbose="ERROR",
    )

    # Preserve actual HydroCel sensor coordinates while exposing common names.
    hydrocel_montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    hydro_pos = hydrocel_montage.get_positions()["ch_pos"]

    common_positions = {
        target: np.asarray(
            hydro_pos[COMMON64_TO_HYDROCEL[target]],
            dtype=float,
        )
        for target in COMMON_64
    }

    common_montage = mne.channels.make_dig_montage(
        ch_pos=common_positions,
        coord_frame="head",
    )

    raw.set_montage(
        common_montage,
        on_missing="raise",
        verbose="ERROR",
    )

    raw._data[~np.isfinite(raw._data)] = 0.0

    info_dict = {
        "source_channel_count": int(len(mne.io.read_raw_eeglab(
            set_path, preload=False, verbose="ERROR"
        ).ch_names)),
        "selected_channel_count": 64,
        "original_sfreq": original_sfreq,
        "common64_to_hydrocel": COMMON64_TO_HYDROCEL,
    }

    return raw, original_sfreq, info_dict


# =============================================================================
# 5. EVENT BLOCKS
# =============================================================================

def load_hbn_rest_blocks(
    events_path: Path,
    recording_end_sec: float,
) -> pd.DataFrame:
    """
    Construct non-overlapping Eyes Closed / Eyes Open blocks.

    HBN protocol interpretation:
      resting_start -> initial Eyes Closed
      instructed_toOpenEyes  -> switch to Eyes Open
      instructed_toCloseEyes -> switch to Eyes Closed
      first break cnt after resting_start -> resting task ends
    """
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events.tsv: {events_path}")

    events = pd.read_csv(events_path, sep="\t")

    required = {"onset", "value"}
    missing = required - set(events.columns)

    if missing:
        raise ValueError(
            f"events.tsv missing columns {sorted(missing)}; "
            f"available={list(events.columns)}"
        )

    events = events.copy()
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events = events[events["onset"].notna()].sort_values("onset").reset_index(drop=True)

    start_rows = events[events["value"].astype(str) == "resting_start"]

    if start_rows.empty:
        raise ValueError("No resting_start marker found.")

    rest_start = float(start_rows.iloc[0]["onset"])

    after_start = events[events["onset"] > rest_start]

    break_rows = after_start[
        after_start["value"].astype(str).str.lower().eq("break cnt")
    ]

    rest_end = (
        float(break_rows.iloc[0]["onset"])
        if not break_rows.empty
        else float(recording_end_sec)
    )

    transitions = events[
        (events["onset"] >= rest_start)
        & (events["onset"] < rest_end)
        & events["value"].astype(str).isin([
            "instructed_toOpenEyes",
            "instructed_toCloseEyes",
        ])
    ][["onset", "value"]].copy()

    blocks = []

    current_state = "Eyes Closed"
    current_start = rest_start
    counters = {"Eyes Open": 0, "Eyes Closed": 0}

    for row in transitions.itertuples(index=False):
        transition_time = float(row.onset)

        if transition_time > current_start:
            counters[current_state] += 1
            short = "EO" if current_state == "Eyes Open" else "EC"
            blocks.append({
                "event_label": current_state,
                "event_code": 20 if current_state == "Eyes Open" else 30,
                "event_block_id": f"{short}_block_{counters[current_state]:02d}",
                "start_sec": current_start,
                "end_sec": transition_time,
            })

        new_state = (
            "Eyes Open"
            if str(row.value) == "instructed_toOpenEyes"
            else "Eyes Closed"
        )

        current_state = new_state
        current_start = transition_time

    if rest_end > current_start:
        counters[current_state] += 1
        short = "EO" if current_state == "Eyes Open" else "EC"
        blocks.append({
            "event_label": current_state,
            "event_code": 20 if current_state == "Eyes Open" else 30,
            "event_block_id": f"{short}_block_{counters[current_state]:02d}",
            "start_sec": current_start,
            "end_sec": rest_end,
        })

    blocks = pd.DataFrame(blocks)

    if blocks.empty:
        raise ValueError("No HBN resting-state EO/EC blocks were constructed.")

    blocks["duration_sec"] = blocks["end_sec"] - blocks["start_sec"]

    # Only blocks capable of providing at least one full 1-s window.
    blocks = blocks[
        blocks["duration_sec"] + 1e-9 >= WINDOW_SECONDS
    ].copy().reset_index(drop=True)

    if blocks.empty:
        raise ValueError("No HBN resting block is at least 1 second long.")

    return blocks


# =============================================================================
# 6. BAD CHANNELS + INTERPOLATION
# =============================================================================

def detect_bad_eeg_channels(raw: mne.io.BaseRaw) -> tuple[list[str], dict[str, float]]:
    eeg = raw.copy().pick(COMMON_64)

    if eeg.times[-1] > 30.0:
        eeg.crop(tmin=0.0, tmax=30.0, include_tmax=False)

    data = eeg.get_data()
    channel_std = np.std(data, axis=1)
    log_std = np.log10(channel_std + np.finfo(float).eps)

    median = np.median(log_std)
    mad = np.median(np.abs(log_std - median))

    if mad < np.finfo(float).eps:
        robust_z = np.zeros_like(log_std)
    else:
        robust_z = 0.67448975 * (log_std - median) / mad

    flat = channel_std < 1e-12
    bad_mask = (
        flat
        | ~np.isfinite(channel_std)
        | (np.abs(robust_z) > BAD_Z_THRESHOLD)
    )

    # Cz is the online reference and may legitimately be flat before CAR.
    if ONLINE_REFERENCE_TARGET in eeg.ch_names:
        idx = eeg.ch_names.index(ONLINE_REFERENCE_TARGET)
        bad_mask[idx] = False
        robust_z[idx] = 0.0

    bads = [
        eeg.ch_names[i]
        for i in np.where(bad_mask)[0]
    ]

    z_scores = {
        eeg.ch_names[i]: float(robust_z[i])
        for i in range(len(eeg.ch_names))
    }

    return bads, z_scores


def get_eeg_positions(raw: mne.io.BaseRaw) -> dict[str, np.ndarray]:
    montage = raw.get_montage()

    if montage is None:
        raise RuntimeError("No montage available for interpolation.")

    montage_positions = montage.get_positions()["ch_pos"]

    positions = {
        ch: np.asarray(montage_positions[ch], dtype=float)
        for ch in COMMON_64
        if ch in montage_positions
        and np.all(np.isfinite(montage_positions[ch]))
    }

    missing = [ch for ch in COMMON_64 if ch not in positions]

    if missing:
        raise RuntimeError(
            f"Missing selected-channel coordinates: {missing}"
        )

    return positions


def interpolate_nearest_electrodes(
    raw: mne.io.BaseRaw,
    bad_channels: list[str],
    positions: dict[str, np.ndarray],
    k: int = NEIGHBOR_COUNT,
) -> tuple[mne.io.BaseRaw, dict[str, list[str]]]:

    if not bad_channels:
        return raw, {}

    interpolation_log = {}
    good_channels = [
        ch for ch in COMMON_64
        if ch not in bad_channels
    ]

    # Preserve original data so one replaced bad channel cannot affect another.
    original = raw.get_data().copy()

    for bad in bad_channels:
        candidates = []

        for good in good_channels:
            distance = float(
                np.linalg.norm(positions[good] - positions[bad])
            )

            if np.isfinite(distance) and distance > 0:
                candidates.append((good, distance))

        candidates.sort(key=lambda item: item[1])
        selected = candidates[:k]

        if len(selected) < k:
            raise RuntimeError(
                f"Cannot interpolate {bad}: only {len(selected)} valid neighbours."
            )

        neighbours = [name for name, _ in selected]
        distances = np.asarray(
            [distance for _, distance in selected],
            dtype=float,
        )

        weights = 1.0 / np.maximum(distances, 1e-12)
        weights /= weights.sum()

        bad_idx = raw.ch_names.index(bad)
        neighbour_idx = [raw.ch_names.index(ch) for ch in neighbours]

        raw._data[bad_idx, :] = np.average(
            original[neighbour_idx, :],
            axis=0,
            weights=weights,
        )

        interpolation_log[bad] = neighbours

    return raw, interpolation_log


# =============================================================================
# 7. OPTIONAL ICA
# =============================================================================

def run_ica_ocular_removal(
    raw: mne.io.BaseRaw,
) -> tuple[mne.io.BaseRaw, list[int], bool, list[str]]:

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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        ica.fit(
            fit_raw,
            picks="eeg",
            verbose="ERROR",
        )

        if any(
            issubclass(w.category, ConvergenceWarning)
            for w in caught
        ):
            converged = False

    if not converged:
        return raw, [], False, proxy_channels

    candidate_scores = {}

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
        raw = ica.apply(raw.copy(), verbose="ERROR")

    return raw, exclude, True, proxy_channels


# =============================================================================
# 8. EVENT-AWARE WINDOWING
# =============================================================================

def make_event_aware_windows(
    raw: mne.io.BaseRaw,
    event_blocks: pd.DataFrame,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:

    eeg = raw.copy().pick(COMMON_64)
    eeg.reorder_channels(COMMON_64)

    sfreq = float(eeg.info["sfreq"])
    samples_per_window = int(round(WINDOW_SECONDS * sfreq))

    windows = []
    starts = []
    ends = []
    labels = []
    event_codes = []
    block_ids = []
    block_starts = []
    block_ends = []

    for block in event_blocks.itertuples(index=False):
        block_start = float(block.start_sec)
        block_end = float(block.end_sec)

        start_sample = max(
            0,
            int(np.ceil(block_start * sfreq - 1e-9)),
        )

        end_sample_limit = min(
            eeg.n_times,
            int(np.floor(block_end * sfreq + 1e-9)),
        )

        sample = start_sample

        while sample + samples_per_window <= end_sample_limit:
            window = eeg.get_data(
                start=sample,
                stop=sample + samples_per_window,
            )

            if window.shape[1] != samples_per_window:
                break

            if APPLY_BASELINE:
                window = window - window.mean(axis=1, keepdims=True)

            actual_start = sample / sfreq
            actual_end = (sample + samples_per_window) / sfreq

            if (
                actual_start + 1e-9 < block_start
                or actual_end - 1e-9 > block_end
            ):
                raise RuntimeError(
                    f"Window [{actual_start}, {actual_end}) crosses "
                    f"block [{block_start}, {block_end})"
                )

            # MNE internal V -> saved uV.
            window_uv = window * OUTPUT_EEG_SCALE

            windows.append(
                np.asarray(window_uv, dtype=np.float32)
            )
            starts.append(actual_start)
            ends.append(actual_end)
            labels.append(str(block.event_label))
            event_codes.append(int(block.event_code))
            block_ids.append(str(block.event_block_id))
            block_starts.append(block_start)
            block_ends.append(block_end)

            sample += samples_per_window

    if not windows:
        raise ValueError("No complete 1-s HBN resting windows were created.")

    return (
        np.stack(windows, axis=0),
        np.asarray(starts, dtype=np.float64),
        np.asarray(ends, dtype=np.float64),
        np.asarray(labels, dtype=np.str_),
        np.asarray(event_codes, dtype=np.int16),
        np.asarray(block_ids, dtype=np.str_),
        np.asarray(block_starts, dtype=np.float64),
        np.asarray(block_ends, dtype=np.float64),
    )


# =============================================================================
# 9. WINDOW QC -- SAME DESIGN AS MODMA FINAL
# =============================================================================

def _window_qc_robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    if not np.isfinite(mad) or mad < np.finfo(float).eps:
        return np.zeros_like(values, dtype=np.float64)

    return 0.67448975 * (values - median) / mad


def reject_artifact_windows(
    data_uv: np.ndarray,
    starts: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:

    if data_uv.ndim != 3:
        raise ValueError(
            f"Window QC expected (N,C,T), got {data_uv.shape}"
        )

    n_windows = int(data_uv.shape[0])

    if not WINDOW_QC_ENABLED:
        return np.ones(n_windows, dtype=bool), {
            "enabled": False,
            "n_before": n_windows,
            "n_kept": n_windows,
            "n_removed": 0,
            "removed_fraction": 0.0,
        }

    x = np.asarray(data_uv, dtype=np.float64)

    finite_window = np.all(
        np.isfinite(x),
        axis=(1, 2),
    )

    x_safe = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    rms_uv = np.sqrt(
        np.mean(x_safe * x_safe, axis=(1, 2))
    )

    max_abs_uv = np.max(
        np.abs(x_safe),
        axis=(1, 2),
    )

    max_p2p_uv = np.max(
        np.ptp(x_safe, axis=2),
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
        robust_amplitude = max_abs_z > WINDOW_QC_ROBUST_Z_THRESHOLD
        robust_p2p = p2p_z > WINDOW_QC_ROBUST_Z_THRESHOLD
    else:
        rms_z = np.zeros(n_windows)
        max_abs_z = np.zeros(n_windows)
        p2p_z = np.zeros(n_windows)
        robust_rms = np.zeros(n_windows, dtype=bool)
        robust_amplitude = np.zeros(n_windows, dtype=bool)
        robust_p2p = np.zeros(n_windows, dtype=bool)

    # Preserve the user's finalized MODMA decision: adaptive RMS rejection
    # plus absolute amplitude/P2P thresholds.
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
    removed_fraction = n_removed / n_windows

    if n_kept == 0:
        raise RuntimeError("Window QC rejected every EEG window.")

    if n_kept / n_windows < WINDOW_QC_MIN_KEEP_FRACTION:
        raise RuntimeError(
            "Window QC rejected too much of the recording: "
            f"kept {n_kept}/{n_windows} "
            f"({n_kept / n_windows:.1%})."
        )

    if removed_fraction > WINDOW_QC_WARN_REMOVAL_FRACTION:
        warnings.warn(
            "Window QC removed a large fraction: "
            f"{n_removed}/{n_windows} ({removed_fraction:.1%}).",
            RuntimeWarning,
        )

    removed_indices = np.flatnonzero(reject)

    qc_info = {
        "enabled": True,
        "n_before": n_windows,
        "n_kept": n_kept,
        "n_removed": n_removed,
        "removed_fraction": float(removed_fraction),
        "thresholds": {
            "max_abs_uV": WINDOW_QC_MAX_ABS_UV,
            "max_p2p_uV": WINDOW_QC_MAX_P2P_UV,
            "min_rms_uV": WINDOW_QC_MIN_RMS_UV,
            "robust_z_threshold": WINDOW_QC_ROBUST_Z_THRESHOLD,
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
        "removed_indices": [int(x) for x in removed_indices],
        "removed_window_start_sec": (
            [float(x) for x in np.asarray(starts)[reject]]
            if starts is not None
            else []
        ),
    }

    return keep, qc_info


def apply_window_mask(
    keep: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(array)[keep]
        for array in arrays
    )


# =============================================================================
# 10. PEARSON FC
# =============================================================================

def compute_pearson_fc(windows: np.ndarray) -> np.ndarray:
    if windows.ndim != 3:
        raise ValueError(
            f"Expected (N,C,T), got {windows.shape}"
        )

    x = np.asarray(windows, dtype=np.float32)
    x = x - x.mean(axis=2, keepdims=True)

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

    valid_channel = norms > np.finfo(np.float32).tiny
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

    np.clip(fc, -1.0, 1.0, out=fc)

    diag = np.arange(fc.shape[1])
    fc[:, diag, diag] = 0.0

    zero_fraction = float(
        np.mean(
            np.all(fc == 0.0, axis=(1, 2))
        )
    )

    if zero_fraction > 0.95:
        raise RuntimeError(
            f"FC sanity check failed: {zero_fraction:.1%} "
            "of FC matrices are completely zero."
        )

    return fc


# =============================================================================
# 11. SAVE EO / EC NPZ
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
    source_file: str,
    original_sfreq: float,
    line_freq: float,
    load_info: dict,
    bad_channels: list[str],
    interpolation_log: dict[str, list[str]],
    ica_excluded: list[int],
    ica_converged: bool,
    ica_proxy_channels: list[str],
    window_qc: dict,
    event_blocks: pd.DataFrame,
) -> None:

    n_windows = int(data.shape[0])

    expected_fc_shape = (
        data.shape[0],
        data.shape[1],
        data.shape[1],
    )

    if data.shape[1:] != (64, 250):
        raise ValueError(
            f"Unexpected EEG shape {data.shape}; expected (N,64,250)."
        )

    if fc.shape != expected_fc_shape:
        raise ValueError(
            f"FC shape {fc.shape} != expected {expected_fc_shape}"
        )

    metadata = {
        "dataset_name": DATASET_NAME,
        "source_file": source_file,
        "subject_id": subject_id,
        "physiological_state": str(labels[0]),
        "shape": list(data.shape),
        "fc_shape": list(fc.shape),
        "channel_order": COMMON_64,
        "common64_to_hydrocel": COMMON64_TO_HYDROCEL,
        "online_reference": "Cz",
        "source_info": load_info,
        "data_unit": OUTPUT_EEG_UNIT,
        "original_sfreq": original_sfreq,
        "target_sfreq": TARGET_SFREQ,
        "window_seconds": WINDOW_SECONDS,
        "windowing": (
            "event-aware HBN resting-state segmentation; "
            "complete non-overlapping 1-s windows only"
        ),
        "event_blocks": event_blocks.to_dict(orient="records"),
        "functional_connectivity": (
            "Pearson correlation computed independently "
            "for each retained 1-s EEG window"
        ),
        "fc_range": [-1.0, 1.0],
        "fc_diagonal": 0.0,
        "bandpass_hz": [BANDPASS_LOW, BANDPASS_HIGH],
        "filter": f"{FILTER_ORDER}th-order Butterworth IIR",
        "notch_hz": line_freq,
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
        "ica_excluded_components": ica_excluded,
        "ica_converged": bool(ica_converged),
        "baseline_correction": (
            "whole-window mean" if APPLY_BASELINE else "none"
        ),
        "window_qc": window_qc,
    }

    np.savez_compressed(
        output_path,
        data=data,
        fc=fc,
        labels=labels,
        event_codes=event_codes,
        event_block_ids=event_block_ids,
        event_block_start_sec=event_block_starts,
        event_block_end_sec=event_block_ends,
        window_start_sec=starts,
        window_end_sec=ends,
        window_times_sec=np.column_stack([starts, ends]),
        subject_ids=np.asarray(
            [subject_id] * n_windows,
            dtype=np.str_,
        ),
        run_ids=np.asarray(
            ["RestingState"] * n_windows,
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
            json.dumps(metadata, ensure_ascii=False),
            dtype=np.str_,
        ),
    )


def count_existing_npz_windows(path: Path) -> int:
    if not path.exists():
        return 0

    try:
        with np.load(path, allow_pickle=False) as npz:
            return int(npz["data"].shape[0])
    except Exception:
        return 0


# =============================================================================
# 12. PROCESS ONE SUBJECT
# =============================================================================

def process_one_recording(
    set_path: Path,
    output_dir: Path,
) -> dict:

    set_path = Path(set_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_id = parse_subject_id(set_path)
    sidecars = sidecar_paths(set_path)

    output_eo = output_dir / f"{set_path.stem}_EO.npz"
    output_ec = output_dir / f"{set_path.stem}_EC.npz"

    if (
        output_eo.exists()
        and output_ec.exists()
        and not OVERWRITE
    ):
        eo_count = count_existing_npz_windows(output_eo)
        ec_count = count_existing_npz_windows(output_ec)

        return {
            "subject_id": subject_id,
            "source_file": set_path.name,
            "status": "SKIPPED_EXISTS",
            "output_file_eo": output_eo.name,
            "output_file_ec": output_ec.name,
            "eyes_open_windows": eo_count,
            "eyes_closed_windows": ec_count,
            "n_windows_total": eo_count + ec_count,
            "n_channels": 64,
            "samples_per_window": 250,
            "fc_shape_per_window": "64x64",
        }

    raw, original_sfreq, load_info = load_hbn_set(set_path)

    line_freq = get_line_frequency(sidecars["eeg_json"])

    # 1) Notch
    if line_freq < raw.info["sfreq"] / 2:
        raw.notch_filter(
            freqs=[line_freq],
            method="iir",
            verbose="ERROR",
        )

    # 2) Band-pass
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

    # 3) Resample
    if not np.isclose(raw.info["sfreq"], TARGET_SFREQ):
        raw.resample(
            TARGET_SFREQ,
            verbose="ERROR",
        )

    # 4) Bad channels
    bad_channels, bad_z_scores = detect_bad_eeg_channels(raw)

    # 5) Interpolation
    positions = get_eeg_positions(raw)

    raw, interpolation_log = interpolate_nearest_electrodes(
        raw,
        bad_channels=bad_channels,
        positions=positions,
        k=NEIGHBOR_COUNT,
    )

    # 6) CAR
    raw.set_eeg_reference(
        ref_channels="average",
        projection=False,
        verbose="ERROR",
    )

    # 7) Optional frontal-proxy ICA
    (
        raw,
        ica_excluded,
        ica_converged,
        ica_proxy_channels,
    ) = run_ica_ocular_removal(raw)

    # 8) Event blocks
    event_blocks = load_hbn_rest_blocks(
        sidecars["events"],
        recording_end_sec=float(
            raw.n_times / raw.info["sfreq"]
        ),
    )

    # 9) Event-aware windows
    (
        data,
        starts,
        ends,
        labels,
        event_codes,
        event_block_ids,
        event_block_starts,
        event_block_ends,
    ) = make_event_aware_windows(
        raw,
        event_blocks,
    )

    # 10) Window QC across all resting windows
    n_windows_before_qc = int(data.shape[0])

    keep, window_qc = reject_artifact_windows(
        data,
        starts=starts,
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
    ) = apply_window_mask(
        keep,
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

    # 11) Save EO and EC separately
    counts = {}

    for short, label, output_path in [
        ("EO", "Eyes Open", output_eo),
        ("EC", "Eyes Closed", output_ec),
    ]:
        mask = labels == label
        counts[short] = int(np.sum(mask))

        if counts[short] == 0:
            continue

        state_data = data[mask]
        state_fc = compute_pearson_fc(state_data)

        state_blocks = event_blocks[
            event_blocks["event_label"] == label
        ].copy()

        save_npz(
            output_path=output_path,
            data=state_data,
            fc=state_fc,
            starts=starts[mask],
            ends=ends[mask],
            labels=labels[mask],
            event_codes=event_codes[mask],
            event_block_ids=event_block_ids[mask],
            event_block_starts=event_block_starts[mask],
            event_block_ends=event_block_ends[mask],
            subject_id=subject_id,
            source_file=set_path.name,
            original_sfreq=original_sfreq,
            line_freq=line_freq,
            load_info=load_info,
            bad_channels=bad_channels,
            interpolation_log=interpolation_log,
            ica_excluded=ica_excluded,
            ica_converged=ica_converged,
            ica_proxy_channels=ica_proxy_channels,
            window_qc=window_qc,
            event_blocks=state_blocks,
        )

    result = {
        "subject_id": subject_id,
        "source_file": set_path.name,
        "output_file_eo": output_eo.name if counts.get("EO", 0) else "",
        "output_file_ec": output_ec.name if counts.get("EC", 0) else "",
        "status": "OK",
        "original_sfreq": original_sfreq,
        "final_sfreq": TARGET_SFREQ,
        "n_windows_before_qc": n_windows_before_qc,
        "n_windows_removed_qc": n_windows_removed_qc,
        "n_windows_total": int(data.shape[0]),
        "eyes_open_windows": counts.get("EO", 0),
        "eyes_closed_windows": counts.get("EC", 0),
        "n_channels": 64,
        "samples_per_window": 250,
        "fc_shape_per_window": "64x64",
        "line_freq": line_freq,
        "event_blocks": int(len(event_blocks)),
        "bad_channels": ",".join(bad_channels) if bad_channels else "",
        "bad_z_scores_json": json.dumps(
            {ch: bad_z_scores[ch] for ch in bad_channels},
            ensure_ascii=False,
        ),
        "interpolation_json": json.dumps(
            interpolation_log,
            ensure_ascii=False,
        ),
        "ica_proxy_channels": ",".join(ica_proxy_channels),
        "ica_excluded": ",".join(map(str, ica_excluded)),
        "ica_converged": bool(ica_converged),
    }

    del raw, data
    gc.collect()

    return result


# =============================================================================
# 13. DISCOVERY + BATCH
# =============================================================================

def discover_recordings(root: Path) -> list[Path]:
    """
    Scan actual HBN subject folders rather than guessing IDs.
    """
    return sorted(
        root.glob(
            "sub-*/eeg/sub-*_task-RestingState_eeg.set"
        )
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recordings = discover_recordings(DATASET_ROOT)

    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Recordings   : {len(recordings)}")

    if not recordings:
        raise RuntimeError(
            "No HBN RestingState SET recordings were found."
        )

    results = []
    success_count = 0
    skip_count = 0
    error_count = 0

    progress_bar = tqdm(
        recordings,
        total=len(recordings),
        desc="HBN preprocessing",
        unit="recording",
        dynamic_ncols=True,
        leave=True,
    )

    for set_path in progress_bar:
        subject_id = parse_subject_id(set_path)

        progress_bar.set_postfix(
            subject=subject_id,
            OK=success_count,
            SKIP=skip_count,
            ERR=error_count,
            refresh=True,
        )

        try:
            result = process_one_recording(
                set_path=set_path,
                output_dir=OUTPUT_DIR,
            )

            results.append(result)

            if result.get("status") == "SKIPPED_EXISTS":
                skip_count += 1
                tqdm.write(
                    f"[SKIP] {subject_id} | "
                    f"EO={result.get('eyes_open_windows', 0)} | "
                    f"EC={result.get('eyes_closed_windows', 0)} | existing"
                )
            else:
                success_count += 1
                tqdm.write(
                    f"[OK] {subject_id} | "
                    f"EO={result.get('eyes_open_windows', 0)} | "
                    f"EC={result.get('eyes_closed_windows', 0)} | "
                    f"removed={result.get('n_windows_removed_qc', 0)} | "
                    "EEG=64x250 uV | FC=64x64"
                )

        except Exception as error:
            error_count += 1

            tqdm.write(
                f"[ERROR] {subject_id} | {error}"
            )

            results.append({
                "subject_id": subject_id,
                "source_file": set_path.name,
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

        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / "HBN_preprocessing_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    progress_bar.close()

    summary = pd.DataFrame(results)

    print("\nBatch finished.")
    print(f"Successful : {success_count}")
    print(f"Skipped    : {skip_count}")
    print(f"Failed     : {error_count}")
    print(f"Total      : {len(recordings)}")

    if not summary.empty:
        print(
            summary["status"].value_counts(
                dropna=False
            )
        )

    print(
        "Summary saved to: "
        f"{OUTPUT_DIR / 'HBN_preprocessing_summary.csv'}"
    )


if __name__ == "__main__":
    main()
