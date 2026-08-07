r"""
PRED+CT (OpenNeuro ds003478) batch EEG preprocessing.

Expected BIDS-like structure:
E:\Downloads\ds003478\
    sub-001\eeg\
        sub-001_task-Rest_run-01_eeg.set
        sub-001_task-Rest_run-01_eeg.fdt
        sub-001_task-Rest_run-01_eeg.json
        sub-001_task-Rest_run-01_channels.tsv
        sub-001_task-Rest_run-01_electrodes.tsv
        sub-001_task-Rest_run-01_events.tsv
        ...
        sub-001_task-Rest_run-02_eeg.set
        sub-001_task-Rest_run-02_eeg.fdt
        ...
    ...
    sub-122\eeg\...

Output:
Two compressed .npz files per run, split by physiological state:
sub-001_task-Rest_run-01_eeg_EO.npz
sub-001_task-Rest_run-01_eeg_EC.npz

Saved data shape:
(N, C, T)
N = number of event-aware non-overlapping 1-s windows
C = 64 EEG channels
T = 250 samples after resampling to 250 Hz

Only complete windows fully contained within one Eyes Open / Eyes Closed
event block are saved; unlabelled gaps and boundary remainders are discarded.
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


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

DATASET_ROOT = Path(r"E:\Downloads\ds003478")
OUTPUT_DIR = Path(r"E:\Downloads\ds003478_processed")

DATASET_NAME = "PRED+CT"

SUBJECT_START = 1
SUBJECT_END = 122

# Preprocessing
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
ICA_EOG_THRESHOLD = 3.0

WINDOW_SECONDS = 1.0
APPLY_BASELINE = True

# Set to True to overwrite existing .npz files.
OVERWRITE = False

# -------------------------------------------------------------------------
# Event-aware segmentation
# -------------------------------------------------------------------------
# PRED+CT events.tsv contains repeated trigger markers inside each Eyes Open /
# Eyes Closed block (e.g. Every 500 ms and Every 2000 ms). These repeated
# markers are grouped into one physiological-state block before 1-s windows
# are created. Windows are never allowed to cross an event-block boundary.
REQUIRE_EVENTS_TSV = True
EVENT_STATE_PREFIXES = ("Eyes Open", "Eyes Closed")


# =============================================================================
# 2. FIXED PRED+CT EEG CHANNEL ORDER
# =============================================================================

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

EOG_CHANNELS = ["HEOG", "VEOG"]

# CB1/CB2 have NaN positions in the provided electrodes.tsv.
# These fallback neighbours are used only if CB1/CB2 themselves are detected bad.
MANUAL_NEIGHBORS = {
    "CB1": ["O1", "PO7", "PO5"],
    "CB2": ["O2", "PO8", "PO6"],
}


# =============================================================================
# 3. HELPERS
# =============================================================================

def normalize_channel_name(name: str) -> str:
    """Normalize PRED+CT channel capitalization to the common-64 convention."""
    mapping = {
        "FP1": "Fp1",
        "FPZ": "Fpz",
        "FP2": "Fp2",
        "FZ": "Fz",
        "FCZ": "FCz",
        "CZ": "Cz",
        "CPZ": "CPz",
        "PZ": "Pz",
        "POZ": "POz",
        "OZ": "Oz",
    }
    return mapping.get(str(name), str(name))


def parse_subject_and_run(path: Path) -> tuple[str, str]:
    text = path.name
    subject_match = re.search(r"(sub-\d+)", text)
    run_match = re.search(r"(run-\d+)", text)
    subject_id = subject_match.group(1) if subject_match else "unknown"
    run_id = run_match.group(1) if run_match else "unknown"
    return subject_id, run_id


def sidecar_paths(set_path: Path) -> dict[str, Path]:
    """Return expected sidecar paths from a *_eeg.set path."""
    stem = set_path.stem  # e.g. sub-001_task-Rest_run-01_eeg
    prefix = stem[:-4] if stem.endswith("_eeg") else stem
    folder = set_path.parent
    return {
        "fdt": folder / f"{stem}.fdt",
        "eeg_json": folder / f"{stem}.json",
        "channels": folder / f"{prefix}_channels.tsv",
        "electrodes": folder / f"{prefix}_electrodes.tsv",
        "coordsystem": folder / f"{prefix}_coordsystem.json",
        "events": folder / f"{prefix}_events.tsv",
        "events_json": folder / f"{prefix}_events.json",
    }


def load_event_blocks(events_path: Path, recording_end_sec: float) -> pd.DataFrame:
    """
    Convert the PRED+CT repeated trigger markers into non-overlapping
    physiological-state blocks.

    The provided events.tsv contains repeated markers such as:
      Eyes Open: Every 500 ms    + Eyes Open: Every 2000 ms
      Eyes Closed: Every 500 ms  + Eyes Closed: Every 2000 ms

    These marker streams describe the same Eyes Open / Eyes Closed block, not
    separate EEG states. Trigger values >= 10 are paired with their low-code
    counterpart (e.g. 12 -> 2, 11 -> 1), which yields block codes 1..6 in the
    supplied PRED+CT recordings.

    Block end is inferred from the last periodic marker plus its stated marker
    interval. Unlabelled gaps between blocks are excluded from windowing.
    """
    if not events_path.exists():
        if REQUIRE_EVENTS_TSV:
            raise FileNotFoundError(f"Missing events.tsv: {events_path}")
        return pd.DataFrame()

    events = pd.read_csv(events_path, sep="\t")
    required = {"onset", "trial_type", "value"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            f"events.tsv is missing columns {sorted(missing)}. "
            f"Available columns: {list(events.columns)}"
        )

    events = events.copy()
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events["numeric_value"] = pd.to_numeric(events["value"], errors="coerce")

    # Keep only the physiological Rest-state markers used for segmentation.
    state_pattern = r"^(Eyes Open|Eyes Closed)"
    events["event_label"] = events["trial_type"].astype(str).str.extract(state_pattern)[0]
    events = events[
        events["event_label"].isin(EVENT_STATE_PREFIXES)
        & events["onset"].notna()
        & events["numeric_value"].notna()
    ].copy()

    if events.empty:
        raise ValueError(f"No Eyes Open / Eyes Closed events found in {events_path.name}")

    # PRED+CT uses paired trigger streams: 1/11, 2/12, ..., 6/16.
    def canonical_code(value: float) -> int:
        code = int(round(value))
        return code - 10 if 11 <= code <= 16 else code

    events["event_code"] = events["numeric_value"].map(canonical_code)

    # Infer each marker's support interval from text such as "Every 500 ms".
    cadence_ms = pd.to_numeric(
        events["trial_type"].astype(str).str.extract(r"Every\s+(\d+)\s*ms")[0],
        errors="coerce",
    )
    events["marker_interval_sec"] = cadence_ms / 1000.0
    events["marker_interval_sec"] = events["marker_interval_sec"].fillna(WINDOW_SECONDS)
    events["marker_end"] = events["onset"] + events["marker_interval_sec"]

    blocks = []
    for (event_label, event_code), group in events.groupby(
        ["event_label", "event_code"], sort=False
    ):
        start_sec = float(group["onset"].min())
        end_sec = float(group["marker_end"].max())
        end_sec = min(end_sec, float(recording_end_sec))
        if end_sec <= start_sec:
            continue

        block_index = int((int(event_code) + 1) // 2)
        state_short = "EO" if event_label == "Eyes Open" else "EC"
        trigger_values = sorted({int(round(v)) for v in group["numeric_value"]})

        blocks.append(
            {
                "event_label": str(event_label),
                "event_code": int(event_code),
                "event_block_id": f"{state_short}_block_{block_index:02d}",
                "event_trigger_values": "|".join(map(str, trigger_values)),
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )

    blocks = pd.DataFrame(blocks).sort_values("start_sec").reset_index(drop=True)
    if blocks.empty:
        raise ValueError(f"No valid event blocks could be constructed from {events_path.name}")

    # Safety check: event blocks used for windowing must not overlap.
    for i in range(len(blocks) - 1):
        if blocks.loc[i, "end_sec"] > blocks.loc[i + 1, "start_sec"]:
            raise ValueError(
                "Constructed event blocks overlap: "
                f"{blocks.loc[i, 'event_block_id']} and "
                f"{blocks.loc[i + 1, 'event_block_id']}"
            )

    return blocks

def get_line_frequency(eeg_json_path: Path) -> float:
    """Read power-line frequency from BIDS sidecar, defaulting to 60 Hz."""
    if eeg_json_path.exists():
        with open(eeg_json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        value = metadata.get("PowerLineFrequency", DEFAULT_LINE_FREQ)
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return float(DEFAULT_LINE_FREQ)


def load_electrode_positions(electrodes_path: Path) -> dict[str, np.ndarray]:
    """
    Load electrode XYZ coordinates for nearest-neighbour interpolation.

    Distances are used only relatively, so mm coordinates are acceptable.
    Rows with missing coordinates are ignored.
    """
    if not electrodes_path.exists():
        return {}

    table = pd.read_csv(electrodes_path, sep="\t")
    needed = {"name", "x", "y", "z"}
    if not needed.issubset(table.columns):
        return {}

    positions: dict[str, np.ndarray] = {}
    for _, row in table.iterrows():
        name = normalize_channel_name(row["name"])
        xyz = np.asarray([row["x"], row["y"], row["z"]], dtype=float)
        if np.all(np.isfinite(xyz)):
            positions[name] = xyz
    return positions


def validate_and_prepare_channel_types(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Normalize channel names, mark HEOG/VEOG as EOG, and verify all 64 EEG channels.
    """
    rename_map = {ch: normalize_channel_name(ch) for ch in raw.ch_names}
    raw.rename_channels(rename_map)

    eog_type_map = {ch: "eog" for ch in EOG_CHANNELS if ch in raw.ch_names}
    if eog_type_map:
        raw.set_channel_types(eog_type_map, verbose="ERROR")

    missing = [ch for ch in COMMON_64 if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing required PRED+CT EEG channels: {missing}")

    return raw


def detect_bad_eeg_channels(raw: mne.io.BaseRaw) -> tuple[list[str], dict[str, float]]:
    """
    Detect persistent bad EEG channels using robust z-score of log channel SD.

    A channel is marked bad if:
      1) it is flat / almost flat, or
      2) abs(robust z) > BAD_Z_THRESHOLD.

    The first 30 seconds are used, matching the validated preprocessing workflow.
    """
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
    bad_mask = flat | (np.abs(robust_z) > BAD_Z_THRESHOLD)

    bads = [eeg.ch_names[i] for i in np.where(bad_mask)[0]]
    z_scores = {eeg.ch_names[i]: float(robust_z[i]) for i in range(len(eeg.ch_names))}
    return bads, z_scores


def interpolate_nearest_electrodes(
    raw: mne.io.BaseRaw,
    bad_channels: list[str],
    positions: dict[str, np.ndarray],
    k: int = NEIGHBOR_COUNT,
) -> tuple[mne.io.BaseRaw, dict[str, list[str]]]:
    """
    Simple nearest-electrode interpolation.

    Each bad EEG channel is replaced by an inverse-distance weighted average
    of the k nearest good EEG channels.

    CB1/CB2 use manual fallback neighbours because their coordinates are NaN
    in the PRED+CT electrodes.tsv sidecars.
    """
    if not bad_channels:
        return raw, {}

    interpolation_log: dict[str, list[str]] = {}
    good_channels = [ch for ch in COMMON_64 if ch not in bad_channels]

    for bad in bad_channels:
        if bad not in raw.ch_names:
            continue

        neighbours: list[str] = []
        weights: np.ndarray | None = None

        # Normal coordinate-based nearest-neighbour interpolation.
        if bad in positions:
            candidates = []
            for good in good_channels:
                if good in positions:
                    distance = float(np.linalg.norm(positions[good] - positions[bad]))
                    if np.isfinite(distance) and distance > 0:
                        candidates.append((good, distance))

            candidates.sort(key=lambda item: item[1])
            selected = candidates[:k]
            if selected:
                neighbours = [name for name, _ in selected]
                distances = np.asarray([distance for _, distance in selected], dtype=float)
                weights = 1.0 / np.maximum(distances, 1e-12)
                weights /= weights.sum()

        # Manual fallback for channels whose coordinates are missing.
        if not neighbours and bad in MANUAL_NEIGHBORS:
            neighbours = [
                ch for ch in MANUAL_NEIGHBORS[bad]
                if ch in good_channels and ch in raw.ch_names
            ][:k]
            if neighbours:
                weights = np.ones(len(neighbours), dtype=float)
                weights /= weights.sum()

        if not neighbours or weights is None:
            raise RuntimeError(
                f"Cannot interpolate bad channel {bad}: "
                "no valid neighbouring electrodes were found."
            )

        bad_idx = raw.ch_names.index(bad)
        neighbour_idx = [raw.ch_names.index(ch) for ch in neighbours]
        neighbour_data = raw._data[neighbour_idx, :]
        raw._data[bad_idx, :] = np.average(neighbour_data, axis=0, weights=weights)
        interpolation_log[bad] = neighbours

    return raw, interpolation_log


def run_ica_ocular_removal(
    raw: mne.io.BaseRaw,
) -> tuple[mne.io.BaseRaw, list[int], bool]:
    """
    Fit FastICA on EEG and remove at most two ocular-related components.

    HEOG and VEOG are used as true EOG reference channels when available.
    ICA is fitted on at most the first 30 s to keep batch processing practical.
    If FastICA does not converge, no ICA components are removed.
    """
    eog_channels = [ch for ch in EOG_CHANNELS if ch in raw.ch_names]
    if not eog_channels:
        return raw, [], True

    fit_raw = raw.copy()
    if fit_raw.times[-1] > ICA_FIT_SECONDS:
        fit_raw.crop(tmin=0.0, tmax=ICA_FIT_SECONDS, include_tmax=False)

    n_components = min(ICA_MAX_COMPONENTS, len(COMMON_64) - 1)
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ica.fit(fit_raw, picks="eeg", verbose="ERROR")
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
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
                score = float(abs(scores[idx])) if idx < len(scores) else 0.0
                candidate_scores[idx] = max(candidate_scores.get(idx, 0.0), score)
        except Exception:
            # One problematic EOG channel should not stop an entire batch.
            continue

    ranked = sorted(candidate_scores, key=candidate_scores.get, reverse=True)
    exclude = ranked[:ICA_MAX_OCULAR_COMPONENTS]

    if exclude:
        ica.exclude = exclude
        raw = ica.apply(raw.copy(), verbose="ERROR")

    return raw, exclude, True


def make_event_aware_one_second_windows(
    raw: mne.io.BaseRaw,
    event_blocks: pd.DataFrame,
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
    Create non-overlapping 1-s windows strictly inside each event block.

    A window is kept only when all of its samples lie inside one event block.
    Samples before the first event, gaps between blocks, and the remainder at
    each block boundary (< 1 s) are discarded.

    Returns
    -------
    data : (N, 64, 250), float32
    starts, ends : actual sample-aligned window times in seconds
    labels : "Eyes Open" / "Eyes Closed"
    event_codes : canonical PRED+CT block codes (1..6 in the sample)
    block_ids : e.g. EO_block_01 / EC_block_01
    block_starts, block_ends : source event-block boundaries
    """
    eeg = raw.copy().pick(COMMON_64)
    eeg.reorder_channels(COMMON_64)

    sfreq = float(eeg.info["sfreq"])
    samples_per_window = int(round(WINDOW_SECONDS * sfreq))
    if samples_per_window <= 0:
        raise ValueError("Invalid window length.")

    windows = []
    starts = []
    ends = []
    labels = []
    event_codes = []
    block_ids = []
    block_starts = []
    block_ends = []

    for _, block in event_blocks.iterrows():
        block_start = float(block["start_sec"])
        block_end = float(block["end_sec"])

        # ceil at the left edge prevents even a few milliseconds from leaking
        # in from the previous event. floor at the right edge prevents crossing
        # into the next event/gap after resampling.
        start_sample = int(np.ceil(block_start * sfreq - 1e-9))
        end_sample_limit = int(np.floor(block_end * sfreq + 1e-9))
        start_sample = max(0, start_sample)
        end_sample_limit = min(eeg.n_times, end_sample_limit)

        sample = start_sample
        while sample + samples_per_window <= end_sample_limit:
            window = eeg.get_data(start=sample, stop=sample + samples_per_window)
            if window.shape[1] != samples_per_window:
                break

            if APPLY_BASELINE:
                window = window - window.mean(axis=1, keepdims=True)

            actual_start = sample / sfreq
            actual_end = (sample + samples_per_window) / sfreq

            # Hard assertion: no saved window may cross its event boundaries.
            if actual_start + 1e-9 < block_start or actual_end - 1e-9 > block_end:
                raise RuntimeError(
                    f"Window [{actual_start}, {actual_end}) crosses event block "
                    f"[{block_start}, {block_end})"
                )

            windows.append(np.asarray(window, dtype=np.float32))
            starts.append(actual_start)
            ends.append(actual_end)
            labels.append(str(block["event_label"]))
            event_codes.append(int(block["event_code"]))
            block_ids.append(str(block["event_block_id"]))
            block_starts.append(block_start)
            block_ends.append(block_end)

            sample += samples_per_window

    if not windows:
        raise ValueError("No complete 1-s windows were found inside event blocks.")

    data = np.stack(windows, axis=0)
    return (
        data,
        np.asarray(starts, dtype=np.float64),
        np.asarray(ends, dtype=np.float64),
        np.asarray(labels, dtype=np.str_),
        np.asarray(event_codes, dtype=np.int16),
        np.asarray(block_ids, dtype=np.str_),
        np.asarray(block_starts, dtype=np.float64),
        np.asarray(block_ends, dtype=np.float64),
    )

def save_npz(
    output_path: Path,
    data: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    labels: np.ndarray,
    event_codes: np.ndarray,
    event_block_ids: np.ndarray,
    event_block_starts: np.ndarray,
    event_block_ends: np.ndarray,
    subject_id: str,
    run_id: str,
    source_file: str,
    line_freq: float,
    bad_channels: list[str],
    interpolation_log: dict[str, list[str]],
    ica_excluded: list[int],
    ica_converged: bool,
    event_blocks: pd.DataFrame,
) -> None:
    """Save event-aware EEG windows and metadata without pickle arrays."""
    n_windows = data.shape[0]

    label_counts = {
        str(label): int(np.sum(labels == label))
        for label in np.unique(labels)
    }

    metadata = {
        "dataset_name": DATASET_NAME,
        "source_file": source_file,
        "subject_id": subject_id,
        "run_id": run_id,
        "shape": list(data.shape),
        "data_unit": "V",
        "target_sfreq": TARGET_SFREQ,
        "window_seconds": WINDOW_SECONDS,
        "windowing": "event-aware; windows never cross event-block boundaries",
        "event_label_source": "events.tsv trial_type, grouped to Eyes Open / Eyes Closed blocks",
        "event_marker_grouping": "paired repeated markers such as 2/12 and 1/11 belong to the same state block",
        "unlabelled_gaps": "discarded",
        "incomplete_boundary_remainders": "discarded",
        "label_counts": label_counts,
        "event_blocks": event_blocks.to_dict(orient="records"),
        "bandpass_hz": [BANDPASS_LOW, BANDPASS_HIGH],
        "filter": f"{FILTER_ORDER}th-order Butterworth IIR",
        "notch_hz": line_freq,
        "bad_channel_method": f"robust z-score of log(SD), |z| > {BAD_Z_THRESHOLD}",
        "bad_channels": bad_channels,
        "interpolation_method": f"{NEIGHBOR_COUNT}-nearest-electrode inverse-distance weighted average",
        "interpolation_neighbours": interpolation_log,
        "reference": "common average reference",
        "ica_method": "FastICA",
        "ica_max_components": ICA_MAX_COMPONENTS,
        "ica_max_ocular_components_removed": ICA_MAX_OCULAR_COMPONENTS,
        "ica_excluded_components": ica_excluded,
        "ica_converged": bool(ica_converged),
        "baseline_correction": "whole-window mean" if APPLY_BASELINE else "none",
    }

    np.savez_compressed(
        output_path,
        data=data,
        labels=labels,
        event_codes=event_codes,
        event_block_ids=event_block_ids,
        event_block_start_sec=event_block_starts,
        event_block_end_sec=event_block_ends,
        window_start_sec=starts,
        window_end_sec=ends,
        window_times_sec=np.column_stack([starts, ends]),
        subject_ids=np.asarray([subject_id] * n_windows, dtype=np.str_),
        run_ids=np.asarray([run_id] * n_windows, dtype=np.str_),
        dataset_names=np.asarray([DATASET_NAME] * n_windows, dtype=np.str_),
        channel_names=np.asarray(COMMON_64, dtype=np.str_),
        sfreq=np.asarray(TARGET_SFREQ, dtype=np.float32),
        data_unit=np.asarray("V", dtype=np.str_),
        source_file=np.asarray(source_file, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=np.str_),
    )


# =============================================================================
# 4. PROCESS ONE RUN
# =============================================================================

def process_one_recording(
    set_path: Path,
    output_dir: Path,
) -> dict:
    """Preprocess one PRED+CT *.set + *.fdt run and save separate EO/EC .npz files."""
    set_path = Path(set_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_id, run_id = parse_subject_and_run(set_path)
    sidecars = sidecar_paths(set_path)

    if not sidecars["fdt"].exists():
        raise FileNotFoundError(
            f"Missing FDT file for {set_path.name}: {sidecars['fdt']}"
        )

    output_path_eo = output_dir / f"{set_path.stem}_EO.npz"
    output_path_ec = output_dir / f"{set_path.stem}_EC.npz"

    if output_path_eo.exists() and output_path_ec.exists() and not OVERWRITE:
        return {
            "subject_id": subject_id,
            "run_id": run_id,
            "source_file": set_path.name,
            "output_file_eo": output_path_eo.name,
            "output_file_ec": output_path_ec.name,
            "status": "SKIPPED_EXISTS",
        }

    # Read SET; MNE automatically loads its paired external FDT.
    raw = mne.io.read_raw_eeglab(
        set_path,
        preload=True,
        verbose="ERROR",
    )

    original_sfreq = float(raw.info["sfreq"])
    raw = validate_and_prepare_channel_types(raw)

    # Replace any non-finite samples before filtering.
    raw._data[~np.isfinite(raw._data)] = 0.0

    line_freq = get_line_frequency(sidecars["eeg_json"])

    # 1) Notch filter.
    if line_freq < raw.info["sfreq"] / 2:
        raw.notch_filter(
            freqs=[line_freq],
            method="iir",
            verbose="ERROR",
        )

    # 2) Band-pass filter.
    raw.filter(
        l_freq=BANDPASS_LOW,
        h_freq=BANDPASS_HIGH,
        method="iir",
        iir_params=dict(order=FILTER_ORDER, ftype="butter"),
        verbose="ERROR",
    )

    # 3) Resample.
    if not np.isclose(raw.info["sfreq"], TARGET_SFREQ):
        raw.resample(TARGET_SFREQ, verbose="ERROR")

    # 4) Bad-channel detection.
    bad_channels, bad_z_scores = detect_bad_eeg_channels(raw)

    # 5) Simple nearest-electrode interpolation.
    positions = load_electrode_positions(sidecars["electrodes"])
    raw, interpolation_log = interpolate_nearest_electrodes(
        raw,
        bad_channels=bad_channels,
        positions=positions,
        k=NEIGHBOR_COUNT,
    )

    # 6) Common average reference on EEG channels.
    raw.set_eeg_reference(
        ref_channels="average",
        projection=False,
        verbose="ERROR",
    )

    # 7) ICA ocular artifact removal using HEOG / VEOG.
    raw, ica_excluded, ica_converged = run_ica_ocular_removal(raw)

    # 8) Keep final 64 EEG channels only.
    raw.pick(COMMON_64)
    raw.reorder_channels(COMMON_64)

    # 9) Read events.tsv and construct Eyes Open / Eyes Closed event blocks.
    event_blocks = load_event_blocks(
        sidecars["events"],
        recording_end_sec=float(raw.n_times / raw.info["sfreq"]),
    )

    # 10) Event-aware non-overlapping 1-s windows.
    (
        data, starts, ends, labels, event_codes, event_block_ids,
        event_block_starts, event_block_ends,
    ) = make_event_aware_one_second_windows(raw, event_blocks)

    # 11) Split Eyes Open and Eyes Closed, then save two independent files.
    state_specs = [
        ("EO", "Eyes Open", output_path_eo),
        ("EC", "Eyes Closed", output_path_ec),
    ]
    state_counts = {}

    for state_short, state_label, state_output_path in state_specs:
        mask = labels == state_label
        state_counts[state_short] = int(np.sum(mask))

        if state_counts[state_short] == 0:
            print(f"  [WARN] No complete {state_label} windows: {set_path.name}")
            continue

        state_blocks = event_blocks[event_blocks["event_label"] == state_label].copy()

        # When resuming a partial batch, keep an existing state file unless overwrite is enabled.
        if state_output_path.exists() and not OVERWRITE:
            continue

        save_npz(
            output_path=state_output_path,
            data=data[mask],
            starts=starts[mask],
            ends=ends[mask],
            labels=labels[mask],
            event_codes=event_codes[mask],
            event_block_ids=event_block_ids[mask],
            event_block_starts=event_block_starts[mask],
            event_block_ends=event_block_ends[mask],
            subject_id=subject_id,
            run_id=run_id,
            source_file=sidecars["fdt"].name,
            line_freq=line_freq,
            bad_channels=bad_channels,
            interpolation_log=interpolation_log,
            ica_excluded=ica_excluded,
            ica_converged=ica_converged,
            event_blocks=state_blocks,
        )

    result = {
        "subject_id": subject_id,
        "run_id": run_id,
        "source_file": sidecars["fdt"].name,
        "output_file_eo": output_path_eo.name if state_counts.get("EO", 0) else "",
        "output_file_ec": output_path_ec.name if state_counts.get("EC", 0) else "",
        "status": "OK",
        "original_sfreq": original_sfreq,
        "final_sfreq": TARGET_SFREQ,
        "n_windows_total": int(data.shape[0]),
        "n_channels": int(data.shape[1]),
        "samples_per_window": int(data.shape[2]),
        "line_freq": line_freq,
        "eyes_open_windows": state_counts.get("EO", 0),
        "eyes_closed_windows": state_counts.get("EC", 0),
        "event_blocks": int(len(event_blocks)),
        "bad_channels": ",".join(bad_channels) if bad_channels else "",
        "bad_z_scores_json": json.dumps(
            {ch: bad_z_scores[ch] for ch in bad_channels},
            ensure_ascii=False,
        ),
        "interpolation_json": json.dumps(interpolation_log, ensure_ascii=False),
        "ica_excluded": ",".join(map(str, ica_excluded)),
        "ica_converged": bool(ica_converged),
    }

    del raw, data
    gc.collect()

    return result


# =============================================================================
# 5. DISCOVER AND BATCH PROCESS SUB-001 ... SUB-122, ALL AVAILABLE RUNS
# =============================================================================

def discover_recordings(root: Path) -> list[Path]:
    recordings: list[Path] = []

    for subject_number in range(SUBJECT_START, SUBJECT_END + 1):
        subject_id = f"sub-{subject_number:03d}"
        eeg_dir = root / subject_id / "eeg"

        if not eeg_dir.exists():
            print(f"[WARN] Missing EEG directory: {eeg_dir}")
            continue

        # This automatically includes run-01, run-02, and any additional runs.
        matches = sorted(eeg_dir.glob(f"{subject_id}_task-Rest_run-*_eeg.set"))

        if not matches:
            print(f"[WARN] No Rest SET files found in: {eeg_dir}")
            continue

        recordings.extend(matches)

    return recordings


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recordings = discover_recordings(DATASET_ROOT)

    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Recordings   : {len(recordings)}")

    if not recordings:
        raise RuntimeError("No PRED+CT recordings were found.")

    results = []

    for index, set_path in enumerate(recordings, start=1):
        print(f"\n[{index}/{len(recordings)}] {set_path}")

        try:
            result = process_one_recording(
                set_path=set_path,
                output_dir=OUTPUT_DIR,
            )
            results.append(result)
            print(
                f"  -> {result['status']}: "
                f"EO={result.get('output_file_eo', '')} "
                f"({result.get('eyes_open_windows', '')} windows), "
                f"EC={result.get('output_file_ec', '')} "
                f"({result.get('eyes_closed_windows', '')} windows)"
            )
        except Exception as error:
            subject_id, run_id = parse_subject_and_run(set_path)
            print(f"  -> ERROR: {error}")
            results.append(
                {
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "source_file": set_path.name,
                    "status": "ERROR",
                    "error": repr(error),
                }
            )

        # Write a continuously updated log so a long batch can be resumed/audited.
        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / "PREDCT_preprocessing_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary = pd.DataFrame(results)
    print("\nBatch finished.")
    print(summary["status"].value_counts(dropna=False))
    print(f"Summary saved to: {OUTPUT_DIR / 'PREDCT_preprocessing_summary.csv'}")


if __name__ == "__main__":
    main()
