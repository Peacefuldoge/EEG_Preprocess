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

Saved arrays in each EO / EC NPZ:
data : (N, C, T)
fc   : (N, C, C)

N = number of event-aware non-overlapping 1-s windows
C = 64 EEG channels
T = 250 samples after resampling to 250 Hz
Saved EEG amplitudes are in microvolts (uV).

fc[i] is the Pearson functional-connectivity matrix computed from data[i].
The FC diagonal is set to 0. Undefined correlations are safely replaced by 0.

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
from tqdm import tqdm


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

DATASET_ROOT = Path(r"E:/Downloads/ds003478")
OUTPUT_DIR = Path(r"E:/Workspace/dataset/preprocessed/PRED_CT")

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

# MNE internally uses volts. Keep all preprocessing in volts, then
# convert only the final exported EEG windows to microvolts (uV).
OUTPUT_EEG_SCALE = 1e6
OUTPUT_EEG_UNIT = "uV"

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
# The 500-ms trigger stream should keep consecutive markers close together.
# A larger gap means a new physiological-state block, even if the same
# canonical trigger code is reused later in the recording.
EVENT_BLOCK_MAX_GAP_SEC = 3.0
# Event blocks shorter than one analysis window cannot contribute any saved data
# and are treated as stray/boundary markers.
MIN_EVENT_BLOCK_SEC = WINDOW_SECONDS
# Small overlaps are usually caused by the inferred support interval of the
# final periodic trigger. Trim the earlier block at the next block onset.
# Larger overlaps remain errors because they may indicate incorrect grouping.
MAX_AUTO_TRIM_OVERLAP_SEC = WINDOW_SECONDS

# Fallback for recordings whose events.tsv contains only STATUS trigger rows.
# Trigger pairs 1/11 ... 6/16 are treated as candidate state blocks; values
# outside those ranges (e.g. 17 boundary markers) are ignored. The final EO/EC
# direction is inferred from posterior alpha power after EEG preprocessing.
ALLOW_STATUS_ALPHA_FALLBACK = True
STATUS_VALID_LOW_CODES = tuple(range(1, 7))
POSTERIOR_ALPHA_CHANNELS = ["O1", "Oz", "O2", "PO3", "POz", "PO4", "PO7", "PO8"]
ALPHA_BAND_HZ = (8.0, 13.0)
ALPHA_REFERENCE_BAND_HZ = (1.0, 30.0)
ALPHA_INFERENCE_MIN_RATIO = 1.10


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


def _canonical_predct_code(value: float) -> int:
    """Map paired PRED+CT triggers 11..16 onto 1..6."""
    code = int(round(value))
    return code - 10 if 11 <= code <= 16 else code


def _infer_marker_intervals(events: pd.DataFrame) -> pd.Series:
    """
    Return the support interval (seconds) for each repeated marker.

    Prefer explicit text such as ``Every 500 ms``. STATUS-only files do not
    contain that text, so their cadence is estimated separately for each raw
    trigger value from the median positive inter-marker interval.
    """
    cadence_ms = pd.to_numeric(
        events["trial_type"].astype(str).str.extract(r"Every\s+(\d+)\s*ms")[0],
        errors="coerce",
    )
    intervals = cadence_ms / 1000.0

    for value, idx in events.groupby("numeric_value").groups.items():
        missing_idx = [i for i in idx if not np.isfinite(intervals.loc[i])]
        if not missing_idx:
            continue

        onsets = np.sort(events.loc[list(idx), "onset"].dropna().to_numpy(float))
        diffs = np.diff(onsets)
        diffs = diffs[(diffs > 0.05) & (diffs <= EVENT_BLOCK_MAX_GAP_SEC)]
        if diffs.size:
            estimated = float(np.median(diffs))
        else:
            estimated = float(WINDOW_SECONDS)

        # Keep support intervals conservative and bounded.
        estimated = float(np.clip(estimated, 0.1, EVENT_BLOCK_MAX_GAP_SEC))
        intervals.loc[missing_idx] = estimated

    return intervals.fillna(WINDOW_SECONDS).astype(float)


def _construct_contiguous_blocks(
    events: pd.DataFrame,
    recording_end_sec: float,
    labelled: bool,
) -> pd.DataFrame:
    """Build time-contiguous blocks from already-filtered marker rows."""
    events = events.copy()
    events["event_code"] = events["numeric_value"].map(_canonical_predct_code)
    events["marker_interval_sec"] = _infer_marker_intervals(events)
    events["marker_end"] = events["onset"] + events["marker_interval_sec"]
    events = events.sort_values(["onset", "event_code"]).reset_index(drop=True)

    block_numbers = []
    current_block = 0
    prev_label = None
    prev_code = None
    prev_onset = None

    for _, row in events.iterrows():
        label = str(row["event_label"]) if labelled else "STATUS"
        code = int(row["event_code"])
        onset = float(row["onset"])
        new_block = (
            prev_code is None
            or code != prev_code
            or (labelled and label != prev_label)
            or (onset - prev_onset) > EVENT_BLOCK_MAX_GAP_SEC
        )
        if new_block:
            current_block += 1
        block_numbers.append(current_block)
        prev_label = label
        prev_code = code
        prev_onset = onset

    events["contiguous_block"] = block_numbers
    blocks = []
    state_running_index = {"EO": 0, "EC": 0, "STATUS": 0}

    for _, group in events.groupby("contiguous_block", sort=True):
        event_code = int(group["event_code"].iloc[0])
        event_label = str(group["event_label"].iloc[0]) if labelled else "Unknown"
        state_short = (
            "EO" if event_label == "Eyes Open"
            else "EC" if event_label == "Eyes Closed"
            else "STATUS"
        )
        start_sec = float(group["onset"].min())
        end_sec = min(float(group["marker_end"].max()), float(recording_end_sec))
        if end_sec <= start_sec:
            continue

        state_running_index[state_short] += 1
        block_index = state_running_index[state_short]
        trigger_values = sorted({int(round(v)) for v in group["numeric_value"]})
        blocks.append({
            "event_label": event_label,
            "event_code": event_code,
            "event_block_id": f"{state_short}_block_{block_index:02d}",
            "event_trigger_values": "|".join(map(str, trigger_values)),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "event_label_method": "events_tsv" if labelled else "status_trigger_candidate",
        })

    blocks = pd.DataFrame(blocks)
    if blocks.empty:
        raise ValueError("No valid event blocks could be constructed.")
    return blocks.sort_values("start_sec").reset_index(drop=True)


def _posterior_relative_alpha(raw: mne.io.BaseRaw, start_sec: float, end_sec: float) -> float:
    """Compute posterior relative alpha power for one candidate state block."""
    channels = [ch for ch in POSTERIOR_ALPHA_CHANNELS if ch in raw.ch_names]
    if len(channels) < 3:
        raise ValueError(
            "Not enough posterior channels for alpha-based EO/EC inference: "
            f"found {channels}"
        )

    sfreq = float(raw.info["sfreq"])
    start = max(0, int(np.ceil(start_sec * sfreq)))
    stop = min(raw.n_times, int(np.floor(end_sec * sfreq)))
    if stop - start < int(2 * sfreq):
        raise ValueError("STATUS candidate block is too short for alpha inference.")

    data = raw.get_data(picks=channels, start=start, stop=stop).astype(np.float64)
    data -= data.mean(axis=1, keepdims=True)
    n = data.shape[1]
    window = np.hanning(n)[None, :]
    spectrum = np.fft.rfft(data * window, axis=1)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)

    alpha_mask = (freqs >= ALPHA_BAND_HZ[0]) & (freqs <= ALPHA_BAND_HZ[1])
    ref_mask = (
        (freqs >= ALPHA_REFERENCE_BAND_HZ[0])
        & (freqs <= ALPHA_REFERENCE_BAND_HZ[1])
    )
    alpha = power[:, alpha_mask].sum(axis=1)
    ref = power[:, ref_mask].sum(axis=1)
    relative = alpha / np.maximum(ref, np.finfo(float).eps)
    return float(np.median(relative))


def _label_status_blocks_by_alpha(
    raw: mne.io.BaseRaw,
    blocks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Infer EO/EC for STATUS-only event blocks from posterior alpha power.

    For the common two-block case, the higher-alpha block is labelled EC and
    the lower-alpha block EO. For more than two blocks, blocks are separated
    into low/high-alpha groups using the largest gap in sorted log-alpha power.
    If the two groups are not separated by at least ALPHA_INFERENCE_MIN_RATIO,
    processing stops instead of guessing.
    """
    blocks = blocks.copy()
    scores = np.asarray([
        _posterior_relative_alpha(raw, float(r.start_sec), float(r.end_sec))
        for r in blocks.itertuples()
    ], dtype=float)

    if len(scores) < 2 or np.any(~np.isfinite(scores)) or np.any(scores <= 0):
        raise ValueError("Cannot infer EO/EC from STATUS events: invalid alpha scores.")

    order = np.argsort(scores)
    log_scores = np.log(scores[order])
    gaps = np.diff(log_scores)
    split = int(np.argmax(gaps)) + 1
    low_idx = order[:split]
    high_idx = order[split:]
    if len(low_idx) == 0 or len(high_idx) == 0:
        raise ValueError("Cannot form EO/EC alpha groups from STATUS blocks.")

    low_center = float(np.median(scores[low_idx]))
    high_center = float(np.median(scores[high_idx]))
    ratio = high_center / max(low_center, np.finfo(float).eps)
    if ratio < ALPHA_INFERENCE_MIN_RATIO:
        raise ValueError(
            "STATUS EO/EC alpha inference is ambiguous: "
            f"high/low alpha ratio={ratio:.3f} < {ALPHA_INFERENCE_MIN_RATIO:.3f}."
        )

    blocks["posterior_relative_alpha"] = scores
    blocks["alpha_group_ratio"] = ratio
    blocks.loc[low_idx, "event_label"] = "Eyes Open"
    blocks.loc[high_idx, "event_label"] = "Eyes Closed"
    blocks["event_label_method"] = "status_trigger+posterior_alpha"

    counters = {"EO": 0, "EC": 0}
    ids = []
    for label in blocks["event_label"]:
        short = "EO" if label == "Eyes Open" else "EC"
        counters[short] += 1
        ids.append(f"{short}_block_{counters[short]:02d}")
    blocks["event_block_id"] = ids
    return blocks


def load_event_blocks(
    events_path: Path,
    recording_end_sec: float,
    raw: mne.io.BaseRaw | None = None,
) -> pd.DataFrame:
    """
    Load PRED+CT EO/EC blocks.

    Primary path: use explicit ``Eyes Open`` / ``Eyes Closed`` trial_type text.
    Fallback path: for STATUS-only files, recover candidate blocks from paired
    triggers 1/11 ... 6/16, then infer EO versus EC from posterior alpha power.
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
    valid_numeric = events["onset"].notna() & events["numeric_value"].notna()

    # 1) Preferred: explicit physiological labels.
    state_pattern = r"^(Eyes Open|Eyes Closed)"
    events["event_label"] = events["trial_type"].astype(str).str.extract(state_pattern)[0]
    labelled = events[
        valid_numeric & events["event_label"].isin(EVENT_STATE_PREFIXES)
    ].copy()

    if not labelled.empty:
        blocks = _construct_contiguous_blocks(labelled, recording_end_sec, labelled=True)
    else:
        # 2) Fallback: STATUS-only paired trigger streams. Ignore markers such
        # as value 17 because they indicate boundaries rather than EO/EC state.
        if not ALLOW_STATUS_ALPHA_FALLBACK:
            raise ValueError(f"No Eyes Open / Eyes Closed events found in {events_path.name}")
        if raw is None:
            raise ValueError(
                "STATUS-only events require the preprocessed Raw object for "
                "posterior-alpha EO/EC inference."
            )

        status = events[valid_numeric].copy()
        status["canonical_code"] = status["numeric_value"].map(_canonical_predct_code)
        status = status[status["canonical_code"].isin(STATUS_VALID_LOW_CODES)].copy()
        if status.empty:
            raise ValueError(
                f"No explicit EO/EC labels and no usable STATUS trigger pairs "
                f"were found in {events_path.name}."
            )
        status["event_label"] = "Unknown"
        blocks = _construct_contiguous_blocks(status, recording_end_sec, labelled=False)
        blocks = _label_status_blocks_by_alpha(raw, blocks)

    # Blocks shorter than one analysis window cannot generate any saved 1-s
    # sample. PRED+CT occasionally contains isolated/boundary triggers near the
    # end of a recording (for example a ~0.5-s EC marker after a long EO block).
    # Keeping such blocks only creates artificial boundary-overlap errors.
    blocks = blocks.copy()
    blocks["duration_sec"] = blocks["end_sec"] - blocks["start_sec"]
    short_mask = blocks["duration_sec"] + 1e-9 < MIN_EVENT_BLOCK_SEC
    if short_mask.any():
        dropped = blocks.loc[short_mask, [
            "event_block_id", "start_sec", "end_sec", "duration_sec"
        ]].copy()
        warnings.warn(
            "Dropping event blocks shorter than one analysis window: "
            + "; ".join(
                f"{r.event_block_id}[{r.start_sec:.3f},{r.end_sec:.3f}) "
                f"duration={r.duration_sec:.3f}s"
                for r in dropped.itertuples()
            ),
            RuntimeWarning,
        )
        blocks = blocks.loc[~short_mask].copy().reset_index(drop=True)

    if blocks.empty:
        raise ValueError("No event blocks long enough to contain a complete window.")

    # Repair only small boundary overlaps. These usually come from assigning a
    # cadence-sized support interval to the last periodic trigger in a block.
    # Trimming the earlier block to the next onset prevents duplicated samples
    # while preserving every full 1-s window that can be labelled unambiguously.
    blocks["boundary_trimmed_sec"] = 0.0
    tolerance = 1e-6
    for i in range(len(blocks) - 1):
        current_end = float(blocks.loc[i, "end_sec"])
        next_start = float(blocks.loc[i + 1, "start_sec"])
        overlap = current_end - next_start

        if overlap > tolerance:
            if overlap <= MAX_AUTO_TRIM_OVERLAP_SEC + tolerance:
                blocks.loc[i, "end_sec"] = next_start
                blocks.loc[i, "boundary_trimmed_sec"] += overlap
                blocks.loc[i, "duration_sec"] = (
                    blocks.loc[i, "end_sec"] - blocks.loc[i, "start_sec"]
                )
            else:
                raise ValueError(
                    "Constructed event blocks have a large overlap that cannot "
                    "be repaired safely: "
                    f"{blocks.loc[i, 'event_block_id']} "
                    f"[{blocks.loc[i, 'start_sec']:.3f}, {current_end:.3f}) and "
                    f"{blocks.loc[i + 1, 'event_block_id']} "
                    f"[{next_start:.3f}, {blocks.loc[i + 1, 'end_sec']:.3f}); "
                    f"overlap={overlap:.3f}s > "
                    f"{MAX_AUTO_TRIM_OVERLAP_SEC:.3f}s."
                )

    # A trim can theoretically make a marginal block too short. Drop it as it
    # cannot contribute a complete window anyway, then verify the final blocks.
    blocks["duration_sec"] = blocks["end_sec"] - blocks["start_sec"]
    blocks = blocks[
        blocks["duration_sec"] + 1e-9 >= MIN_EVENT_BLOCK_SEC
    ].copy().reset_index(drop=True)

    for i in range(len(blocks) - 1):
        if float(blocks.loc[i, "end_sec"]) > float(blocks.loc[i + 1, "start_sec"]) + tolerance:
            raise RuntimeError("Internal error: repaired event blocks still overlap.")

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

            # MNE data are in volts. Export final windows in microvolts.
            window = window * OUTPUT_EEG_SCALE

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


def compute_pearson_fc(windows: np.ndarray) -> np.ndarray:
    """
    Compute one Pearson functional-connectivity matrix per EEG window.

    Parameters
    ----------
    windows
        EEG windows with shape (N, C, T).

    Returns
    -------
    fc
        Pearson correlation matrices with shape (N, C, C), dtype float32.

    Notes
    -----
    - Each FC matrix corresponds exactly to the EEG window at the same index.
    - Correlation coefficients are in [-1, 1].
    - The diagonal is explicitly set to 0 because self-connections are not
      used as graph edges in T2S-SSL.
    - If a channel has zero variance in a window, undefined correlations are
      replaced by 0 rather than NaN/Inf.
    """
    if windows.ndim != 3:
        raise ValueError(
            f"Expected windows with shape (N, C, T), got {windows.shape}"
        )

    # Work in float32 to keep memory usage practical for large batches.
    x = np.asarray(windows, dtype=np.float32)

    # Pearson correlation requires channel-wise demeaning. Baseline correction
    # already makes the mean close to zero, but this keeps FC computation
    # mathematically correct even if APPLY_BASELINE is later disabled.
    x = x - x.mean(axis=2, keepdims=True)

    # Numerator: pairwise dot products for every window.
    numerator = np.einsum(
        "nct,ndt->ncd",
        x,
        x,
        optimize=True,
        dtype=np.float32,
    )

    # Denominator: product of channel L2 norms.
    norms = np.sqrt(
        np.sum(x * x, axis=2, dtype=np.float32)
    )
    denominator = norms[:, :, None] * norms[:, None, :]

    # Do not use float32.eps as an absolute threshold here.
    # Pearson correlation is scale-invariant, and valid EEG signals may have
    # very small absolute values when represented in volts.
    valid_channel = norms > np.finfo(np.float32).tiny
    valid_pair = valid_channel[:, :, None] & valid_channel[:, None, :]

    fc = np.zeros_like(numerator, dtype=np.float32)
    np.divide(
        numerator,
        denominator,
        out=fc,
        where=valid_pair,
    )

    # Numerical safety.
    fc = np.nan_to_num(
        fc,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.clip(fc, -1.0, 1.0, out=fc)

    # Remove self-connections.
    diag = np.arange(fc.shape[1])
    fc[:, diag, diag] = 0.0

    # Prevent silent generation of broken all-zero FC batches.
    zero_fraction = float(np.mean(np.all(fc == 0.0, axis=(1, 2))))
    if zero_fraction > 0.95:
        raise RuntimeError(
            f"FC sanity check failed: {zero_fraction:.1%} of FC matrices "
            "are completely zero."
        )

    return fc


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
        "fc_shape": list(fc.shape),
        "data_unit": OUTPUT_EEG_UNIT,
        "data_scale_from_mne_volts": OUTPUT_EEG_SCALE,
        "functional_connectivity": "Pearson correlation computed independently for each 1-s EEG window",
        "fc_range": [-1.0, 1.0],
        "fc_diagonal": 0.0,
        "target_sfreq": TARGET_SFREQ,
        "window_seconds": WINDOW_SECONDS,
        "windowing": "event-aware; windows never cross event-block boundaries",
        "event_label_source": (
            str(event_blocks["event_label_method"].iloc[0])
            if "event_label_method" in event_blocks.columns and len(event_blocks)
            else "unknown"
        ),
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

    if fc.shape != (data.shape[0], data.shape[1], data.shape[1]):
        raise ValueError(
            f"FC shape {fc.shape} does not match EEG data shape {data.shape}."
        )

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
        subject_ids=np.asarray([subject_id] * n_windows, dtype=np.str_),
        run_ids=np.asarray([run_id] * n_windows, dtype=np.str_),
        dataset_names=np.asarray([DATASET_NAME] * n_windows, dtype=np.str_),
        channel_names=np.asarray(COMMON_64, dtype=np.str_),
        sfreq=np.asarray(TARGET_SFREQ, dtype=np.float32),
        data_unit=np.asarray(OUTPUT_EEG_UNIT, dtype=np.str_),
        source_file=np.asarray(source_file, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=np.str_),
    )


def count_existing_npz_windows(path: Path) -> int:
    """Return N from an existing saved NPZ without loading all EEG arrays."""
    if not path.exists():
        return 0
    try:
        with np.load(path, allow_pickle=False) as npz:
            if "data" in npz:
                return int(npz["data"].shape[0])
    except Exception:
        pass
    return 0



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
        eo_count = count_existing_npz_windows(output_path_eo)
        ec_count = count_existing_npz_windows(output_path_ec)
        return {
            "subject_id": subject_id,
            "run_id": run_id,
            "source_file": set_path.name,
            "output_file_eo": output_path_eo.name,
            "output_file_ec": output_path_ec.name,
            "status": "SKIPPED_EXISTS",
            "eyes_open_windows": eo_count,
            "eyes_closed_windows": ec_count,
            "n_windows_total": eo_count + ec_count,
            "n_channels": len(COMMON_64),
            "samples_per_window": int(round(WINDOW_SECONDS * TARGET_SFREQ)),
            "fc_shape_per_window": "64x64",
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
        raw=raw,
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

        state_data = data[mask]
        state_fc = compute_pearson_fc(state_data)

        save_npz(
            output_path=state_output_path,
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
        "fc_shape_per_window": f"{data.shape[1]}x{data.shape[1]}",
        "fc_method": "Pearson correlation; diagonal=0",
        "line_freq": line_freq,
        "eyes_open_windows": state_counts.get("EO", 0),
        "eyes_closed_windows": state_counts.get("EC", 0),
        "event_blocks": int(len(event_blocks)),
        "event_label_method": (
            str(event_blocks["event_label_method"].iloc[0])
            if "event_label_method" in event_blocks.columns and len(event_blocks)
            else "unknown"
        ),
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
    success_count = 0
    skip_count = 0
    error_count = 0

    # tqdm automatically displays:
    # percentage, processed/total recordings, elapsed time, speed and ETA.
    progress_bar = tqdm(
        recordings,
        total=len(recordings),
        desc="PRED+CT preprocessing",
        unit="recording",
        dynamic_ncols=True,
        leave=True,
    )

    for set_path in progress_bar:
        subject_id, run_id = parse_subject_and_run(set_path)

        progress_bar.set_postfix(
            subject=subject_id,
            run=run_id,
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
                    f"[SKIP] {subject_id} {run_id} | existing files | "
                    f"EO={result.get('eyes_open_windows', 0)} windows | "
                    f"EC={result.get('eyes_closed_windows', 0)} windows | "
                    f"FC=64x64"
                )
            else:
                success_count += 1
                tqdm.write(
                    f"[OK] {subject_id} {run_id} | "
                    f"EO={result.get('eyes_open_windows', 0)} windows | "
                    f"EC={result.get('eyes_closed_windows', 0)} windows | "
                    f"FC=64x64"
                )

        except Exception as error:
            error_count += 1
            tqdm.write(
                f"[ERROR] {subject_id} {run_id} | {error}"
            )
            results.append(
                {
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "source_file": set_path.name,
                    "status": "ERROR",
                    "error": repr(error),
                }
            )

        # Update the visible success/error counters after each recording.
        progress_bar.set_postfix(
            subject=subject_id,
            run=run_id,
            OK=success_count,
            SKIP=skip_count,
            ERR=error_count,
            refresh=True,
        )

        # Continuously save the log, so long jobs can be audited/resumed.
        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / "PREDCT_preprocessing_summary.csv",
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
    print(summary["status"].value_counts(dropna=False))
    print(
        f"Summary saved to: "
        f"{OUTPUT_DIR / 'PREDCT_preprocessing_summary.csv'}"
    )


if __name__ == "__main__":
    main()
