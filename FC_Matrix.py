from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

# 可以直接指定四个 npz 文件
DATASETS = {
    "PRED+CT": r"E:\Workspace\dataset\preprocessed\PRED_CT\sub-001_task-Rest_run-01_eeg_EC.npz",

    "HBN": r"E:\Workspace\dataset\preprocessed\HBN\sub-NDARAC904DMU_task-RestingState_eeg_EC.npz",

    "MODMA": r"E:\Workspace\dataset\preprocessed\MODMA\02010011_EC.npz",

    "TDBRAIN": r"E:\Workspace\dataset\preprocessed\TDBRAIN\sub-87958057_ses-1_task-restEC_eeg_EC.npz",
}


# ============================================================
# FC plotting mode
# ============================================================

# 推荐使用 "mean"
#
# "mean"
#     对一个被试所有 window 的 FC 求平均
#
# "single"
#     只显示某一个 1-s window 的 FC
#
# "recompute"
#     将所有 window EEG 拼起来后重新计算一次 Pearson correlation
#
FC_MODE = "single"


# 如果 FC_MODE == "single"，使用第几个 window
WINDOW_INDEX = 33


# 你的 npz 里面 FC 对角线为了 graph reconstruction 被设成了 0。
#
# 但标准 Pearson FC 图通常显示：
# Corr(channel_i, channel_i) = 1
#
# 如果只是为了论文/PPT可视化，推荐 True。
DISPLAY_DIAGONAL_AS_ONE = True


# 是否使用绝对值
# False = -1 ~ 1
# True  = 0 ~ 1
USE_ABSOLUTE_FC = False


# 保存文件
OUTPUT_FILE = "four_datasets_fc_after_preprocessing.png"

# 图片 DPI
DPI = 300


# ============================================================
# Load NPZ
# ============================================================

def load_npz_fc(file_path, dataset_name):
    """
    Read unified preprocessed NPZ.

    Expected:
        data: (windows, channels, samples)
        fc:   (windows, channels, channels)
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"{dataset_name}: file not found:\n{file_path}"
        )

    npz = np.load(
        file_path,
        allow_pickle=True
    )

    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"File:    {file_path}")
    print("=" * 70)

    print("Available keys:")
    print(npz.files)

    # --------------------------------------------------------
    # EEG
    # --------------------------------------------------------

    if "data" not in npz:
        raise KeyError(
            f"{dataset_name}: 'data' not found in NPZ."
        )

    data = np.asarray(
        npz["data"],
        dtype=np.float32
    )

    print(f"EEG data shape : {data.shape}")

    # --------------------------------------------------------
    # FC
    # --------------------------------------------------------

    if "fc" not in npz:
        raise KeyError(
            f"{dataset_name}: 'fc' not found in NPZ."
        )

    fc_windows = np.asarray(
        npz["fc"],
        dtype=np.float32
    )

    print(f"FC data shape  : {fc_windows.shape}")

    # --------------------------------------------------------
    # Channel names
    # --------------------------------------------------------

    if "channel_names" in npz:
        channel_names = npz["channel_names"].astype(str)
    else:
        channel_names = np.array(
            [
                f"Ch{i + 1}"
                for i in range(data.shape[1])
            ]
        )

    print(f"Channels       : {len(channel_names)}")

    # --------------------------------------------------------
    # Sampling frequency
    # --------------------------------------------------------

    if "sfreq" in npz:
        sfreq = float(npz["sfreq"])
        print(f"Sampling rate  : {sfreq:.1f} Hz")
    else:
        sfreq = None

    # --------------------------------------------------------
    # Number of windows
    # --------------------------------------------------------

    print(f"Windows        : {data.shape[0]}")

    return {
        "data": data,
        "fc": fc_windows,
        "channel_names": channel_names,
        "sfreq": sfreq,
        "file": file_path,
    }


# ============================================================
# Generate one FC matrix
# ============================================================

def get_fc_matrix(dataset):
    """
    Convert window-level FCs into one FC matrix for visualization.
    """

    data = dataset["data"]
    fc_windows = dataset["fc"]

    # ========================================================
    # Mode 1:
    # Mean of precomputed window FCs
    # ========================================================

    if FC_MODE == "mean":

        fc = np.mean(
            fc_windows,
            axis=0
        )

    # ========================================================
    # Mode 2:
    # Show one individual window
    # ========================================================

    elif FC_MODE == "single":

        if WINDOW_INDEX >= fc_windows.shape[0]:
            raise IndexError(
                f"WINDOW_INDEX={WINDOW_INDEX}, "
                f"but only {fc_windows.shape[0]} windows exist."
            )

        fc = fc_windows[
            WINDOW_INDEX
        ].copy()

    # ========================================================
    # Mode 3:
    # Recompute FC from concatenated EEG
    # ========================================================

    elif FC_MODE == "recompute":

        # data:
        # windows × channels × samples
        #
        # Convert to:
        # channels × total_samples

        concatenated = (
            data
            .transpose(1, 0, 2)
            .reshape(data.shape[1], -1)
        )

        fc = np.corrcoef(
            concatenated
        )

    else:

        raise ValueError(
            f"Unknown FC_MODE: {FC_MODE}"
        )

    # --------------------------------------------------------
    # Remove NaN / Inf
    # --------------------------------------------------------

    fc = np.nan_to_num(
        fc,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Optional absolute FC
    # --------------------------------------------------------

    if USE_ABSOLUTE_FC:
        fc = np.abs(fc)

    # --------------------------------------------------------
    # Display standard Pearson diagonal
    # --------------------------------------------------------

    if DISPLAY_DIAGONAL_AS_ONE:
        np.fill_diagonal(
            fc,
            1.0
        )

    return fc


# ============================================================
# Plot one FC
# ============================================================

def plot_fc(
    ax,
    fc,
    dataset_name
):

    if USE_ABSOLUTE_FC:

        vmin = 0
        vmax = 1

    else:

        vmin = -1
        vmax = 1

    im = ax.imshow(
        fc,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
        aspect="equal"
    )

    ax.set_title(
        dataset_name,
        fontsize=11
    )

    ax.set_xlabel(
        "Channel index",
        fontsize=9
    )

    ax.set_ylabel(
        "Channel index",
        fontsize=9
    )

    ax.tick_params(
        axis="both",
        labelsize=8
    )

    return im


# ============================================================
# Main
# ============================================================

def main():

    results = {}

    # --------------------------------------------------------
    # Read four datasets
    # --------------------------------------------------------

    for dataset_name, file_path in DATASETS.items():

        try:

            dataset = load_npz_fc(
                file_path,
                dataset_name
            )

            fc = get_fc_matrix(
                dataset
            )

            print(
                f"Final FC shape : {fc.shape}"
            )

            print(
                f"FC range       : "
                f"{fc.min():.4f} ~ {fc.max():.4f}"
            )

            results[dataset_name] = {
                "fc": fc,
                "dataset": dataset
            }

        except Exception as e:

            print(
                f"\nERROR processing {dataset_name}:"
            )

            print(e)

    if len(results) == 0:

        raise RuntimeError(
            "No dataset was successfully loaded."
        )

    # ========================================================
    # Plot
    # ========================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8)
    )

    axes = axes.flatten()

    image = None

    for ax, dataset_name in zip(
        axes,
        DATASETS.keys()
    ):

        if dataset_name not in results:
            ax.axis("off")
            continue

        fc = results[
            dataset_name
        ]["fc"]

        image = plot_fc(
            ax,
            fc,
            dataset_name
        )

    # --------------------------------------------------------
    # Remove unused subplot
    # --------------------------------------------------------

    for i in range(
        len(DATASETS),
        len(axes)
    ):
        axes[i].axis("off")

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.07,
        right=0.87,
        bottom=0.08,
        top=0.91,
        wspace=0.20,
        hspace=0.22
    )

    # --------------------------------------------------------
    # Shared color bar
    # --------------------------------------------------------

    if image is not None:

        cbar_ax = fig.add_axes(
            [
                0.90,
                0.18,
                0.020,
                0.65
            ]
        )

        cbar = fig.colorbar(
            image,
            cax=cbar_ax
        )

        if USE_ABSOLUTE_FC:
            cbar.set_label(
                "|Pearson r|",
                fontsize=9
            )
        else:
            cbar.set_label(
                "Pearson r",
                fontsize=9
            )

        cbar.ax.tick_params(
            labelsize=8
        )

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        "Functional Connectivity After Preprocessing",
        fontsize=13,
        y=0.99
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.savefig(
        OUTPUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="none",
        transparent=True
    )

    print(
        f"\nSaved figure:\n{OUTPUT_FILE}"
    )

    plt.show()


if __name__ == "__main__":
    main()