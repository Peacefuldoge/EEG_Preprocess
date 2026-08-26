from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


# ============================================================
# 1. 输入文件：以后主要改这里
# ============================================================
FILES = [
    r"E:\Workspace\dataset\preprocessed\MODMA\02010011_EC.npz",
]

# 输出目录
OUT_DIR = Path(r"./fc_circle_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 绘图参数
# ============================================================
# "mean"   : 对一个文件内所有窗口的 FC 取平均
# "window" : 只画某个窗口
PLOT_MODE = "window"
WINDOW_IDX = 2

# 最强连接数量
N_LINES_64 = 2016
N_LINES_26 = 676

# 普通 FC 连线宽度
NORMAL_LINEWIDTH = 1.15
NORMAL_LINE_ALPHA = 0.90

# 被 mask 的连接：纯黑、加粗
MASKED_LINEWIDTH = 2.5

# 可以指定任意多对。
# 这里放了 3 对示例，并且这 3 对在你给的 64ch / 26ch 文件中都存在。
# MASKED_PAIRS = [
#     ("Fp1", "O2"),
#     ("F3", "P4"),
#     ("C3", "C4"),
#     ("F3", "F5"),
#     ("F7", "P1"),
#     ("AF4", "F1"),
#     ("P7", "P5"),
#     ("P5", "P3"),
#     ("P3", "O1"),
# ]
MASKED_PAIRS = []

# FC 色标范围
VMIN = -1.0
VMAX = 1.0
COLORMAP = "RdBu_r"

# 图片设置
FIGSIZE = (10.8, 9.6)
DPI = 300

# 右侧 colorbar 的位置：[left, bottom, width, height]
COLORBAR_RECT = [0.905, 0.275, 0.020, 0.46]

# 标题位置；越小越往下
TITLE_Y = 0.965


# ============================================================
# 3. 电极圆环顺序
# ============================================================
def make_node_order(ch_names):
    ch_names = [str(ch) for ch in ch_names]

    if len(ch_names) == 26:
        template = [
            "Fp1", "Fp2",
            "F7", "F3", "Fz", "F4", "F8",
            "FC3", "FCz", "FC4",
            "T7", "C3", "Cz", "C4", "T8",
            "CP3", "CPz", "CP4",
            "P7", "P3", "Pz", "P4", "P8",
            "O1", "Oz", "O2",
        ]
    elif len(ch_names) == 64:
        template = [
            "Fp1", "AF3", "F7", "F5", "F3", "F1",
            "FT7", "FC5", "FC3", "FC1",
            "T7", "C5", "C3", "C1", "M1",
            "TP7", "CP5", "CP3", "CP1",
            "P7", "P5", "P3", "P1",
            "PO7", "PO5", "PO3", "CB1", "O1",
            "Oz",
            "O2", "CB2", "PO8", "PO6", "PO4",
            "P2", "P4", "P6", "P8",
            "CP2", "CP4", "CP6", "TP8", "M2",
            "C2", "C4", "C6", "T8",
            "FC2", "FC4", "FC6", "FT8",
            "F2", "F4", "F6", "F8", "AF4", "Fp2",
            "Fpz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz",
        ]
    else:
        template = ch_names

    ordered = [ch for ch in template if ch in ch_names]
    ordered += [ch for ch in ch_names if ch not in ordered]
    return ordered


def make_node_angles(ch_names, node_order):
    """返回与 ch_names 一一对应的角度（弧度）。"""
    n = len(node_order)
    order_angle = {
        ch: 2.0 * np.pi * idx / n
        for idx, ch in enumerate(node_order)
    }
    return np.array([order_angle[ch] for ch in ch_names], dtype=float)


# ============================================================
# 4. 数据读取
# ============================================================
def load_fc_and_names(npz_path, plot_mode="mean", window_idx=0):
    with np.load(npz_path, allow_pickle=True) as z:
        fc = np.asarray(z["fc"], dtype=np.float64)
        ch_names = [str(ch) for ch in z["channel_names"]]

        if fc.ndim == 3:
            if plot_mode == "mean":
                fc_plot = np.nanmean(fc, axis=0)
            elif plot_mode == "window":
                if not 0 <= window_idx < fc.shape[0]:
                    raise IndexError(
                        f"{npz_path}: WINDOW_IDX={window_idx} 超出范围，"
                        f"共有 {fc.shape[0]} 个窗口。"
                    )
                fc_plot = fc[window_idx]
            else:
                raise ValueError("PLOT_MODE 只能是 'mean' 或 'window'")
        elif fc.ndim == 2:
            fc_plot = fc
        else:
            raise ValueError(f"无法识别的 fc 维度: {fc.shape}")

        # 保证完全对称，对角线为 0
        fc_plot = 0.5 * (fc_plot + fc_plot.T)
        np.fill_diagonal(fc_plot, 0.0)

        subject = (
            str(z["subject_ids"][0])
            if "subject_ids" in z.files
            else Path(npz_path).stem
        )
        dataset = (
            str(z["dataset_names"][0])
            if "dataset_names" in z.files
            else "Unknown"
        )
        n_windows = int(fc.shape[0]) if fc.ndim == 3 else 1

    return fc_plot, ch_names, {
        "dataset": dataset,
        "subject": subject,
        "n_windows": n_windows,
        "n_channels": len(ch_names),
    }


# ============================================================
# 5. mask 电极对处理
# ============================================================
def resolve_masked_pairs(ch_names, masked_pairs):
    """
    把 [('Fp1','O2'), ...] 转成索引对。
    不存在的通道自动跳过；重复边自动去重。
    """
    name_to_idx = {str(name): idx for idx, name in enumerate(ch_names)}

    resolved = []
    seen = set()

    for a, b in masked_pairs:
        if a not in name_to_idx or b not in name_to_idx:
            missing = [x for x in (a, b) if x not in name_to_idx]
            print(
                f"  [WARN] 跳过 masked pair ({a}, {b})，"
                f"当前文件缺少通道: {missing}"
            )
            continue

        ia, ib = name_to_idx[a], name_to_idx[b]
        if ia == ib:
            print(f"  [WARN] 跳过 masked pair ({a}, {b})：是同一个电极")
            continue

        key = tuple(sorted((ia, ib)))
        if key in seen:
            continue

        seen.add(key)
        resolved.append((key[0], key[1], a, b))

    return resolved


# ============================================================
# 6. 画一条圆内曲线
# ============================================================
def draw_connection(
    ax,
    theta1,
    theta2,
    color,
    linewidth,
    alpha=1.0,
    zorder=1,
):
    """
    在 polar axes 内画从一个电极到另一个电极的三次 Bezier 曲线。
    坐标是 (theta, radius)，中间控制点靠近圆心。
    """
    # 两个控制点半径越小，曲线越向圆心弯。
    # 这里让远距离电极更明显穿过中间，同时保留近邻的短弧形态。
    angular_distance = abs(np.angle(np.exp(1j * (theta2 - theta1))))
    control_r = 0.06 + 0.35 * (1.0 - angular_distance / np.pi)
    control_r = float(np.clip(control_r, 0.06, 0.38))

    verts = [
        (theta1, 1.0),
        (theta1, control_r),
        (theta2, control_r),
        (theta2, 1.0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
    ]

    path = MplPath(verts, codes)
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        transform=ax.transData,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)


# ============================================================
# 7. 核心圆环图
# ============================================================
def plot_fc_circle(fc, ch_names, title, out_path, n_lines, masked_pairs):
    ch_names = [str(x) for x in ch_names]
    n_channels = len(ch_names)

    node_order = make_node_order(ch_names)
    node_angles = make_node_angles(ch_names, node_order)

    # mask 边索引
    masked = resolve_masked_pairs(ch_names, masked_pairs)
    masked_idx_set = {
        tuple(sorted((i, j))) for i, j, _, _ in masked
    }

    # 所有上三角边，并排除 mask 边；mask 边稍后统一画成黑色
    rows, cols = np.triu_indices(n_channels, k=1)
    values = fc[rows, cols]

    candidates = []
    for i, j, value in zip(rows, cols, values):
        if tuple(sorted((int(i), int(j)))) in masked_idx_set:
            continue
        if np.isfinite(value):
            candidates.append((abs(float(value)), int(i), int(j), float(value)))

    # 取绝对值最强的 n_lines 条
    candidates.sort(key=lambda x: x[0], reverse=True)
    strongest = candidates[: min(n_lines, len(candidates))]

    # Figure / polar axes
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_alpha(0.0)

    # 给右侧 colorbar 留空间
    ax = fig.add_axes([0.055, 0.075, 0.79, 0.84], projection="polar")
    ax.patch.set_alpha(0.0)

    # 从 12 点钟方向开始，顺时针
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.34)
    ax.set_axis_off()

    norm = Normalize(vmin=VMIN, vmax=VMAX)
    cmap = plt.get_cmap(COLORMAP)

    # ---------- 普通 FC 边 ----------
    for _, i, j, value in strongest:
        draw_connection(
            ax,
            node_angles[i],
            node_angles[j],
            color=cmap(norm(value)),
            linewidth=NORMAL_LINEWIDTH,
            alpha=NORMAL_LINE_ALPHA,
            zorder=1,
        )

    # ---------- 被 mask 的边：纯黑、加粗 ----------
    for i, j, a, b in masked:
        draw_connection(
            ax,
            node_angles[i],
            node_angles[j],
            color="black",
            linewidth=MASKED_LINEWIDTH,
            alpha=1.0,
            zorder=2,
        )
        print(f"  [MASK] {a} <-> {b}")

    # ---------- 外圈节点 ----------
    # 节点颜色仅用于区分电极位置，不参与 FC 数值表达
    position_rank = {name: idx for idx, name in enumerate(node_order)}
    node_colors = [
        plt.get_cmap("Spectral")(position_rank[name] / max(1, n_channels - 1))
        for name in ch_names
    ]

    node_width = 2.0 * np.pi / n_channels * 0.95
    ax.bar(
        node_angles,
        height=0.115,
        width=node_width,
        bottom=1.0,
        color=node_colors,
        edgecolor="black",
        linewidth=1.7,
        align="center",
        zorder=3,
    )

    # ---------- 电极文字 ----------
    for theta, name in zip(node_angles, ch_names):
        # 由于我们使用 theta_offset + 顺时针显示，这里用显示角度处理文字旋转
        deg = (90.0 - np.degrees(theta)) % 360.0

        if 90.0 < deg < 270.0:
            rotation = deg + 180.0
            ha = "right"
        else:
            rotation = deg
            ha = "left"

        ax.text(
            theta,
            1.165,
            name,
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=8.6 if n_channels >= 64 else 10.0,
            color="black",
            zorder=4,
        )

    # ---------- 标题：稍微下移 ----------
    fig.suptitle(
        title,
        x=0.45,
        y=TITLE_Y,
        fontsize=14,
        color="black",
    )

    # ---------- Colorbar：右边居中 ----------
    cax = fig.add_axes(COLORBAR_RECT)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=9, colors="black", width=1.0)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1.0)
    cax.patch.set_alpha(0.0)

    # 保存透明 PNG
    fig.savefig(
        out_path,
        dpi=DPI,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.10,
    )
    plt.close(fig)


# ============================================================
# 8. 单文件 + 主程序
# ============================================================
def plot_one_file(npz_path):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"找不到文件: {npz_path}")

    fc, ch_names, meta = load_fc_and_names(
        npz_path,
        plot_mode=PLOT_MODE,
        window_idx=WINDOW_IDX,
    )

    if PLOT_MODE == "mean":
        title = (
            f"{meta['dataset']} | {meta['subject']} | "
            f"Mean FC ({meta['n_windows']} windows)"
        )
    else:
        title = (
            f"{meta['dataset']} | {meta['subject']} | "
            f"FC window {WINDOW_IDX}"
        )

    n_lines = N_LINES_64 if len(ch_names) >= 64 else N_LINES_26
    out_path = OUT_DIR / f"{npz_path.stem}_fc_circle_masked.png"

    print(f"\n[FILE] {npz_path.name}")
    print(f"  dataset={meta['dataset']} subject={meta['subject']}")
    print(f"  fc_shape={fc.shape}, channels={len(ch_names)}, n_lines={n_lines}")

    plot_fc_circle(
        fc,
        ch_names,
        title,
        out_path,
        n_lines=n_lines,
        masked_pairs=MASKED_PAIRS,
    )

    print(f"  [OK] saved -> {out_path}")
    return out_path


def main():
    outputs = []
    for file_path in FILES:
        outputs.append(plot_one_file(file_path))

    print("\n全部完成：")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
