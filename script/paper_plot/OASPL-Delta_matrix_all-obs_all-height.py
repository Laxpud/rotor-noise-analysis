import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
import json
import os
import numpy as np


# 本脚本用于绘制组合噪声相对自由场噪声的 OASPL 变化量：
#   Delta OASPL = OASPL_merged - OASPL_FF
# OWSGE 只有自由场结果，没有组合噪声，因此不参与本图。
# Case03-05 每个状态都读取一份自由场空间 OASPL 和一份组合场空间 OASPL，
# 然后按 OBS 1-12 恢复为 4 行 x 3 列矩阵，每个子图展示同一观测点下 3 个状态的 Delta OASPL。

# 获得当前脚本文件名并去掉扩展名，用作输出 SVG 文件名。
# 输出目录保持在 paper_plot 下，便于和同目录其他论文图脚本统一管理。
script_name = os.path.basename(__file__).split('.')[0]
output_dir = r".\script\paper_plot"
os.makedirs(output_dir, exist_ok=True)

# ----------- 全局尺寸设置
# 论文图统一使用 scienceplots 风格，具体字体、网格、tick 和图例参数由 plot_config.json 管理。
plt.style.use(['science'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# 读取同目录绘图配置。使用脚本所在目录定位，避免从其他工作目录运行时找错文件。
json_path = os.path.join(os.path.dirname(__file__), 'plot_config.json')
with open(json_path, 'r', encoding='utf-8') as f:
    plot_config = json.load(f)

# plot_config.json 中的自定义键不能直接传给 rcParams，需要先取出。
palettes = plot_config.pop('palettes', {})
colors_wong = palettes.get('wong', [])
plt.rcParams.update(plot_config)
# hatch 用于黑白打印时辅助区分状态，线宽保持与论文图细线风格一致。
plt.rcParams['hatch.linewidth'] = 0.35

# ----------- 排版边距（绝对英寸，确保不同栏宽图片的绘图区域在 Inkscape 中对齐）
FIG_W = 7.0       # 双栏 7.0，单栏 3.5
FIG_H = 4.0
ML = 0.20          # 左侧留白（给 y 轴标签）
MR = 0.10          # 右侧留白
MT = 0.35          # 顶部留白（给 legend）
MB = 0.35          # 底部留白（给 x 轴标签）

# -----------
title = ''
filename = f'{script_name}.svg'
x_name = r'$\Delta$OASPL (dB)'
y_name = ''
# -----------
# 旋翼半径，单位 m。数据文件中的 Y/Z 是物理坐标，论文图中以 R 为单位标注观测点位置。
ROTOR_RADIUS = 1.5

# 每个元素为 (图例标签, 自由场数据路径, 组合场数据路径, 条形颜色)。
# Case03-05 是有组合噪声的 IWSGE 状态；OWSGE 没有组合噪声，所以不放入本图。
case_specs = [
    ('IWSGE-1.5R', r"data\Case03\Case03_SPL_FF.dat", r"data\Case03\Case03_SPL_merged.dat", colors_wong[1]),
    ('IWSGE-1.0R', r"data\Case04\Case04_SPL_FF.dat", r"data\Case04\Case04_SPL_merged.dat", colors[0]),
    ('IWSGE-0.5R', r"data\Case05\Case05_SPL_FF.dat", r"data\Case05\Case05_SPL_merged.dat", colors_wong[3]),
]


def read_oaspl_data(data_path):
    """读取 Tecplot OBS 总声压级数据，并按 IOBS 建立索引。

    数据契约：
    - 文件前 3 行是 Tecplot 头信息，所以用 skiprows=3 跳过；
    - 后续列顺序固定为 X, Y, Z, SPL(dB), IOBS；
    - IOBS 是观测点编号，后续用它恢复 OBS 物理矩阵；
    - 返回值以 IOBS 为索引，避免依赖文件行顺序。
    """
    data = pd.read_csv(
        data_path,
        sep=r"\s+",
        skiprows=3,
        names=['X', 'Y', 'Z', 'SPL(dB)', 'IOBS'],
        engine='python',
    )
    data['IOBS'] = data['IOBS'].astype(int)
    return data.set_index('IOBS').sort_index()


def format_radius_coordinate(value):
    """把物理坐标按旋翼半径 R 无量纲化。

    例如 Y=4.5 m 且 R=1.5 m 时显示为 3R；
    Z=-0.375 m 显示为 -0.25R。
    """
    ratio = value / ROTOR_RADIUS
    if np.isclose(ratio, round(ratio)):
        return f'{int(round(ratio))}R'
    return f'{ratio:.2f}R'.rstrip('0').rstrip('.')


# 读入每个状态的 FF 和 merged 数据，并计算 Delta OASPL。
# delta_data[label] 的行索引仍然是 IOBS，列包含 X/Y/Z 坐标和 Delta OASPL。
delta_data = {}
for label, ff_path, merged_path, color in case_specs:
    ff_data = read_oaspl_data(ff_path)
    merged_data = read_oaspl_data(merged_path)

    # 以自由场数据为坐标参考；merged 和 FF 应共享同一 OBS 坐标与编号。
    delta = ff_data[['X', 'Y', 'Z']].copy()
    delta['DeltaOASPL(dB)'] = merged_data['SPL(dB)'] - ff_data['SPL(dB)']
    delta_data[label] = delta

# OBS 编号固定为 1-12，对应 4 行 x 3 列观测点矩阵。
obs_numbers = np.arange(1, 13)
reference_data = next(iter(delta_data.values()))
matrix_shape = (4, 3)

# 条形标签、颜色和纹理由 case_specs 统一派生，保证数据、图例和样式顺序一致。
bar_labels = [label for label, ff_path, merged_path, color in case_specs]
bar_colors = [color for label, ff_path, merged_path, color in case_specs]
bar_hatches = ['////', r'\\\\', '......']
bar_y = np.arange(len(bar_labels))

# 汇总所有 Delta OASPL，用于确定统一 x 轴范围。
# 如果未来数据同时出现正负值，则使用 0 居中的对称 diverging 轴；
# 如果当前数据全为正或全为负，则把 0 放在边界，减少无效空白。
all_delta = np.concatenate([
    data.loc[obs_numbers, 'DeltaOASPL(dB)'].to_numpy()
    for data in delta_data.values()
])
has_negative = all_delta.min() < 0
has_positive = all_delta.max() > 0
if has_negative and has_positive:
    x_abs = np.ceil((np.abs(all_delta).max() + 0.5) / 1) * 1
    xmin, xmax = -x_abs, x_abs
else:
    xmin = min(0, np.floor((all_delta.min() - 0.5) / 1) * 1)
    xmax = max(0, np.ceil((all_delta.max() + 0.5) / 1) * 1)

# -----------
# 创建 OBS 矩阵子图。sharex/sharey 让 12 个观测点的 Delta OASPL 可直接比较。
fig, axes = plt.subplots(matrix_shape[0], matrix_shape[1], figsize=(FIG_W, FIG_H), sharex=True, sharey=True)
fig.subplots_adjust(left=ML/FIG_W, right=1-MR/FIG_W, top=1-MT/FIG_H, bottom=MB/FIG_H, wspace=0.10, hspace=0.18)

# ----------- Delta OASPL 横向条形矩阵
legend_handles = []
for obs_number in obs_numbers:
    # 将 IOBS 从 1 基编号转换为 axes 的 0 基 row/col。
    # OBS 01-03 在第一行，OBS 04-06 在第二行，依此类推。
    row = (obs_number - 1) // matrix_shape[1]
    col = (obs_number - 1) % matrix_shape[1]
    ax = axes[row, col]

    # 当前 OBS 下 3 个 IWSGE 状态的组合场相对自由场变化量。
    delta_values = np.array([
        delta_data[label].loc[obs_number, 'DeltaOASPL(dB)']
        for label in bar_labels
    ])
    bars = ax.barh(
        bar_y,
        delta_values,
        left=0,
        height=0.65,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.15,
        alpha=0.85,
        zorder=3,
    )
    for bar, hatch in zip(bars, bar_hatches):
        bar.set_hatch(hatch)
    if obs_number == 1:
        legend_handles = bars

    # x=0 是 Delta OASPL 的物理参考线：右侧表示组合场高于自由场，左侧表示低于自由场。
    ax.axvline(0, color='black', linewidth=0.35, zorder=2)

    # 数值标注放在条形末端外侧；正值向右偏移，负值向左偏移。
    for y_pos, delta_value in zip(bar_y, delta_values):
        if delta_value >= 0:
            text_x = delta_value + 0.10
            ha = 'left'
        else:
            text_x = delta_value - 0.10
            ha = 'right'
        ax.text(
            text_x,
            y_pos,
            f'{delta_value:.2f}',
            ha=ha,
            va='center',
            zorder=4,
        )

    ax.set_xlim([xmin, xmax])
    ax.set_xticks(np.arange(xmin, xmax + 2, 3))
    ax.set_ylim([-0.5, len(bar_labels) - 0.5])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.text(
        0.97,
        0.92,
        rf'\textbf{{OBS-{obs_number:02d}}}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontweight='bold',
        zorder=4,
    )

    if row == 0:
        y_coord = reference_data.loc[obs_number, 'Y']
        ax.set_title(rf'$Y={format_radius_coordinate(y_coord)}$')
    if col == 0:
        z_coord = reference_data.loc[obs_number, 'Z']
        ax.set_ylabel(rf'$Z={format_radius_coordinate(z_coord)}$')
    else:
        ax.tick_params(labelleft=False)
    if row != matrix_shape[0] - 1:
        ax.tick_params(labelbottom=False)

fig.supxlabel(x_name, y=0.00)

# ----------- 图例
fig.legend(
    legend_handles,
    bar_labels,
    ncol=3,
    loc='lower right',
    bbox_to_anchor=(1-MR/FIG_W, 1-MT/FIG_H+0.04),
)
# -----------
plt.savefig(os.path.join(output_dir, f'{filename}'), transparent=True)
#plt.show()
print(f"Export plot: {os.path.join(output_dir, f'{filename}')}")
