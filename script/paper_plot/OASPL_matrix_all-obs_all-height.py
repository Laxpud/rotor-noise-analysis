import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import json
import os
import numpy as np


# 本脚本用于把不同 Case 的 OASPL 结果放回 OBS 观测点矩阵中对比。
# 数据文件是 Tecplot 风格的点数据：
#   1. 前 3 行是标题、变量名和 zone 信息；
#   2. 后续 12 行分别对应 OBS 1-12；
#   3. 每行字段含义为 X, Y, Z, SPL(dB), IOBS。
# 绘图时不按曲线顺序展示，而是把 IOBS 映射为 4 行 x 3 列的物理观测点矩阵，
# 每个子图中用 4 个横向条形比较 4 个 Case 在同一 OBS 位置的 OASPL。

# 获得当前脚本文件名并去掉扩展名，用作输出 SVG 文件名。
# 输出目录保持在 paper_plot 下，便于和同目录其他论文图脚本统一管理。
script_name = os.path.basename(__file__).split('.')[0]
output_dir = r".\script\paper_plot"
os.makedirs(output_dir, exist_ok=True)

# ----------- 全局尺寸设置
# 论文图统一使用 scienceplots 风格，具体字体、网格、tick 和图例参数由
# plot_config.json 管理；这样同目录脚本导出的图在 Inkscape/论文中风格一致。
plt.style.use(['science'])
# 获取 scienceplots 当前颜色循环。后面 Case04 继续沿用第一个默认主色，
# 以保持和已有脚本中的 IWSGE-1.0R/2.0R 视觉约定一致。
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
# 读取全局绘图配置文件。
# 使用 os.path.dirname(__file__) 获取当前脚本所在目录，避免从其他工作目录运行脚本时找错配置。
json_path = os.path.join(os.path.dirname(__file__), 'plot_config.json')
with open(json_path, 'r', encoding='utf-8') as f:
    plot_config = json.load(f)
# plot_config.json 中包含一些自定义键，不属于 matplotlib rcParams。
# 在更新 rcParams 前必须先 pop 出来，否则 plt.rcParams.update 会报未知参数错误。
scatter_lw = plot_config.pop('scatter.linewidths', 1.0) # 若没有此参数，则默认为 1.0
palettes = plot_config.pop('palettes', {})  # 自定义调色板，不能传 rcParams。
colors_wong = palettes.get('wong', [])  # Wong 调色板用于区分多个 Case，并保持色盲友好。
# 更新全局 rcParams，使字号、线宽、网格、LaTeX 字体等设置对本脚本全部图元生效。
plt.rcParams.update(plot_config)
# hatch 默认线宽通常比论文图轴框更粗；这里显式压到细线，保证黑白纹理只是辅助编码。
plt.rcParams['hatch.linewidth'] = 0.35

# ----------- 排版边距（绝对英寸，确保不同栏宽图片的绘图区域在 Inkscape 中对齐）
# 这里使用绝对英寸边距，而不是 tight_layout/constrained_layout。
# 原因是论文图需要在后处理软件中对齐绘图区，绝对边距可以让不同图片的轴框位置可控。
FIG_W = 7.0       # 双栏 7.0，单栏 3.5
FIG_H = 4.0
ML = 0.20          # 左侧留白（给 y 轴标签）
MR = 0.10          # 右侧留白
MT = 0.35          # 顶部留白（给 legend）
MB = 0.35          # 底部留白（给 x 轴标签）

# -----------
title = ''
filename = f'{script_name}.svg'
x_name = 'OASPL (dB)'
y_name = ''
# -----------
# 旋翼半径，单位 m。数据文件中的 Y/Z 是物理坐标，论文图中希望以 R 为单位展示观测点位置。
ROTOR_RADIUS = 1.5
# 每个元素为 (图例标签, 数据路径, 条形颜色)。
# Case01 使用自由场 OASPL；Case03-05 使用合成场 OASPL，与当前论文对比逻辑一致。
case_specs = [
    ('OWSGE', r"data\Case01\Case01_SPL_FF.dat", 'grey'),
    ('IWSGE-1.5R', r"data\Case03\Case03_SPL_merged.dat", colors_wong[1]),
    ('IWSGE-1.0R', r"data\Case04\Case04_SPL_merged.dat", colors[0]),
    ('IWSGE-0.5R', r"data\Case05\Case05_SPL_merged.dat", colors_wong[3]),
]


def read_oaspl_data(data_path):
    """读取 Tecplot OBS 总声压级数据，并按 IOBS 建立索引。

    数据契约：
    - 文件前 3 行不是数值数据，所以用 skiprows=3 跳过；
    - 后续列顺序固定为 X, Y, Z, SPL(dB), IOBS；
    - IOBS 是 OBS 观测点编号，后续用它恢复 4x3 空间矩阵位置；
    - 返回值以 IOBS 为索引，方便按 obs_number 精确取值，避免依赖行号。
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
    """把物理坐标按旋翼半径 R 无量纲化，避免图中混用 m 与 OBS 矩阵位置。

    例如 Y=4.5 m 且 R=1.5 m 时显示为 3R。
    非整数倍半径保留两位小数，并去掉末尾多余的 0，使 Z=-0.375 m 显示为 -0.25R。
    """
    ratio = value / ROTOR_RADIUS
    if np.isclose(ratio, round(ratio)):
        return f'{int(round(ratio))}R'
    return f'{ratio:.2f}R'.rstrip('0').rstrip('.')


# 读入所有 Case 数据。字典键使用图例标签，使后续绘图顺序和图例顺序来自同一份 case_specs。
case_data = {
    label: read_oaspl_data(data_path)
    for label, data_path, color in case_specs
}
# OBS 编号固定为 1-12。数据文件也包含 IOBS 字段，这里显式列出可避免后续矩阵维度隐式变化。
obs_numbers = np.arange(1, 13)
# 各 Case 的 OBS 坐标应相同，因此取第一个 Case 作为 Y/Z 标签的参考坐标源。
reference_data = next(iter(case_data.values()))
# OBS 矩阵为 4 行 x 3 列：
#   第 1 行：OBS 01-03；
#   第 2 行：OBS 04-06；
#   第 3 行：OBS 07-09；
#   第 4 行：OBS 10-12。
matrix_shape = (4, 3)
# 条形标签和颜色统一由 case_specs 派生，避免图例、数据和颜色三者顺序不一致。
bar_labels = [label for label, data_path, color in case_specs]
bar_colors = [color for label, data_path, color in case_specs]
# 条形纹理用于黑白打印时区分不同 Case。
# 颜色仍然承担屏幕阅读时的主识别功能；hatch 和黑色细边框提供灰度/复印场景下的冗余编码。
# 避免使用交叉纹理 xx：在窄横条中交叉线相位容易显得不一致，黑白打印也会过重。
bar_hatches = ['', '////', r'\\\\', '......']
# 每个子图内部的 y 位置只表示 4 个 Case 的排列顺序；实际 Case 名称放在顶部图例中。
bar_y = np.arange(len(bar_labels))
# 汇总所有 Case/OBS 的 SPL，用于计算统一 x 轴范围。
# 统一范围保证 12 个子图的条形长度可以直接横向比较。
all_spl = np.concatenate([
    data.loc[obs_numbers, 'SPL(dB)'].to_numpy()
    for data in case_data.values()
])
# 横向条形从 xmin 起画，而不是从 0 起画。
# OASPL 约为 90-110 dB，若从 0 起画会浪费大量空白；统一左基线可以突出 Case 间差异。
xmin = np.floor((all_spl.min() - 1) / 5) * 5
# 右侧额外留 3 dB 余量，用于放置条形末端的数值标注，避免文字被轴框裁切。
xmax = np.ceil((all_spl.max() + 3) / 5) * 5

# -----------
# 创建 OBS 矩阵子图。sharex/sharey 让所有子图坐标尺度一致，避免视觉比较时误读。
fig, axes = plt.subplots(matrix_shape[0], matrix_shape[1], figsize=(FIG_W, FIG_H), sharex=True, sharey=True)
# 手动设置子图间距，延续本目录论文图“绝对边距 + 可控绘图区”的排版方式。
fig.subplots_adjust(left=ML/FIG_W, right=1-MR/FIG_W, top=1-MT/FIG_H, bottom=MB/FIG_H, wspace=0.10, hspace=0.18)

# ----------- 横向条形矩阵
# legend_handles 只需要保存第一幅子图生成的 4 个条形对象，用来构造全图公共图例。
legend_handles = []
for obs_number in obs_numbers:
    # 将 IOBS 从 1 基编号转换为 matplotlib axes 的 0 基 row/col。
    # 这个映射是“严格对应 OBS 矩阵”的核心：OBS 01-03 在第一行，之后每 3 个换行。
    row = (obs_number - 1) // matrix_shape[1]
    col = (obs_number - 1) % matrix_shape[1]
    ax = axes[row, col]

    # 取当前 OBS 在 4 个 Case 中的 OASPL。
    # loc[obs_number] 依赖前面 set_index('IOBS')，这样即使数据文件行顺序变化也能按 OBS 编号取值。
    spl_values = np.array([
        case_data[label].loc[obs_number, 'SPL(dB)']
        for label in bar_labels
    ])
    # 使用 left=xmin 与 width=spl_values-xmin，使条形从统一的 SPL 左边界起画。
    # 这样可以在 dB 量级较高时仍保留对不同 Case 小差异的视觉分辨率。
    bars = ax.barh(
        bar_y,
        spl_values - xmin,
        left=xmin,
        height=0.65,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.15,
        alpha=0.85,
        zorder=3,
    )
    # matplotlib 的 hatch 需要逐个 patch 设置。
    # 因为 legend_handles 来自 OBS 01 的 bars，图例会自动继承这里的颜色、边框和纹理。
    for bar, hatch in zip(bars, bar_hatches):
        bar.set_hatch(hatch)

    # 全图图例不需要 12 份重复 handle，只保存第一个 OBS 的条形即可。
    if obs_number == 1:
        legend_handles = bars

    # 在每个条形末端外侧写具体 OASPL 数值。
    # x 位置使用 spl_value + 0.25，表示文字左边缘略微离开条形末端，避免贴边难读。
    for y_pos, spl_value in zip(bar_y, spl_values):
        ax.text(
            spl_value + 0.25,
            y_pos,
            f'{spl_value:.2f}',
            ha='left',
            va='center',
            # fontsize=6,
            zorder=4,
        )

    # 每个子图使用同一组 x/y 轴范围。
    # y 轴不显示 Case 名称，因为 Case 已由顶部图例说明，保留空间给矩阵结构和数值标注。
    ax.set_xlim([xmin, xmax])
    ax.set_xticks(np.arange(xmin, xmax + 1, 5))
    ax.set_ylim([-0.5, len(bar_labels) - 0.5])
    ax.set_yticks([])
    # 反转 y 轴，使 case_specs 中的第一个 Case 显示在最上方，和图例阅读顺序一致。
    ax.invert_yaxis()
    # OBS 编号放在每个子图右上角，作为该小图的观测点身份标识。
    # 使用 axes 坐标系而不是数据坐标系，保证位置不随 x 轴范围变化。
    ax.text(
        0.97,
        0.92,
        f'OBS {obs_number:02d}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        zorder=4,
    )

    # 第一行显示 Y/R 列坐标，用于标明矩阵的横向观测位置。
    # 只在第一行显示可以减少重复文字，同时保持矩阵列语义明确。
    if row == 0:
        y_coord = reference_data.loc[obs_number, 'Y']
        ax.set_title(rf'$Y={format_radius_coordinate(y_coord)}$')
    # 第一列显示 Z/R 行坐标，用于标明矩阵的垂向观测位置。
    # 只在第一列显示可以让 4 行高度信息清晰且不挤占其他子图空间。
    if col == 0:
        z_coord = reference_data.loc[obs_number, 'Z']
        ax.set_ylabel(rf'$Z={format_radius_coordinate(z_coord)}$')
    else:
        # 非第一列不显示 y 轴标签/刻度文字，避免重复信息造成拥挤。
        ax.tick_params(labelleft=False)
    if row != matrix_shape[0] - 1:
        # 非最后一行隐藏 x 轴刻度文字，只保留底部公共 x 轴语义。
        ax.tick_params(labelbottom=False)

# 全图公共 x 轴标签。使用 supxlabel 而不是单独给每个子图设置 xlabel，避免重复文字。
fig.supxlabel(x_name, y=0.02)

# ----------- 图例
# 图例放在矩阵上方右侧，与其他 all-height 对比图保持类似布局。
# bbox_to_anchor 使用绝对边距换算后的 figure 坐标，使图例位置跟 MT/MR 同步。
fig.legend(
    legend_handles,
    bar_labels,
    ncol=4,                                 # 保持ncol列布局
    loc='lower right',                      # 图例自身的锚点
    bbox_to_anchor=(1-MR/FIG_W, 1-MT/FIG_H+0.04),              # 锚定到矩阵顶部
)                                             # 显示图例
# -----------
plt.savefig(os.path.join(output_dir, f'{filename}'), transparent=True)  # 保存图片
#plt.show()                                     # 显示图形
print(f"Export plot: {os.path.join(output_dir, f'{filename}')}")
