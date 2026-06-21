import json
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd
import scienceplots



OBS_Number = 1
# 获得当前脚本文件名并去掉扩展名, 并创建输出目录
script_name = os.path.basename(__file__).split('.')[0]
output_dir = r".\script\paper_plot"
os.makedirs(output_dir, exist_ok=True)

# ----------- 全局尺寸设置
plt.style.use(['science'])
# 获取当前颜色循环
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
# 读取全局绘图配置文件
# 使用 os.path.dirname(__file__) 获取当前脚本所在目录，确保找到同目录下的 json 文件
json_path = os.path.join(os.path.dirname(__file__), 'plot_config.json')
with open(json_path, 'r', encoding='utf-8') as f:
    plot_config = json.load(f)
# 提取自定义非标准参数，避免 rcParams 报错
palettes = plot_config.pop('palettes', {})  # 自定义调色板，不能传 rcParams
colors_wong = palettes.get('wong', [])  # 直接从全局配置中获取颜色列表
# 更新全局 rcParams
plt.rcParams.update(plot_config)

# ----------- 排版边距（绝对英寸，确保不同栏宽图片的绘图区域在 Inkscape 中对齐）
FIG_W = 7.0       # 双栏 7.0，单栏 3.5
FIG_H = 2.0
ML = 0.50          # 左侧留白（给 y 轴标签）
MR = 0.15          # 右侧留白
MT = 0.20          # 顶部留白（给 legend）
MB = 0.35          # 底部留白（给 x 轴标签）


# -----------
title = ""
filename = f'{script_name}.svg'
x_name = 'Azimuth (deg)'
y_name = 'Sound Pressure (Pa)'
# -----------
data_1_path = fr"data\Case01\Case01_Rotor_OBS{OBS_Number:04d}_FF.csv"
data_1 = pd.read_csv(data_1_path, sep=",", header=0)  # 读取数据
# data_2_path = fr"data\Case02\Case02_Rotor_OBS{OBS_Number:04d}_FF.csv" # 数据错误，弃用
# data_2 = pd.read_csv(data_2_path, sep=",", header=0)  # 读取数据
data_3_path = fr"data\Case03\Case03_Rotor_OBS{OBS_Number:04d}_FF.csv"
data_3 = pd.read_csv(data_3_path, sep=",", header=0)  # 读取数据
data_4_path = fr"data\Case04\Case04_Rotor_OBS{OBS_Number:04d}_FF.csv"
data_4 = pd.read_csv(data_4_path, sep=",", header=0)  # 读取数据
data_5_path = fr"data\Case05\Case05_Rotor_OBS{OBS_Number:04d}_FF.csv"
data_5 = pd.read_csv(data_5_path, sep=",", header=0)  # 读取数据

# -----------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))  # 创建图形和坐标轴对象
fig.subplots_adjust(left=ML/FIG_W, right=1-MR/FIG_W, top=1-MT/FIG_H, bottom=MB/FIG_H)
# ----------- 线图
data_range = slice(0, 2700)
x_data_1, y_data_1 = data_1['Time'][data_range], data_1['Load'][data_range]
# x_data_2, y_data_2 = data_2['Time'][data_range], data_2['Load'][data_range]
x_data_3, y_data_3 = data_3['Time'][data_range], data_3['Load'][data_range]
x_data_4, y_data_4 = data_4['Time'][data_range], data_4['Load'][data_range]
X_data_5, y_data_5 = data_5['Time'][data_range], data_5['Load'][data_range]
ax.plot(x_data_1, y_data_1, label='OWSGE', color='grey', linestyle='-', alpha=0.9, zorder=1)
# ax.plot(x_data_2, y_data_2, label='IWSGE-2.0R', color=colors_wong[2], linestyle=':', alpha=0.9, zorder=2)
ax.plot(x_data_3, y_data_3, label='IWSGE-1.5R', color=colors_wong[1], linestyle='-.', alpha=0.9, zorder=3)
ax.plot(x_data_4, y_data_4, label='IWSGE-1.0R', color=colors[0], linestyle='--',  alpha=0.9, zorder=4)
ax.plot(X_data_5, y_data_5, label='IWSGE-0.5R', color=colors_wong[3], linestyle=(0, (8, 2, 1.5, 2, 1.5, 2)),  alpha=0.9, zorder=5)

ax.set_title(title) # 设置标题
# Set X-axis limits
ax.set_xlabel(x_name)              # 设置X轴标签
x_min = min(x_data_1.min(), x_data_3.min(), x_data_4.min(), X_data_5.min())
x_max = max(x_data_1.max(), x_data_3.max(), x_data_4.max(), X_data_5.max())
xstep = 50
ax.set_xlim(left=x_min, right=x_max)
ax.set_xticks(np.arange(x_min, x_max, xstep))
ax.set_ylabel(y_name)              # 设置Y轴标签
ymin, ymax, ystep = -2, 2.4, 1
ax.set_ylim([ymin, ymax])
# ax.set_yticks(np.arange(ymin, ymax, ystep))
ax.set_yticks([-2,-1,0,1,2])
# -----------
# Add alternating background color blocks
period_points = 180
x_values = x_data_1.values
num_points = len(x_values)

for i in range(0, num_points, period_points):
    if (i // period_points) % 2 == 0:
        start_idx = i
        end_idx = min(i + period_points - 1, num_points - 1)

        # Ensure we have valid indices
        if start_idx < num_points:
            x_start = x_values[start_idx]
            x_end = x_values[end_idx]

            # If end_idx is the last point, we might want to extend slightly if needed,
            # but for now let's just use the data points.
            # Actually, to make it look continuous, we should probably use the start of the next block as the end of the current block
            # if it exists, to avoid gaps.
            if i + period_points < num_points:
                x_end = x_values[i + period_points]

            ax.axvspan(x_start, x_end, facecolor='gray', alpha=0.1, zorder=1, linewidth=0)
# -----------

# ----------- 图例
ax.legend(
    ncol=5,                                 # 保持4列布局
    loc='lower right',                      # 图例自身的锚点：右下角
    bbox_to_anchor=(1.02, 1.0),              # 锚定到坐标轴的(1,1.0)位置（x轴最右、y轴最上）
)                                           # 显示图例
# -----------
plt.savefig(os.path.join(output_dir, f'{filename}'), transparent=True)  # 保存图片
#plt.show()                                     # 显示图形
print(f"Export plot: {os.path.join(output_dir, f'{filename}')}")
