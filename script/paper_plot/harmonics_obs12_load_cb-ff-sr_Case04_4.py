import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import json
import os
import numpy as np



OBS_Number = 12
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
FIG_W = 3.5       # 双栏 7.0，单栏 3.5
FIG_H = 2.0
ML = 0.45          # 左侧留白（给 y 轴标签）
MR = 0.1          # 右侧留白
MT = 0.1          # 顶部留白
MB = 0.35          # 底部留白（给 x 轴标签）

# -----------
title = ''
filename = f'{script_name}.svg'
x_name = 'Harmonic Order'
y_name = 'SPL (dB)'
# -----------
data_1_path = fr"data\Case04\Case04_Rotor_OBS{OBS_Number:04d}_Harmonics.csv"
data_1 = pd.read_csv(data_1_path, sep=",", header=0)  # 读取数据

# -----------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))  # 创建图形和坐标轴对象
fig.subplots_adjust(left=ML/FIG_W, right=1-MR/FIG_W, top=1-MT/FIG_H, bottom=MB/FIG_H)
ax.set_xlabel(x_name)              # 设置X轴标签
xmin, xmax, xstep = -0, 45, 5
ax.set_xlim([xmin, xmax])
ax.set_xticks(np.arange(xmin, xmax + 1, xstep))
ax.set_ylabel(y_name)              # 设置Y轴标签
ymin, ymax, ystep = 10, 110, 20
ax.set_ylim([ymin, ymax])
ax.set_yticks(np.arange(ymin, ymax + 1, ystep))
# ----------- 散点图
# 数据
ax.plot(data_1['Harmonic Order'], data_1['SPL_FF_Load(dB)'], label='Free-field', color='grey', marker='o', alpha=0.5, zorder=3, linestyle='none')
ax.plot(data_1['Harmonic Order'], data_1['SPL_SR_Load(dB)'], label='Reflected', color=colors[0], marker='s', alpha=0.8, zorder=2, linestyle='none')
ax.plot(data_1['Harmonic Order'], data_1['SPL_merged_Load(dB)'], label='Combined', color=colors_wong[1], marker='^', alpha=0.8, zorder=2, linestyle='none')

# # ----------- 差异连接线 (Difference Lines)
# White mask to hide lines inside markers (zorder=1.5, between lines and visible points)
# ax.plot(data_1['Harmonic Order'], data_1['SPL_Load(dB)'], color='white', marker='o', alpha=1, zorder=1.5, linestyle='none')
# ax.plot(data_2['Harmonic Order'], data_2['SPL_FF_Load(dB)'], color='white', marker='s', alpha=1, zorder=1.5, linestyle='none')
# x = data_1['Harmonic Order']
# y1 = data_1['SPL_Load(dB)']
# y2 = data_2['SPL_FF_Load(dB)']
# # 确保索引对齐 (Assuming aligned by row index as per user instruction)
# # 如果需要按列对齐，应在此时确保 x, y1, y2 长度和顺序一致
# mask_pos = y2 >= y1
# mask_neg = y2 < y1
# if mask_pos.any():
#     ax.vlines(x[mask_pos], y1[mask_pos], y2[mask_pos], colors=colors[0], alpha=0.8, zorder=1) # Red-ish
# if mask_neg.any():
#     ax.vlines(x[mask_neg], y1[mask_neg], y2[mask_neg], colors='grey', alpha=0.5, zorder=1) # Green-ish
# ----------- 图例
ax.legend(
    ncol=1,                                 # 保持ncol列布局
    loc='upper right',                      # 图例自身的锚点
    bbox_to_anchor=(0.98, 0.98),              # 锚定到坐标轴的位置
)                                             # 显示图例
# -----------
plt.savefig(os.path.join(output_dir, f'{filename}'), transparent=True)  # 保存图片
#plt.show()                                     # 显示图形
print(f"Export plot: {os.path.join(output_dir, f'{filename}')}")
