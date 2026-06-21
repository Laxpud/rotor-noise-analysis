import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import json
import os


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
FIG_W = 3.5       # 双栏 7.0，单栏 3.5
FIG_H = 2.0
ML = 0.50          # 左侧留白（给 y 轴标签）
MR = 0.15          # 右侧留白
MT = 0.10          # 顶部留白（给 legend）
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
azimuth_start_idx, azimuth_end_idx = 30, 121
data_range = slice(180*5+azimuth_start_idx, 180*5+azimuth_end_idx)
x_azimuth = np.arange(azimuth_start_idx, azimuth_end_idx) * 2
x_data_1, y_data_1 = data_1['Time'][data_range], data_1['Load'][data_range]
# x_data_2, y_data_2 = data_2['Time'][data_range], data_2['Load'][data_range]
x_data_3, y_data_3 = data_3['Time'][data_range], data_3['Load'][data_range]
x_data_4, y_data_4 = data_4['Time'][data_range], data_4['Load'][data_range]
X_data_5, y_data_5 = data_5['Time'][data_range], data_5['Load'][data_range]
ax.plot(x_azimuth, y_data_1, label='OWSGE', color='grey', linestyle='-', alpha=0.9, zorder=1)
# ax.plot(x_azimuth, y_data_2, label='IWSGE-2.0R', color=colors_wong[2], linestyle=':', alpha=0.9, zorder=2)
ax.plot(x_azimuth, y_data_3, label='IWSGE-1.5R', color=colors_wong[1], linestyle='-.', alpha=0.9, zorder=3)
ax.plot(x_azimuth, y_data_4, label='IWSGE-1.0R', color=colors[0], linestyle='--',  alpha=0.9, zorder=4)
ax.plot(x_azimuth, y_data_5, label='IWSGE-0.5R', color=colors_wong[3], linestyle=(0, (8, 2, 1.5, 2, 1.5, 2)),  alpha=0.9, zorder=5)
ax.set_title(title) # 设置标题
# Set X-axis limits
ax.set_xlabel(x_name)              # 设置X轴标签
xmin, xmax, xstep = 60, 240, 30
ax.set_xlim([xmin, xmax])
ax.set_xticks(np.arange(xmin, xmax + xstep, xstep))
# ax.set_xlim(left=min(x_exp), right=max(x_exp))
ax.set_ylabel(y_name)              # 设置Y轴标签
ymin, ymax, ystep = -2, 2.4, 1
ax.set_ylim([ymin, ymax])
# ax.set_yticks(np.arange(ymin, ymax, ystep))
ax.set_yticks([-2,-1,0,1,2])

# -----------
# 添加局部放大子图
x_zoom_min, x_zoom_max = 90, 120
zoom_mask = (x_azimuth >= x_zoom_min) & (x_azimuth <= x_zoom_max)
zoom_color = "0.7"

# [left, bottom, width, height]，坐标是相对于主坐标轴 ax 的比例
axins = ax.inset_axes([0.45, 0.5, 0.45, 0.45])
axins.set_facecolor("none")
axins.add_patch(Rectangle((0, 0), 1, 1,
                          transform=axins.transAxes,
                          facecolor=zoom_color,
                          alpha=0.22,
                          edgecolor="none",
                          zorder=-10))

axins.plot(x_azimuth[zoom_mask], y_data_1.to_numpy()[zoom_mask],
           color='grey', linestyle='-', alpha=0.9)
# axins.plot(x_azimuth[zoom_mask], y_data_2.to_numpy()[zoom_mask],
#            color=colors_wong[2], linestyle=':', alpha=0.9)
axins.plot(x_azimuth[zoom_mask], y_data_3.to_numpy()[zoom_mask],
           color=colors_wong[1], linestyle='-.', alpha=0.9)
axins.plot(x_azimuth[zoom_mask], y_data_4.to_numpy()[zoom_mask],
           color=colors[0], linestyle='--', alpha=0.9)
axins.plot(x_azimuth[zoom_mask], y_data_5.to_numpy()[zoom_mask],
           color=colors_wong[3], linestyle=(0, (8, 2, 1.5, 2, 1.5, 2)), alpha=0.9)

axins.set_xlim(x_zoom_min, x_zoom_max)

# 自动设置局部 y 范围，留一点边距
zoom_y = np.concatenate([
    y_data_1.to_numpy()[zoom_mask],
    # y_data_2.to_numpy()[zoom_mask],
    y_data_3.to_numpy()[zoom_mask],
    y_data_4.to_numpy()[zoom_mask],
    y_data_5.to_numpy()[zoom_mask],
])
y_pad = 0.08 * (zoom_y.max() - zoom_y.min())
axins.set_ylim(zoom_y.min() - y_pad, zoom_y.max() + y_pad)

axins.grid(False)
axins.set_xticks([])
axins.set_yticks([])
axins.tick_params(left=False, bottom=False, right=False, top=False,
                  labelleft=False, labelbottom=False)
axins.set_xlabel("")
axins.set_ylabel("")

# 在主图上标出被放大的区域，并连到子图
ax.axvspan(x_zoom_min, x_zoom_max, color=zoom_color, alpha=0.18, linewidth=0)
for spine in axins.spines.values():
    spine.set_linewidth(0.5)
    spine.set_edgecolor("0.25")


# -----------
plt.savefig(os.path.join(output_dir, f'{filename}'), transparent=True)  # 保存图片
#plt.show()                                     # 显示图形
print(f"Export plot: {os.path.join(output_dir, f'{filename}')}")
