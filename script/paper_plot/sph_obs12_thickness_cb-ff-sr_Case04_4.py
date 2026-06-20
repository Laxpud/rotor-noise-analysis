import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator
import json
import os


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
ML = 0.40          # 左侧留白（给 y 轴标签）
MR = 0.15          # 右侧留白
MT = 0.20          # 顶部留白（给 legend）
MB = 0.35          # 底部留白（给 x 轴标签）

# -----------
title = ""
filename = f'{script_name}.svg'
x_name = 'Azimuth (deg)'
y_name = 'Sound Pressure (Pa)'
# -----------
data_1_path = fr"data\Case04\Case04_Rotor_OBS{OBS_Number:04d}_FF.csv"
data_1 = pd.read_csv(data_1_path, sep=",", header=0)  # 读取数据
data_2_path = fr"data\Case04\Case04_Rotor_OBS{OBS_Number:04d}_SR.csv"
data_2 = pd.read_csv(data_2_path, sep=",", header=0)  # 读取数据
data_3_path = fr"data\Case04\Case04_Rotor_OBS{OBS_Number:04d}_merged.csv"
data_3 = pd.read_csv(data_3_path, sep=",", header=0)  # 读取数据

# -----------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))  # 创建图形和坐标轴对象
fig.subplots_adjust(left=ML/FIG_W, right=1-MR/FIG_W, top=1-MT/FIG_H, bottom=MB/FIG_H)
ax.set_xlabel(x_name)              # 设置X轴标签
ax.set_ylabel(y_name)              # 设置Y轴标签
# ax.set_xlim(left = 213, right = 426)
# ax.set_ylim(bottom = -2.3, top = 2.3)  # 设置Y轴范围
ax.set_title(title) # 设置标题
ax.xaxis.set_major_locator(MultipleLocator(50))
# ax.xaxis.set_major_locator(MaxNLocator(nbins=9))  # nbins参数控制大致刻度数量
# ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # nbins参数控制大致刻度数量
# ----------- 线图
data_range = slice(180*3+30, 180*3+120)
# Assuming 180 points correspond to 360 degrees
x_azimuth = np.linspace(0, 180, 90)
# OWSGE Data
y_1 = data_1['Thickness'].values[data_range]
y_2 = data_2['Thickness'].values[data_range]
y_3 = data_3['Thickness'].values[data_range]
# Plotting
ax.plot(x_azimuth, y_1, label='Free-field', color='grey', linestyle='-', alpha=0.5, zorder=2)
ax.plot(x_azimuth, y_2, label='Reflected', color=colors[0], linestyle='--', alpha=0.8, zorder=2)
ax.plot(x_azimuth, y_3, label='Combined', color=colors_wong[1], linestyle='-.', alpha=0.9, zorder=3)
# Set X-axis limits
ax.set_xlim(left=0, right=180)
ax.xaxis.set_major_locator(MultipleLocator(30)) # Tick every 30 degrees
# -----------

# ----------- 图例
ax.legend(
    ncol=3,                                 # 保持ncol列布局
    loc='lower right',                      # 图例自身的锚点：右下角
    bbox_to_anchor=(1.03, 1.0),              # 锚定到坐标轴的(1,1.0)位置（x轴最右、y轴最上）
)                                           # 显示图例
# -----------
plt.savefig(os.path.join(output_dir, f'{filename}'), transparent=True)  # 保存图片
#plt.show()                                     # 显示图形
print(f"Export plot: {os.path.join(output_dir, f'{filename}')}")
