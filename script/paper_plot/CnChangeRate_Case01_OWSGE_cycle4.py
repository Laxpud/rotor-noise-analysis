import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import json
import os
import numpy as np


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
scatter_lw = plot_config.pop('scatter.linewidths', 1.0) # 若没有此参数，则默认为 1.0
palettes = plot_config.pop('palettes', {})  # 自定义调色板，不能传 rcParams
colors_wong = palettes.get('wong', [])  # 直接从全局配置中获取颜色列表
# 更新全局 rcParams
plt.rcParams.update(plot_config)

# -----------
title = ''
filename = f'{script_name}.svg'
# -----------
data_1_path = r'.\data\Cn\Case01_Cn_distribution.csv'

# ============================================================
# 第 1 步：读入 Cn 分布数据
# 数据列示意：psi_rad, psi_deg, r_bar_0.125, r_bar_0.250, ...
# ============================================================
data_1 = pd.read_csv(data_1_path)

# 找出所有半径列的列名（即包含 'r_bar_' 的列）
r_cols = [col for col in data_1.columns if 'r_bar_' in col]
# 从列名中提取无量纲半径值（如 'r_bar_0.125' → 0.125）
radii = [float(col.replace('r_bar_', '')) for col in r_cols]

# ============================================================
# 第 2 步：计算 Cn 的时间变化率  d(Cn)/dt
# 方法：中心差分（二阶精度，无相位偏移）
#   np.gradient 对内部点用中心差分，端点用单侧差分
#   dt = 角度步长 [rad] / 旋转角速度 [rad/s]
# ============================================================
angular_step_rad = 2.0 / 180.0 * np.pi        # 每行间隔 2° → 弧度
omega_rad_per_s  = 0.651 * 340.0 / 1.5         # 旋转角速度 = 桨尖马赫数 × 声速 / 旋翼半径
dt = angular_step_rad / omega_rad_per_s        # 两行之间的物理时间间隔 [s]

data_1[r_cols] = np.gradient(data_1[r_cols].values, dt, axis=0)

# ============================================================
# 第 3 步：截取第 4 圈的数据
# 每 360°（2°/行 × 180 行）为完整一圈；多取 1 行到 row 720 是为了极坐标首尾闭合
# ============================================================
cycle = 4
cycle_start_deg = 360 * (cycle - 1)            # 第 4 圈的起始方位角 = 1080°
row_start = 180 * (cycle - 1)                  # 起始行号
row_end   = 180 * cycle                        # 结束行号（含）
data_1_cycle = data_1.iloc[row_start:row_end+1]

# 将方位角从绝对角度 [°] 转为以本圈起点为 0 的弧度
theta_abs_deg = data_1_cycle['psi_deg'].values
theta_rad = (theta_abs_deg - cycle_start_deg) * np.pi / 180.0

# ============================================================
# 第 4 步：构造绘图网格（极坐标）
# meshgrid 的结果形状为 (len(theta), len(radii))，与 Z 对齐
# ============================================================
R, Theta = np.meshgrid(radii, theta_rad)        # R: 径向坐标, Theta: 角向坐标
Z = data_1_cycle[r_cols].values                 # Z: d(Cn)/dt 的值矩阵

# ============================================================
# 第 5 步：绘制极坐标填充云图
# ============================================================
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# ----- 固定颜色范围（多图对比时保持相同色标）-----
vmin, vmax = -70, 70                         # ← 根据数据范围自行调整
levels = np.linspace(vmin, vmax, 41)         # 40 个等间距色阶

contour = ax.contourf(Theta, R, Z,
                      levels=levels,
                      vmin=vmin, vmax=vmax,
                      cmap='viridis',
                      extend='both')          # 超出 [vmin, vmax] 的值用三角箭头标记

cbar = plt.colorbar(contour, ax=ax, pad=0.1)
cbar.set_label('Cn Change Rate')

# 在 0° 方位角处画一条参考线（从最小半径到最大半径）
ax.plot([0, 0], [min(radii), max(radii)],
        color='black', linestyle='-', linewidth=0.35)

if title:
    ax.set_title(title)
ax.grid(True)

# -----------
plt.savefig(os.path.join(output_dir, f'{filename}'), transparent=True)  # 保存图片
#plt.show()                                     # 显示图形
print(f"Export plot: {os.path.join(output_dir, f'{filename}')}")