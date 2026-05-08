"""
积分循环谱 I(α) = ∫|SCD(f, α)| df

沿频率方向对循环谱密度积分，得到各循环频率 α 处的总循环平稳能量。
峰值出现在 BPF 整数倍处，表明该循环频率处存在显著的周期性能量；
BPF 谐频处的峰值高度反映该阶循环平稳性的强弱。

输入:
  - {prefix}_IntegratedCyclicSpectrum.csv  (alpha(Hz), I(alpha))
输出:
  - plot/cyclic_integrated_spectrum/OBS{NNNN}.png
"""

import pandas as pd
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import json
import os

# ---------- 用户配置 ----------
CASE_DIR = r"data\Case01"
CASE_PREFIX = "Case01_Rotor"
BPF = 46.9698  # 叶片通过频率 (Hz)，用于设置 x 轴刻度
# ------------------------------


OBS_range = range(1, 13)
script_name = os.path.basename(__file__).split(".")[0]
output_dir = os.path.join(".\script\draft\plot", script_name)
os.makedirs(output_dir, exist_ok=True)

for OBS_Number in OBS_range:
    title = None
    filename = f"OBS{OBS_Number:04d}.png"
    x_name = r"$\alpha$ (Hz)"
    y_name = r"$I(\alpha)$"

    # ---------- 读取数据 ----------
    obs_str = f"{CASE_PREFIX}_OBS{OBS_Number:04d}"
    ics_path = os.path.join(CASE_DIR, f"{obs_str}_IntegratedCyclicSpectrum.csv")

    ics_data = pd.read_csv(ics_path, sep=",", header=0)

    # ---------- 样式设置 ----------
    plt.style.use(["science"])
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    json_path = os.path.join(os.path.dirname(__file__), "plot_config.json")
    with open(json_path, "r", encoding="utf-8") as f:
        plot_config = json.load(f)
    scatter_lw = plot_config.pop("scatter.linewidths", 1.0)
    plt.rcParams.update(plot_config)

    # ---------- 绘图 ----------
    fig, ax = plt.subplots()
    ax.set_xlabel(x_name)
    ax.xaxis.set_major_locator(MultipleLocator(BPF * 5))
    ax.set_ylabel(y_name)
    ax.set_ylim(bottom=0)

    # 积分循环谱曲线
    ax.plot(ics_data["alpha(Hz)"], ics_data["I(alpha)"],
            color=colors[0], alpha=0.9, linewidth=0.75)

    # 标注 BPF 整数倍位置（竖虚线）
    max_alpha = ics_data["alpha(Hz)"].max()
    for k in range(1, int(max_alpha / BPF) + 1, 5):
        ax.axvline(x=k * BPF, color="grey", alpha=0.15, linewidth=0.35, linestyle="--")

    # ---------- 保存 ----------
    plt.savefig(os.path.join(output_dir, filename), dpi=600)
    plt.close()
    print(f"Export plot: {os.path.join(output_dir, filename)}")
