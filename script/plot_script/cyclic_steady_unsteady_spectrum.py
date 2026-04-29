"""
循环谱定常/非定常/厚度噪声连续频谱对比

从循环谱分析的 SteadySpectrum.csv 读取定常和非定常分量的连续 SPL 频谱，
与 FreqDomain.csv 中的厚度噪声 SPL 叠加对比，展示三者在全频段的分布。

输入:
  - {prefix}_SteadySpectrum.csv   (Frequency, Steady_SPL, Unsteady_SPL)
  - {prefix}_FreqDomain.csv       (Frequency, SPL_Thickness)
输出:
  - plot/cyclic_steady_unsteady_spectrum/OBS{NNNN}.png
"""

import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import json
import os

# ---------- 用户配置 ----------
CASE_DIR = r"Case05"
CASE_PREFIX = "Case05_Rotor"
THICKNESS_COL = "SPL_Thickness(dB)"  # FF-only: SPL_Thickness(dB); FF+SR: SPL_FF_Thickness(dB)
# ------------------------------


OBS_range = range(1, 13)
script_name = os.path.basename(__file__).split(".")[0]
output_dir = os.path.join("plot", script_name)
os.makedirs(output_dir, exist_ok=True)

for OBS_Number in OBS_range:
    title = None
    filename = f"OBS{OBS_Number:04d}.png"
    x_name = "Frequency (Hz)"
    y_name = "SPL (dB)"

    # ---------- 读取数据 ----------
    obs_str = f"{CASE_PREFIX}_OBS{OBS_Number:04d}"
    steady_path = os.path.join(CASE_DIR, f"{obs_str}_FF_SteadySpectrum.csv")
    steady_data = pd.read_csv(steady_path, sep=",", header=0)
    fd_path = os.path.join(CASE_DIR, f"{obs_str}_FreqDomain.csv")
    fd_data = pd.read_csv(fd_path, sep=",", header=0)

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
    ax.set_ylabel(y_name)
#     ax.set_ylim(bottom=0)

    # 厚度噪声 — 灰色
    ax.plot(fd_data["Frequency(Hz)"], fd_data[THICKNESS_COL],
            color="grey", alpha=0.7, linewidth=0.5, label="Thickness")
    # 定常载荷 — colors[0]
    ax.plot(steady_data["Frequency(Hz)"], steady_data["Steady_SPL(dB)"],
            color=colors[0], alpha=0.9, label="Steady Load (cyclic)")
    # 非定常载荷 — colors[1]
    ax.plot(steady_data["Frequency(Hz)"], steady_data["Unsteady_SPL(dB)"],
            color=colors[1], alpha=0.9, label="Unsteady Load (cyclic)")

    # ---------- 图例 ----------
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0)

    # ---------- 保存 ----------
    plt.savefig(os.path.join(output_dir, filename), dpi=600)
    plt.close()
    print(f"Export plot: {os.path.join(output_dir, filename)}")
