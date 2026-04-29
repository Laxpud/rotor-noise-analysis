"""
循环相干度随谐频阶次变化

从 CyclicCoherence.csv 中提取各 BPF 谐频处的循环相干度 γ(k·BPF, k·BPF)，
展示"周期性"随谐频阶次的衰减规律。γ ≈ 1 表示该阶谐频的声压为纯周期信号
（定常载荷主导），γ ≈ 0 表示该频率处能量虽在 BPF 谐频上，但来自随机涨落
（非定常载荷主导）。

输入:
  - {prefix}_CyclicCoherence.csv  (alpha(Hz) + 频率列)
  - {prefix}_CyclicSummary.csv    (BPF(Hz))
输出:
  - plot/cyclic_harmonic_coherence/OBS{NNNN}.png
"""

import pandas as pd
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import json
import os

# ---------- 用户配置 ----------
CASE_DIR = r"Case01"
CASE_PREFIX = "Case01_Rotor"
# ------------------------------


OBS_range = range(1, 13)
script_name = os.path.basename(__file__).split(".")[0]
output_dir = os.path.join("plot", script_name)
os.makedirs(output_dir, exist_ok=True)

for OBS_Number in OBS_range:
    title = None
    filename = f"OBS{OBS_Number:04d}.png"
    x_name = "Harmonic Order"
    y_name = "Cyclic Coherence"

    # ---------- 读取数据 ----------
    obs_str = f"{CASE_PREFIX}_OBS{OBS_Number:04d}"
    coh_path = os.path.join(CASE_DIR, f"{obs_str}_CyclicCoherence.csv")
    summary_path = os.path.join(CASE_DIR, f"{obs_str}_CyclicSummary.csv")

    coh_data = pd.read_csv(coh_path, sep=",", header=0)
    summary_data = pd.read_csv(summary_path, sep=",", header=0)
    bpf = summary_data["BPF(Hz)"].values[0]

    # 提取每阶谐频 (k*BPF) 处的循环相干度
    # CyclicCoherence.csv: 第一列为 alpha(Hz)，其余列为各频率的相干度
    alpha_col = coh_data.columns[0]
    freq_cols = [float(c) for c in coh_data.columns[1:]]
    freq_array = np.array(freq_cols)

    harmonic_orders = []
    harmonic_coherence = []

    for k in range(1, 46):  # 1 到 45 阶谐频
        target_alpha = k * bpf
        target_freq = k * bpf

        # 找到 alpha 最接近的行
        alpha_row = np.argmin(np.abs(coh_data[alpha_col].values - target_alpha))
        # 找到频率最接近的列索引
        freq_col_idx = np.argmin(np.abs(freq_array - target_freq))

        if alpha_row < len(coh_data):
            coh_value = coh_data.iloc[alpha_row, freq_col_idx + 1]  # +1 跳过 alpha 列
            harmonic_orders.append(k)
            harmonic_coherence.append(coh_value)

    harmonic_orders = np.array(harmonic_orders)
    harmonic_coherence = np.array(harmonic_coherence)

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
    ax.set_xlim([-1, 47])
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylabel(y_name)
    ax.set_ylim([0, 1.05])

    # 循环相干度散点
    ax.scatter(harmonic_orders, harmonic_coherence,
               color=colors[0], marker="o", alpha=0.8, zorder=3,
               linewidths=scatter_lw, label="Cyclic Coherence")

    # γ=1 参考线（纯周期信号）
    ax.axhline(y=1.0, color="grey", alpha=0.3, linewidth=0.5, linestyle="--")

    # ---------- 图例 ----------
    ax.legend(ncol=1, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0)

    # ---------- 保存 ----------
    plt.savefig(os.path.join(output_dir, filename), dpi=600)
    plt.close()
    print(f"Export plot: {os.path.join(output_dir, filename)}")
