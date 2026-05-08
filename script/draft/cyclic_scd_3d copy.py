"""
循环谱密度 SCD 瀑布图

横轴为频率 f (Hz)，每条曲线对应一个循环频率 α (Hz)，
曲线沿 y 轴（α）从前到后排列。z 轴为 log10(PSD / Pa²/Hz)。

输入:
  - {prefix}_SCD_3D.npz  (f, alpha, f_bpf, alpha_bpf, scd_power, scd_psd)
输出:
  - plot/cyclic_scd_3d/OBS{NNNN}.png
"""

import numpy as np
import scienceplots
import matplotlib.pyplot as plt
import json
import os

# ---------- 用户配置 ----------
CASE_DIR = r"data\Case01"
CASE_PREFIX = "Case01_Rotor"
BPF = 46.9698
MAX_ALPHA = BPF * 5  # 显示的 α 上限 (Hz)
Z_BASE = -10.0        # z 轴下限 (log10 Pa²/Hz)
Z_MAX = 0.0
# ------------------------------


OBS_range = range(1, 13)
script_name = os.path.basename(__file__).split(".")[0]
output_dir = os.path.join(".\script\draft\plot", script_name)
os.makedirs(output_dir, exist_ok=True)

for OBS_Number in OBS_range:
    filename = f"OBS{OBS_Number:04d}.png"

    # ---------- 读取数据 ----------
    obs_str = f"{CASE_PREFIX}_OBS{OBS_Number:04d}"
    npz_path = os.path.join(CASE_DIR, f"{obs_str}_SCD_3D.npz")

    data = np.load(npz_path)
    freq = np.asarray(data["f"])
    alpha = np.asarray(data["alpha"])
    scd = np.asarray(data["scd_psd"])  # (N_alpha, N_freq), Pa²/Hz

    # 截取显示范围
    alpha_mask = alpha <= MAX_ALPHA
    alpha = alpha[alpha_mask]
    scd = scd[alpha_mask, :]

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
    # 更新全局 rcParams
    plt.rcParams.update(plot_config)

    # ---------- 绘图 ----------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$\alpha$ (Hz)")
    ax.set_zlabel(r"$\log_{10}(\mathrm{PSD}\ \mathrm{Pa^2/Hz})$")

    scd_log = np.log10(scd + 1e-20)
    # z_max = float(np.max(scd_log))

    # 每条 α 绘制一条曲线，统一颜色，无填充
    LINE_COLOR = "0.15"
    for i, a in enumerate(alpha):
        ax.plot(
            freq,
            np.full_like(freq, a),
            scd_log[i, :],
            color=LINE_COLOR,
            linewidth=0.5,
        )

    # ---- 坐标轴范围 ----
    ax.set_xlim(freq[0], freq[-1])
    ax.set_ylim(alpha[0], alpha[-1])
    ax.set_zlim(Z_BASE, Z_MAX)

    # ---- 视角 ----
    ax.view_init(elev=22, azim=-60)

    # ---------- 保存 ----------
    plt.savefig(os.path.join(output_dir, filename), dpi=600)
    plt.close()
    print(f"Export plot: {os.path.join(output_dir, filename)}")
