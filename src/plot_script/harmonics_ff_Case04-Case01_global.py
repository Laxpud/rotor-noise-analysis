import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import json
import os


OBS_range = range(1, 13)
# 获得当前脚本文件名并去掉扩展名, 并创建输出目录
script_name = os.path.basename(__file__).split('.')[0]
output_dir = os.path.join('plot', script_name)
os.makedirs(output_dir, exist_ok=True)

for OBS_Number in OBS_range:
     # -----------
    title = None
    filename = f'OBS{OBS_Number:04d}.png'
    x_name = 'Harmonic Order'
    y_name = 'SPL (dB)'
    # -----------
    data_1_path = fr'.\Case01\Case01_Rotor_OBS{OBS_Number:04d}_Harmonics.csv'
    data_1 = pd.read_csv(data_1_path, sep=",", header=0)  # 读取数据
    data_2_path = fr'.\Case04\Case04_Rotor_OBS{OBS_Number:04d}_Harmonics.csv'
    data_2 = pd.read_csv(data_2_path, sep=",", header=0)  # 读取数据

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

    # -----------
    fig, ax = plt.subplots()  # 创建图形和坐标轴对象，尺寸由全局配置决定
    ax.set_xlabel(x_name)              # 设置X轴标签
    ax.set_xlim([-1, 47])
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylabel(y_name)              # 设置Y轴标签
    ax.set_ylim([10, 110])
    # ----------- 散点图
    # White mask to hide lines inside markers (zorder=1.5, between lines and visible points)
    ax.scatter(data_1['Harmonic Order'], data_1['SPL(dB)'], color='white', marker='o', alpha=1, zorder=1.5, linewidths=scatter_lw)
    ax.scatter(data_2['Harmonic Order'], data_2['SPL_FF(dB)'], color='white', marker='s', alpha=1, zorder=1.5, linewidths=scatter_lw)
    # 数据
    ax.scatter(data_1['Harmonic Order'], data_1['SPL(dB)'], label='OWSGE', color='grey', marker='o', alpha=0.5, zorder=3, linewidths=scatter_lw)
    ax.scatter(data_2['Harmonic Order'], data_2['SPL_FF(dB)'], label='IWSGE', color=colors[0], marker='s', alpha=0.8, zorder=2, linewidths=scatter_lw)

    # ----------- 差异连接线 (Difference Lines)
    x = data_1['Harmonic Order']
    y1 = data_1['SPL(dB)']
    y2 = data_2['SPL_FF(dB)']
    # 确保索引对齐 (Assuming aligned by row index as per user instruction)
    # 如果需要按列对齐，应在此时确保 x, y1, y2 长度和顺序一致
    mask_pos = y2 >= y1
    mask_neg = y2 < y1
    if mask_pos.any():
        ax.vlines(x[mask_pos], y1[mask_pos], y2[mask_pos], colors=colors[0], alpha=0.8, zorder=1) # Red-ish
    if mask_neg.any():
        ax.vlines(x[mask_neg], y1[mask_neg], y2[mask_neg], colors='grey', alpha=0.5, zorder=1) # Green-ish

    # ----------- 图例
    ax.legend(
        ncol=2,                                 # 保持2列布局
        loc='lower right',                      # 图例自身的锚点：右下角
        bbox_to_anchor=(1.0, 1.0),              # 锚定到坐标轴的(1,1.0)位置（x轴最右、y轴最上）
        borderaxespad=0,                        # 图例与锚点的间距（可微调，0为紧贴）
        # frameon=False,                        # 可选：去掉图例边框，更美观
        # handletextpad=0.5,                    # 可选：图例符号与文字的间距
        # columnspacing=1.0                     # 可选：列之间的间距
    )                                           # 显示图例                             # 显示图例
    # -----------
    plt.savefig(os.path.join(output_dir, f'{filename}'), dpi=600)  # 保存图片
    #plt.show()                                     # 显示图形
    print(f"Export plot: {os.path.join(output_dir, f'{filename}')}")