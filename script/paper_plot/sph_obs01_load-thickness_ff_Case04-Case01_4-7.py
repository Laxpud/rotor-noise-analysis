import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator
import json
import os


OBS_range = range(1, 2)
# 获得当前脚本文件名并去掉扩展名, 并创建输出目录
script_name = os.path.basename(__file__).split('.')[0]
output_dir = ".\script\paper_plot"
os.makedirs(output_dir, exist_ok=True)

for OBS_Number in OBS_range:
    # -----------
    title = None
    filename = f'{script_name}.svg'
    x_name = 'Azimuth (deg)'
    y_name = 'Load Noise Sound Pressure (Pa)'
    # -----------
    data_1_path = fr"data\Case01\Case01_Rotor_OBS{OBS_Number:04d}_FF.csv"
    data_1 = pd.read_csv(data_1_path, sep=",", header=0)  # 读取数据
    data_2_path = fr"data\Case04\Case04_Rotor_OBS{OBS_Number:04d}_FF.csv"
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
    palettes = plot_config.pop('palettes', {})  # 自定义调色板，不能传 rcParams
    colors_wong = palettes.get('wong', [])  # 直接从全局配置中获取颜色列表

    # 更新全局 rcParams
    plt.rcParams.update(plot_config)

    # -----------
    fig, ax = plt.subplots()  # 创建图形和坐标轴对象
    ax.set_xlabel(x_name)              # 设置X轴标签
    ax.set_ylabel(y_name)              # 设置Y轴标签
    # ax.set_xlim(left = 213, right = 426)
    # ax.set_ylim(bottom = -2.3, top = 2.3)  # 设置Y轴范围
    ax.set_title(title) # 设置标题
    ax.xaxis.set_major_locator(MultipleLocator(50))
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=9))  # nbins参数控制大致刻度数量
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # nbins参数控制大致刻度数量
    # ----------- 线图
    data_range_4 = slice(180*4, 180*5)
    data_range_7 = slice(180*7, 180*8)
    # Assuming 180 points correspond to 360 degrees
    x_azimuth = np.linspace(0, 360, 180)
    # OWSGE Data
    y_owsge_4 = data_1['Load'].values[data_range_4]
    y_iwsge_4 = data_2['Load'].values[data_range_4]
    y_iwsge_7 = data_2['Load'].values[data_range_7]
    # Plotting
    ax.plot(x_azimuth, y_owsge_4, label='OWSGE', color='grey', linestyle='-', alpha=0.5, zorder=2)
    ax.plot(x_azimuth, y_iwsge_4, label='IWSGE 4th', color=colors[0], linestyle='-.', alpha=0.8, zorder=2)
    ax.plot(x_azimuth, y_iwsge_7, label='IWSGE 7th', color=colors_wong[1], linestyle='--', alpha=0.9, zorder=3)
    # Set X-axis limits
    ax.set_xlim(left=0, right=360)
    ax.xaxis.set_major_locator(MultipleLocator(60)) # Tick every 90 degrees
    # -----------
    
    # ----------- 图例
    ax.legend(
        ncol=4,                                 # 保持4列布局
        loc='lower right',                      # 图例自身的锚点：右下角
        bbox_to_anchor=(1.0, 1.0),              # 锚定到坐标轴的(1,1.0)位置（x轴最右、y轴最上）
        borderaxespad=0,                        # 图例与锚点的间距（可微调，0为紧贴）
        # frameon=False,                        # 可选：去掉图例边框，更美观
        # handletextpad=0.5,                    # 可选：图例符号与文字的间距
        # columnspacing=1.0                     # 可选：列之间的间距
    )                                           # 显示图例
    # -----------
    plt.savefig(os.path.join(output_dir, f'{filename}'), dpi=600)  # 保存图片
    #plt.show()                                     # 显示图形
    print(f"Export plot: {os.path.join(output_dir, f'{filename}')}")
    