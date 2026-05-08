import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
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
    x_name = 'Time (ms)'
    y_name = 'Sound Pressure (Pa)'
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
    fig, ax = plt.subplots(figsize=(8, 2))  # 创建图形和坐标轴对象
    ax.set_xlabel(x_name)              # 设置X轴标签
    ax.set_ylabel(y_name)              # 设置Y轴标签
    # ax.set_xlim(left = 213, right = 426)
    # ax.set_ylim(bottom = -2.3, top = 2.3)  # 设置Y轴范围
    ax.set_title(title) # 设置标题
    ax.xaxis.set_major_locator(MultipleLocator(50))
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=9))  # nbins参数控制大致刻度数量
    #ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # nbins参数控制大致刻度数量
    # ----------- 线图
    data_range = slice(0, 2700)
    x_data_1, y_data_1 = data_1['Time'][data_range], data_1['Load'][data_range]
    x_data_2, y_data_2 = data_2['Time'][data_range], data_2['Load'][data_range]
    x_data_3, y_data_3 = data_1['Time'][data_range], data_1['Thickness'][data_range]
    x_data_4, y_data_4 = data_2['Time'][data_range], data_2['Thickness'][data_range]
    ax.plot(x_data_1, y_data_1, label='OWSGE Load', color='grey', linestyle='-', alpha=0.9, zorder=3)
    ax.plot(x_data_2, y_data_2, label='IWSGE Load', color=colors[0], linestyle='--', alpha=0.9, zorder=4)
    ax.plot(x_data_3, y_data_3, label='OWSGE Thickness', color=colors_wong[1], linestyle='-', alpha=0.9, zorder=1)
    ax.plot(x_data_4, y_data_4, label='IWSGE Thickness', color=colors_wong[2], linestyle='--', alpha=0.9, zorder=2)
    x_min = min(x_data_1.min(), x_data_2.min(), x_data_3.min(), x_data_4.min())
    x_max = max(x_data_1.max(), x_data_2.max(), x_data_3.max(), x_data_4.max())
    ax.set_xlim(left=x_min, right=x_max)
    # -----------
    # Add alternating background color blocks
    period_points = 180
    x_values = x_data_1.values
    num_points = len(x_values)
    
    for i in range(0, num_points, period_points):
        if (i // period_points) % 2 == 0:
            start_idx = i
            end_idx = min(i + period_points - 1, num_points - 1)
            
            # Ensure we have valid indices
            if start_idx < num_points:
                x_start = x_values[start_idx]
                x_end = x_values[end_idx]
                
                # If end_idx is the last point, we might want to extend slightly if needed, 
                # but for now let's just use the data points.
                # Actually, to make it look continuous, we should probably use the start of the next block as the end of the current block
                # if it exists, to avoid gaps.
                if i + period_points < num_points:
                    x_end = x_values[i + period_points]
                
                ax.axvspan(x_start, x_end, facecolor='gray', alpha=0.1, zorder=1, linewidth=0)
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
    