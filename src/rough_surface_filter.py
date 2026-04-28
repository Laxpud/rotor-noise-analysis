import numpy as np
import pandas as pd

class RoughSurfaceFilter:
    """
    粗糙表面滤波类 - 用于对输入的时间信号进行粗糙表面滤波处理。
    """
    def __init__(self):
        pass

    def filter_rough_surface(self, time_signal, wave_rms_height, grazing_angle):
        """
        对输入的时间信号进行粗糙表面滤波处理。

        参数:
        time_signal (np.array): 输入的时间信号的二维数组，第一列为时间，第二列为幅值。
        wave_rms_height (float): 波高的均方根值 (m)。
        grazing_angle (float):  掠射角 (radians).

        返回:
        freq (np.array): 频率点数组。
        fft (np.array): 输入时间信号的傅里叶变换值，包含频率点和对应的信号值。
        fft_corrected (np.array): 修正后的傅里叶变换值，包含频率点和对应的信号值。
        time_signal_corrected (np.array): 修正后的时间域信号，包含时间点和对应的信号值。
        """
        Z_air = 415.0
        Z_water = 1.5e6
        R0 = (Z_water - Z_air) / (Z_water + Z_air)
        C = 340

        time_intervals = np.diff(time_signal[0]) / 1000
        mean_interval = np.mean(time_intervals)
        fs = 1 / mean_interval
        freq = np.fft.rfftfreq(len(time_signal[0]), d=mean_interval)
        fft = np.fft.rfft(time_signal[1]) + 1e-10

        k = 2 * np.pi * freq / C
        Gamma = (2 * k * wave_rms_height * np.sin(grazing_angle))**2
        R_coh = R0 * np.exp(- Gamma / 2)

        fft_corrected = R_coh * fft

        time_signal_corrected = np.array([time_signal[0], np.fft.irfft(fft_corrected)])

        return freq, fft, fft_corrected, R_coh, time_signal_corrected

    def calculate_grazing_angle(FreeSurface_Direction, FreeSurface_Position, OBS_Position):
        """
        计算声波由坐标原点传播到水面再反射到观测点时的掠射角
        """
        # 确保输入为numpy数组
        normal = np.array(FreeSurface_Direction, dtype=float)
        surface_pos = np.array(FreeSurface_Position, dtype=float)
        obs_pos = np.array(OBS_Position, dtype=float)
        source_pos = np.array([0.0, 0.0, 0.0])  # 坐标原点

        # 归一化法向量
        normal = normal / np.linalg.norm(normal)

        # 计算镜像声源位置
        # 1. 声源到平面上某点的向量
        vec_ps = source_pos - surface_pos
        # 2. 声源到平面的距离（投影到法向量）
        dist = np.dot(vec_ps, normal)
        # 3. 镜像声源 S' = S - 2 * dist * n
        image_source = source_pos - 2 * dist * normal

        # 计算镜像声源到观测点的向量
        vec_image_to_obs = obs_pos - image_source

        # 计算掠射角
        # 掠射角是射线与平面的夹角，等于射线与法向量夹角的余角
        # sin(grazing_angle) = |dot(ray, normal)| / (|ray| * |normal|)

        numerator = np.abs(np.dot(vec_image_to_obs, normal))
        denominator = np.linalg.norm(vec_image_to_obs)

        if denominator == 0:
            return 0.0

        sin_grazing = numerator / denominator

        # 数值稳定性处理
        sin_grazing = np.clip(sin_grazing, 0.0, 1.0)

        grazing_angle = np.arcsin(sin_grazing)

        return grazing_angle

def main(
    file_path, 
    Sound_FreeField_file_list, 
    Sound_SurfaceReflection_file_list, 
    OBS_Position_list, 
    p_ref, 
    Radius, 
    wave_rms_height,
    FreeSurface_Direction, 
    FreeSurface_Position,
    cycles
    ):
    """
    
    """
    print(f"File_path: {file_path}")

    if len(Sound_FreeField_file_list) != len(Sound_SurfaceReflection_file_list):
        raise ValueError("Sound_FreeField_file_list and Sound_SurfaceReflection_file_list must have the same length.")
    if len(Sound_FreeField_file_list) != len(OBS_Position_list):
        raise ValueError("Sound_FreeField_file_list and OBS_Position_list must have the same length.")
    
    print(f'Total files number: {len(Sound_FreeField_file_list)}')
    # 创建输出文件名的前缀
    out_prefix = [f"{Sound_FreeField_file_list[i].replace('_FF.csv', '')}" for i in range(len(Sound_FreeField_file_list))]
    # 创建一个DataFrame来存储每个文件的SPLs
    SPLs_final_df = pd.DataFrame(columns=out_prefix)
    for i in range(len(Sound_FreeField_file_list)):
        print(f"\nFile index: {i}")
        OBS_Position = OBS_Position_list[i]
        print(f"OBS_Position: {OBS_Position}")
        Sound_FF = pd.read_csv(f"{file_path}\\{Sound_FreeField_file_list[i]}", header=0)
        print(f"Read sound_FF_file: {Sound_FreeField_file_list[i]}")
        Sound_SR = pd.read_csv(f"{file_path}\\{Sound_SurfaceReflection_file_list[i]}", header=0)
        print(f"Read sound_SR_file: {Sound_SurfaceReflection_file_list[i]}")

        # 将dataframe转换为nparray
        time_signal_FF = np.array([Sound_FF['Time'], Sound_FF['Total']])
        time_signal_SR = np.array([Sound_SR['Time'], Sound_SR['Total']])

        # 粗糙表面滤波
        grazing_angle = calculate_grazing_angle(FreeSurface_Direction*Radius, FreeSurface_Position*Radius, OBS_Position*Radius)
        print(f"Grazing angle: {grazing_angle}")
        print("Rough surface filter")
        freq_SR, fft_SR, fft_corrected_SR, R_coh, time_signal_SR_corrected = filter_rough_surface(
            time_signal_SR, wave_rms_height, grazing_angle=grazing_angle
            )
        print("Merge free field and surface reflection")
        time_signal_final = np.array([
            time_signal_SR_corrected[0],
            time_signal_FF[1]+time_signal_SR_corrected[1]-np.mean(time_signal_FF[1]+time_signal_SR_corrected[1])
            ])
        # 获得time_signal_final的频域
        if len(time_signal_final[0]) != len(time_signal_FF[0]) or len(time_signal_final[0]) != len(time_signal_SR[0]): 
            raise ValueError("Time signal of free field and final do not match.")
        N = len(time_signal_final[0])
        freq_final = np.fft.rfftfreq(N, d=np.mean(np.diff(time_signal_final[0]))/1000)
        fft_final = np.fft.rfft(time_signal_final[1]) + 1e-10

        # 获得自由场噪声频域
        freq_FF = np.fft.rfftfreq(N, d=np.mean(np.diff(time_signal_FF[0]))/1000)
        fft_FF = np.fft.rfft(time_signal_FF[1]) + 1e-10

        # 获得频域的幅度谱
        amp_fft_FF = 2.0/N * np.abs(fft_FF)
        amp_fft_FF[0] = amp_fft_FF[0] / 2.0
        amp_fft_SR = 2.0/N * np.abs(fft_SR)
        amp_fft_SR[0] = amp_fft_SR[0] / 2.0
        amp_fft_SR_corrected = 2.0/N * np.abs(fft_corrected_SR)
        amp_fft_SR_corrected[0] = amp_fft_SR_corrected[0] / 2.0
        R_coh = np.abs(R_coh)
        amp_fft_final = 2.0/N * np.abs(fft_final)
        amp_fft_final[0] = amp_fft_final[0] / 2.0
        # 根据幅度计算声压级
        SPL_fft_FF = 20 * np.log10(amp_fft_FF / p_ref)
        SPL_fft_SR = 20 * np.log10(amp_fft_SR / p_ref)
        SPL_fft_SR_corrected = 20 * np.log10(amp_fft_SR_corrected / p_ref)
        SPL_fft_final = 20 * np.log10(amp_fft_final / p_ref)
    
        # 比对所有频域数据的频率是否一致
        if not np.allclose(freq_FF, freq_SR):
            raise ValueError("Frequency of free field and surface reflection do not match.")
        if not np.allclose(freq_FF, freq_final):
            raise ValueError("Frequency of free field and final do not match.") 

        # 将频域结果输出到一个csv文件
        freq_domain_data = np.array([
            freq_FF, 
            SPL_fft_FF, 
            SPL_fft_SR, 
            SPL_fft_SR_corrected, 
            SPL_fft_final,
            R_coh,
            amp_fft_FF,
            amp_fft_SR,
            amp_fft_SR_corrected,
            amp_fft_final,
            ]).T
        freq_domain_data = pd.DataFrame(freq_domain_data, columns=['Frequency(Hz)', 'SPL_FF(dB)', 'SPL_SR(dB)', 'SPL_SR_corrected(dB)', 'SPL_Final(dB)', 'R_coh', 'amp_FF(Pa)', 'amp_SR(Pa)', 'amp_SR_corrected(Pa)', 'amp_Final(Pa)'])
        freq_domain_data.to_csv(f"{file_path}\\{out_prefix[i]}_FreqDomain.csv", index=False)
        print(f"Export freq domain data: {file_path}\\{out_prefix[i]}_FreqDomain.csv")
        
        # 将时域结果输出到一个csv中
        time_domain_data = np.array([
            time_signal_FF[0],
            time_signal_FF[1],
            time_signal_SR[1],
            time_signal_SR_corrected[1],
            time_signal_final[1]
            ]).T
        time_domain_data = pd.DataFrame(time_domain_data, columns=['Time(ms)', 'FF(Pa)', 'SR(Pa)', 'SR_corrected(Pa)', 'Final(Pa)'])
        time_domain_data.to_csv(f"{file_path}\\{out_prefix[i]}_TimeDomain.csv", index=False)
        print(f"Export time domain data: {file_path}\\{out_prefix[i]}_TimeDomain.csv")
        
        # 计算每个周期的声压级
        SPLs_final_df[out_prefix[i]] = calculate_SPLs(time_signal_final, p_ref, cycles)

    print("\n", SPLs_final_df)
    return time_signal_final, freq_domain_data, time_domain_data, SPLs_final_df

if __name__ == "__main__":
        
    file_path = r"Case04"
    Filename_list = [
        "Case04-1_Rotor",
        "Case04-2_Rotor",
        "Case04-3_Rotor",
    ]
    OBS_Numbers = 12
    OBS_Position_list = np.array([
        [0, 3, 0], 
        [0, 3, -0.25],
        [0, 3, -0.5],
        [0, 3, -0.75],
        [0, 4, 0.0],
        [0, 4, -0.25],
        [0, 4, -0.5],
        [0, 4, -0.75],
        [0, 5, 0.0],
        [0, 5, -0.25],
        [0, 5, -0.5],
        [0, 5, -0.75],
        ])
    if OBS_Numbers != len(OBS_Position_list):
        raise ValueError("OBS_Numbers must be equal to the length of OBS_Position_list")

    p_ref = 20e-6
    # 基于半径无量纲化的参数
    Radius = 1.5
    wave_rms_height = 0.02
    FreeSurface_Direction = np.array([0, 0, 1])
    FreeSurface_Position = np.array([0, 0, -1])
    cycles = 5
    
    # 组合 OBS_Numbers 和 Filename_list 中的元素形成要读取的文件名
    Sound_FreeField_file_list = [
        f"{Filename_list[i]}_OBS{j+1:04d}_FF.csv" for i in range(len(Filename_list)) for j in range(OBS_Numbers)
    ]
    print(Sound_FreeField_file_list)
    Sound_SurfaceReflection_file_list = [
        f"{Filename_list[i]}_OBS{j+1:04d}_SR.csv" for i in range(len(Filename_list)) for j in range(OBS_Numbers)
    ]
    print(Sound_SurfaceReflection_file_list)
    # 组合 Filename_list 和 OBS_Position_list 中的元素形成输入的 OBS_Position 列表
    OBS_Position_list = np.array([
        OBS_Position_list[i] for j in range(len(Filename_list)) for i in range(OBS_Numbers) 
    ])
    print(OBS_Position_list)

    # 调用主函数
    time_signal_final, freq_domain_data, time_domain_data, SPLs_final_df = main(
        file_path, 
        Sound_FreeField_file_list, 
        Sound_SurfaceReflection_file_list, 
        OBS_Position_list, 
        p_ref, 
        Radius, 
        wave_rms_height,
        FreeSurface_Direction, 
        FreeSurface_Position,
        cycles
        )
    # 按 Filename_list 输出 SPLs_final
    SPLs_final_filename_list = [f"{Filename_list[i].split('_')[0]}_SPLs_final.csv" for i in range(len(Filename_list))]
    # 将 SPLs_final_df 中的数据分组并输出，分组数量为SPLs_final_filename_list中的元素数量
    for i in range(len(SPLs_final_filename_list)):
        data = SPLs_final_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        # 修改列名为 OBS 编号
        data.columns = [f"OBS{j+1}" for j in range(OBS_Numbers)]
        # 在第一列插入一个名为 Cycles 的列，值为 1到 cycles
        data.insert(0, 'Cycles', range(1, cycles+1))
        # 输出结果，小数点后保留5位
        data = data.round(5)
        data.to_csv(f"{file_path}\\{SPLs_final_filename_list[i]}", index=False)
        print(f"Export SPLs_final data: {file_path}\\{SPLs_final_filename_list[i]}")
