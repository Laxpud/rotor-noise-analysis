import numpy as np
from scipy import signal

class BandContributionAnalyzer:
    """频带贡献分析器 - 用于分析频谱能量在不同频带的分布"""
    
    def __init__(self, freqs):
        """
        参数:
            freqs: 原始频率数据数组
        """
        self.freqs = np.asarray(freqs)
    
    def create_octave_bands(self, fraction=3, f_low=10, f_high=20000):
        """
        创建倍频程频带
        
        参数:
            fraction: 倍频程分数 (1=1/1倍频程, 3=1/3倍频程, 12=1/12倍频程)
            f_low: 最低中心频率 (默认为10Hz，以更好地覆盖旋翼基频)
            f_high: 最高中心频率
            
        返回:
            bands: 频带字典列表
        """
        # 标准中心频率 (ISO标准，扩展至低频以支持旋翼噪声分析)
        # 增加了 1Hz 到 12.5Hz 的低频部分
        base_freqs = [
            1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5,
            16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 
            250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 
            2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
        ]
        
        # 选择在范围内的中心频率
        center_freqs = [f for f in base_freqs if f_low <= f <= f_high]
        
        bands = []
        for f_center in center_freqs:
            # 计算频带边界
            if fraction == 1:  # 1/1倍频程
                f_lower = f_center / np.sqrt(2)
                f_upper = f_center * np.sqrt(2)
            elif fraction == 3:  # 1/3倍频程
                f_lower = f_center / (2 ** (1/6))
                f_upper = f_center * (2 ** (1/6))
            elif fraction == 12:  # 1/12倍频程
                f_lower = f_center / (2 ** (1/24))
                f_upper = f_center * (2 ** (1/24))
            else:
                raise ValueError("fraction必须是1, 3或12")
            
            bands.append({
                'center': f_center,
                'lower': f_lower,
                'upper': f_upper,
                'bandwidth': f_upper - f_lower
            })
        
        return bands
    
    def calculate_band_energy(self, spectrum, bands, p_ref=2e-6):
        """
        计算每个频带的能量
        
        参数:
            spectrum: 频谱幅值（线性）
            bands: 频带列表
            p_ref: 参考声压值 (默认2e-6)
            
        返回:
            band_energies: 频带能量字典
        """
        band_energies = []
        
        for band in bands:
            # 找到频带内的频率索引
            indices = np.where((self.freqs >= band['lower']) & (self.freqs <= band['upper']))[0]
            
            if len(indices) > 0:
                # 计算频带内的能量（频谱幅值的平方和）
                band_spectrum = spectrum[indices]
                energy = np.sum(band_spectrum ** 2)
                
                # 计算等效声压级（假设频谱已校准）
                # 这里简化处理，实际应用中需要根据校准系数调整
                energy_dB = 10 * np.log10(energy / (p_ref**2) + 1e-12)  # 避免log(0)
                
                # 计算能量占比
                total_energy = np.sum(spectrum ** 2)
                energy_ratio = energy / total_energy if total_energy > 0 else 0
                
                band_energies.append({
                    'center_freq': band['center'],
                    'energy': energy,
                    'energy_dB': energy_dB,
                    'energy_ratio': energy_ratio,
                    'indices': indices
                })
        
        return band_energies
    
    def analyze_band_contribution(self, spectrum, method='octave', fraction=3, f_low=10, p_ref=2e-6):
        """
        分析频带贡献
        
        参数:
            spectrum: 频谱幅值
            method: 分析方法 ('octave'或'custom')
            fraction: 倍频程分数
            f_low: 最低中心频率 (默认10Hz)
            p_ref: 参考声压值 (默认2e-6)
            
        返回:
            analysis_result: 分析结果
        """
        if method == 'octave':
            bands = self.create_octave_bands(fraction=fraction, f_low=f_low)
            band_energies = self.calculate_band_energy(spectrum, bands, p_ref=p_ref)
            band_type = f'Low-frequency extended 1/{fraction} octave'
        else:
            # 自定义频带
            custom_bands = [
                {'center': 50, 'lower': 20, 'upper': 100},     # 超低频
                {'center': 125, 'lower': 100, 'upper': 160},   # 低频
                {'center': 250, 'lower': 160, 'upper': 315},   # 中低频
                {'center': 500, 'lower': 315, 'upper': 630},   # 中频
                {'center': 1000, 'lower': 630, 'upper': 1250}, # 中高频
                {'center': 2000, 'lower': 1250, 'upper': 2500},# 高频
                {'center': 4000, 'lower': 2500, 'upper': 5000},# 超高频
            ]
            band_energies = self.calculate_band_energy(spectrum, custom_bands, p_ref=p_ref)
            band_type = 'Custom frequency band'
        
        # 计算统计信息
        if band_energies:
            energies = [band['energy'] for band in band_energies]
            ratios = [band['energy_ratio'] for band in band_energies]
            
            # 找到主导频带
            max_energy_idx = np.argmax(energies)
            dominant_band = {
                'center_freq': band_energies[max_energy_idx]['center_freq'],
                'energy_ratio': ratios[max_energy_idx],
                'energy_dB': band_energies[max_energy_idx]['energy_dB']
            }
            
            # 计算累积能量
            sorted_indices = np.argsort(energies)[::-1]
            cumulative_energy = 0
            dominant_bands = []
            
            for idx in sorted_indices:
                cumulative_energy += ratios[idx]
                dominant_bands.append({
                    'center_freq': band_energies[idx]['center_freq'],
                    'energy_ratio': ratios[idx],
                    'cumulative_ratio': cumulative_energy
                })
                if cumulative_energy >= 0.8:  # 80%能量
                    break
            
            result = {
                'band_type': band_type,
                'band_energies': band_energies,
                'dominant_band': dominant_band,
                'dominant_bands': dominant_bands[:5],  # 前5个主导频带
                'total_energy': np.sum(spectrum ** 2),
                'energy_distribution': {
                    'low_freq': np.sum([e['energy'] for e in band_energies 
                                       if e['center_freq'] < 250]),  # <250Hz
                    'mid_freq': np.sum([e['energy'] for e in band_energies 
                                       if 250 <= e['center_freq'] < 2000]),  # 250-2000Hz
                    'high_freq': np.sum([e['energy'] for e in band_energies 
                                        if e['center_freq'] >= 2000])  # >=2000Hz
                }
            }
        else:
            result = {}
        
        return result

class PeakFrequencyAnalyzer:
    """峰值频率分析器 - 仅用于线性幅值频谱分析"""
    
    def __init__(self, freqs):
        """
        参数:
            freqs: 原始频率数据数组
        """
        self.freqs = np.asarray(freqs)
    
    def find_peaks(self, spectrum, height=None, distance=None, prominence=None, 
                         width=None, rel_height=0.5):
        """
        使用scipy的find_peaks函数识别峰值
        
        参数:
            spectrum: 线性幅值频谱
            height: 最小高度阈值
            distance: 峰值之间最小距离(点数)
            prominence: 最小突出度
            width: 最小宽度
            rel_height: 计算宽度的相对高度
            
        返回:
            peak_indices: 峰值索引
            peak_properties: 峰值属性字典
        """
        peak_indices, peak_properties = signal.find_peaks(
            spectrum, 
            height=height,
            distance=distance,
            prominence=prominence,
            width=width,
            rel_height=rel_height
        )
        
        return peak_indices, peak_properties
    
    def identify_fundamental_and_harmonics(self, peak_freqs, peak_magnitudes, 
                                          all_freqs=None, all_magnitudes=None,
                                          tolerance=0.05, max_harmonic_order=30):
        """
        识别基频和谐频
        
        参数:
            peak_freqs: 峰值频率数组 (仅用于识别基频)
            peak_magnitudes: 峰值幅值数组 (仅用于识别基频)
            all_freqs: 所有频率点的数组 (用于提取谐频)
            all_magnitudes: 所有幅值点的数组 (用于提取谐频)
            tolerance: 基频检测容差
            max_harmonic_order: 用于基频识别的最大谐波阶数限制
            
        返回:
            fundamental_freq: 基频估计
            harmonic_indices: 谐频索引列表 (相对于all_freqs)
        """
        if len(peak_freqs) < 2:
            return None, []
        
        # 确保输入是numpy数组
        peak_freqs = np.asarray(peak_freqs)
        peak_magnitudes = np.asarray(peak_magnitudes)
        
        # 尝试找到基频
        possible_fundamentals = []
        
        for i, f1 in enumerate(peak_freqs):
            for j, f2 in enumerate(peak_freqs[i+1:], i+1):
                ratio = f2 / f1
                
                # 检查是否接近整数比
                nearest_int = round(ratio)
                # 使用max_harmonic_order作为基频识别的限制
                if 2 <= nearest_int <= max_harmonic_order:
                    if abs(ratio - nearest_int) / nearest_int < tolerance:
                        # 候选基频
                        candidate_f0 = f1
                        candidate_f1 = f2 / nearest_int
                        
                        # 选择较小的作为候选基频
                        candidate = min(candidate_f0, candidate_f1)
                        
                        # 计算加权分数（基于两个峰值的幅值）
                        weight = peak_magnitudes[i] + peak_magnitudes[j]
                        possible_fundamentals.append((candidate, weight, nearest_int))
        
        if not possible_fundamentals:
            return None, []
        
        # 选择最可能的基频（基于加权投票）
        fundamental_votes = {}
        for f, weight, order in possible_fundamentals:
            f_rounded = round(f, 1)
            if f_rounded in fundamental_votes:
                fundamental_votes[f_rounded] += weight
            else:
                fundamental_votes[f_rounded] = weight
        
        fundamental_freq = max(fundamental_votes, key=fundamental_votes.get) if fundamental_votes else None
        
        if fundamental_freq is None or all_freqs is None:
            return fundamental_freq, []
        
        # 识别基频的倍频（谐频）- 从所有频率点中提取
        harmonic_indices = []
        
        # 不再限制谐频数量，直到超出频率范围
        current_order = 1
        max_freq = np.max(all_freqs)
        
        while True:
            target_freq = fundamental_freq * current_order
            
            if target_freq > max_freq:
                break
            
            # 在所有频率点中寻找最接近的倍频
            if len(all_freqs) > 0:
                freq_diff = np.abs(all_freqs - target_freq)
                min_idx = np.argmin(freq_diff)
                
                # 计算相对误差
                if target_freq > 0:
                    relative_diff = freq_diff[min_idx] / target_freq
                else:
                    relative_diff = float('inf')
                
                # 如果误差在容忍范围内，添加到谐波列表
                if relative_diff < tolerance:
                    harmonic_indices.append(min_idx)
                
            current_order += 1
        
        return fundamental_freq, harmonic_indices
    
    def analyze_spectrum(self, spectrum, **find_peaks_kwargs):
        """
        频谱分析主函数
        
        参数:
            spectrum: 线性幅值频谱
            **find_peaks_kwargs: 传递给find_peaks函数的参数
            
        返回:
            analysis_result: 分析结果字典
        """
        # 使用scipy方法检测峰值
        peak_indices, peak_properties = self.find_peaks(spectrum, **find_peaks_kwargs)
        
        # 获取峰值频率和幅值
        peak_freqs = self.freqs[peak_indices]
        if 'peak_heights' in peak_properties:
            peak_magnitudes = peak_properties['peak_heights']
        else:
            peak_magnitudes = spectrum[peak_indices]
        
        # 识别基频和谐频 - 使用所有频率点
        fundamental_freq, harmonic_indices = self.identify_fundamental_and_harmonics(
            peak_freqs, peak_magnitudes,
            all_freqs=self.freqs,  # 传入所有频率点
            all_magnitudes=spectrum,  # 传入所有幅值点
            tolerance=0.05,
            max_harmonic_order=30
        )
        
        # 整理结果
        result = {
            'peak_freqs': peak_freqs,               # 所有峰值频率
            'peak_magnitudes': peak_magnitudes,     # 所有峰值幅值
            'peak_indices': peak_indices,           # 所有峰值索引
            'num_peaks': len(peak_freqs),           # 峰值总数
            'fundamental_freq': fundamental_freq,   # 基频
            'harmonic_indices': harmonic_indices,   # 谐频索引（相对于原始数据）
            'harmonic_freqs': self.freqs[harmonic_indices] if len(harmonic_indices) > 0 else [],  # 谐频
            'harmonic_magnitudes': spectrum[harmonic_indices] if len(harmonic_indices) > 0 else [],  # 谐频幅值
            'peak_properties': peak_properties      # 峰值属性
        }
        
        return result
