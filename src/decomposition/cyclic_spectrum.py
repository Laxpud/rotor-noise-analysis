"""
循环平稳谱分析模块 (Cyclostationary Spectrum Analysis).

基于 FFT Accumulation Method (FAM) 实现循环谱密度 (Spectral Correlation
Density, SCD) 的计算，利用已知的叶片通过频率 (BPF) 将循环频率搜索
约束在 α = k·BPF (k = 0, 1, ..., K_max) 的离散值上，大幅降低计算量。

理论基础
--------
旋翼噪声具有循环平稳 (cyclostationary) 特性：其统计矩（尤其是自相关函数）
是时间的周期函数，周期由 BPF 决定。循环谱密度 S_x^α(f) 量化了信号在
频率 f 处的频谱分量与频率 f±α/2 处分量之间的统计相关性。

FAM 算法流程
------------
1. **信号分段**: 将时域信号按 N0 = fs/BPF（每个 BPF 周期的采样点数）
   划分为 M 个连续的数据段，每段覆盖恰好一个转子周期。
2. **2x 补零 FFT**: 对每段数据进行 2 倍补零（FFT 长度 = 2*N0），
   使得频率分辨率 df = fs/(2*N0)。补零的关键在于：循环频率 α = k·BPF
   对应的半频偏 α/2 = k·BPF/2 恰好落在 FFT 的整数 bin 上，
   避免了频谱插值引入的误差。
3. **循环周期图**: 对每个数据段的 FFT 结果 X_m(f)，计算
   S_x^α(f) = (1/M) * Σ_m X_m(f+α/2) · X_m*(f-α/2)
   其中 X_m* 表示复共轭。这一定义来自循环维纳-辛钦定理的频域形式。
4. **相干度计算**: γ_x^α(f) = |S_x^α(f)|² / (S_x^0(f+α/2) · S_x^0(f-α/2))
   α=0 时的 SCD 退化为经典功率谱密度 (PSD)。

2x 补零的原理说明
-----------------
FFT 频率分辨率 Δf = fs / N_fft。若 N_fft = N0，则 Δf = fs/N0 = BPF，
而 α/2 = k·BPF/2，对于奇数 k 会落在半 bin 位置（如 BPF/2, 3BPF/2, ...），
无法用整数索引访问。将 N_fft 设为 2*N0 后，Δf = BPF/2，α/2 恒为
Δf 的整数倍，保证所有频移操作都在准确的 bin 位置进行。

定常频谱重建逻辑
----------------
从循环相干度重建定常分量的 PSD：
    S_steady(f) = max_{α≠0} γ_x^α(f) · S_x^0(f)
直观含义：循环相干度 γ_x^α(f) 衡量了频率 f 处信号在多大程度上由
周期为 1/α 的周期过程产生。取所有非零 α 的最大相干度，乘以 PSD，
得到由所有周期性（定常）过程贡献的功率。

局限性
------
- FAM 的频率分辨率受限于 N_fft（Δf = fs/(2*N0) = BPF/2）。
- 信号段数 M 影响 SCD 估计的方差（M 越大方差越小）。
- 对于宽带非平稳噪声，FAM 的平稳性假设可能不满足。
- 补零不增加频率分辨率，仅改变 bin 对齐。

参考文献
--------
- Gardner, W. A. "Measurement of Spectral Correlation."
  IEEE Trans. ASSP, 1986.
- Antoni, J. "Cyclic spectral analysis in practice."
  Mechanical Systems and Signal Processing, 2007.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from signal_utils import P_REF


class CyclicSpectrumAnalyzer:
    """循环谱分析器 -- 基于循环平稳分析定量评估定常/非定常载荷贡献.

    本分析器不依赖厚度噪声参考信号，而是通过信号的统计循环平稳特性
    区分周期分量（定常载荷产生的确定性噪声）与随机分量（非定常载荷
    产生的随机噪声）。核心工具是循环谱密度 (SCD) 和循环相干度。

    Parameters
    ----------
    time_signal : np.ndarray
        形状为 (2, N) 的二维数组，第一行为时间 (ms)，第二行为声压 (Pa)。
    sample_rate : float
        信号采样率 (Hz)。
    bpf : float
        叶片通过频率 (Hz)，必须精确已知。BPF 精度直接影响 SCD 中
        循环频率 α 的准确性。

    Attributes
    ----------
    x : np.ndarray
        提取的声压序列（一维）。
    t : np.ndarray
        提取的时间序列（一维，单位 ms）。
    fs : float
        采样率。
    bpf : float
        叶片通过频率。
    f0 : float
        BPF 别名，与 bpf 相同。
    N0 : int
        每个 BPF 周期的采样点数 = round(fs/bpf)。
    freq : np.ndarray
        单边频率轴 (Hz)，由 rfftfreq 生成。

    Notes
    -----
    - 该类在初始化时预计算了所有数据段的 FFT（存储在 _seg_fft 中），
      后续的 SCD/相干度/指标计算均基于这些预计算结果，避免重复 FFT。
    - SCD 和 coherence 结果会被缓存（_scd 和 _coherence），后续
      调用会复用。
    """

    def __init__(self, time_signal: np.ndarray, sample_rate: float, bpf: float):
        """初始化循环谱分析器并预计算分段 FFT.

        Parameters
        ----------
        time_signal : np.ndarray
            (2, N) 数组，第一行为时间 (ms)，第二行为声压 (Pa)。
        sample_rate : float
            采样率 (Hz)。
        bpf : float
            叶片通过频率 (Hz)，精确已知。
        """
        # 提取声压序列和时间序列
        self.x = time_signal[1]  # 声压序列 (Pa)
        self.t = time_signal[0]  # 时间序列 (ms)
        self.fs = sample_rate    # 采样率 (Hz)
        self.bpf = bpf           # 叶片通过频率 (Hz)
        self.f0 = bpf            # BPF 别名

        # N0: 每个 BPF 周期内的采样点数（四舍五入确保整数）
        self.N0 = int(round(sample_rate / bpf))

        # ---- 预计算分段 FFT ----
        # 分段数 M = 总采样点数 / 每段点数（向下取整）
        self._N_seg = len(self.x) // self.N0

        # FFT 长度 = 2 * N0（2x 补零，原因见模块 docstring）
        self._N_fft = 2 * self.N0

        # 频率分辨率：Δf = fs / N_fft = fs / (2*N0) = BPF / 2
        self._df = self.fs / self._N_fft

        # 逐段计算 FFT，存储在列表中
        self._seg_fft = []
        for m in range(self._N_seg):
            # 取出第 m 段数据（长度为 N0）
            seg = self.x[m * self.N0:(m + 1) * self.N0]
            # 2x 补零 FFT（n=N_fft 自动补零到 2*N0）
            self._seg_fft.append(np.fft.fft(seg, n=self._N_fft))
        # 转换为 (M, N_fft) 的二维数组，便于向量化操作
        self._seg_fft = np.array(self._seg_fft)  # (N_seg, N_fft)

        # 单边频率轴（非负频率部分），用于 SCD 和 PSD 输出
        self._freq = np.fft.rfftfreq(self._N_fft, d=1.0 / self.fs)

        # 缓存变量：延迟计算，首次调用时填充
        self._scd: Optional[Dict[int, np.ndarray]] = None
        self._coherence: Optional[np.ndarray] = None

    # ---- 公共方法 ----

    def compute_scd(self, max_harmonic_order: int = 30) -> Dict:
        """计算循环谱密度 (SCD)，仅计算 α = k·BPF 的离散循环频率.

        FAM (FFT Accumulation Method) 实现细节
        ---------------------------------------
        对于每个循环频率 α = k·BPF：
        1. 计算频移量 k_half = round(α/2 / df)，即 α/2 对应的 FFT bin 索引。
        2. 遍历所有 M 个数据段的 FFT 结果 X_m(f)：
           S_x^α(f) += (1/M) * X_m(f + α/2) * X_m*(f - α/2)
        3. 这里使用实信号对称性处理负频率索引：
           当 f - α/2 < 0 时，X(-f) = X*(f)（实信号的 Hermitian 对称性）。
           由于 SCD 的对称性质，f - α/2 < 0 时的贡献等价于取 conj。

        Parameters
        ----------
        max_harmonic_order : int, optional
            最高谐频阶次 K_max，默认 30。循环频率 α 取 0 和
            k*BPF (k=1,2,...,K_max)。

        Returns
        -------
        scd : Dict[int, np.ndarray]
            字典，键为谐频阶次 k，值为对应的一维复数 SCD 频谱：
            - scd[0]: PSD S_x^0(f)，即经典功率谱密度（实数）
            - scd[k]: S_x^{k·BPF}(f)，循环谱密度，k >= 1（复数）

        Notes
        -----
        - α=0 时的 SCD 退化为标准 PSD：S_x^0(f) = E[|X(f)|²]。
        - 结果缓存至 self._scd，重复调用会触发重新计算。
        """
        # 构建循环频率列表：α=0 (PSD) + k*BPF (k=1...K_max)
        alpha_list = [0] + [
            k * self.bpf for k in range(1, max_harmonic_order + 1)
        ]
        scd = {}

        for alpha in alpha_list:
            # α/2 对应的 FFT bin 索引（四舍五入取整）
            # 补零 2x 后 Δf = BPF/2，所以 k*BPF/2 / (BPF/2) = k，恒为整数
            k_half = int(round(alpha / 2 / self._df))

            # 单边频率的 bin 数量（含 0 和 Nyquist）
            N_bins = self._N_fft // 2 + 1

            # 初始化当前 α 的 SCD 累加器
            scd_alpha = np.zeros(N_bins, dtype=complex)

            # ---- FAM 累加：遍历所有数据段 ----
            for X in self._seg_fft:
                # X 是长度为 N_fft 的复数 FFT 结果
                # 对每个非负频率 bin i，计算:
                #   X(f_i + α/2) · X*(f_i - α/2)
                for i in range(N_bins):
                    # 正频偏索引: f + α/2 在 FFT 数组中的位置
                    i_plus = i + k_half
                    # 负频偏索引的绝对值: |f - α/2|
                    i_minus = abs(i - k_half)

                    if i_plus < N_bins and i_minus < N_bins:
                        # 处理 f - α/2 < 0 的情况：
                        #   实信号的 FFT 满足 X[-idx] = conj(X[idx])
                        #   因此 X*(f - α/2) 对于负索引 = X*(conj(X[|idx|]))
                        #   = X[|idx|], 即不需要额外的共轭翻转
                        # 实际上两种情况都在频域直接使用正索引的共轭：
                        #   X(f+α/2) 始终用 i_plus 索引
                        #   X*(f-α/2) 始终用 |i-k_half| 索引取共轭
                        scd_alpha[i] += X[i_plus] * np.conj(X[i_minus])

            # 除以段数 M，得到 SCD 的无偏估计
            scd_alpha /= self._N_seg

            # 确定谐频阶次（α=0 时为 0，否则为 α/BPF）
            order = int(round(alpha / self.bpf)) if alpha > 0 else 0
            scd[order] = scd_alpha

        # 缓存结果
        self._scd = scd
        return scd

    def compute_cyclic_coherence(self) -> np.ndarray:
        """计算循环相干度 (Cyclic Coherence).

        循环相干度定义为：
            γ_x^α(f) = |S_x^α(f)|² / (S_x^0(f+α/2) · S_x^0(f-α/2))

        直观含义
        --------
        γ_x^α(f) ∈ [0, 1]，衡量频率 f 处的信号能量中有多大比例
        是由周期为 1/α 的确定性周期过程贡献的：
        - γ = 1: 信号在频率 f 处完全是周期性的（纯谐频）
        - γ = 0: 信号在频率 f 处完全随机（纯宽频）
        - 中间值表示两种成分的混合

        Returns
        -------
        coherence : np.ndarray
            形状为 (K_max+1, N_freq) 的二维数组：
            - coherence[0, :]: α=0 处的相干度，恒为 1.0
            - coherence[k, :]: k >= 1，在频率 f 处的循环相干度 γ_x^{k·BPF}(f)

        Notes
        -----
        - α=0 时，分子 |S_x^0(f)|² = S_x^0(f)²，分母 S_x^0(f)·S_x^0(f) = S_x^0(f)²，
          因此 γ_x^0(f) ≡ 1.0，没有区分能力。
        - 相干度被裁剪到 [0, 1] 范围（数值误差可能导致略微超界）。
        - 分母小于 1e-20 时，相干度置为 0（避免除零）。
        - 要求先计算 SCD；若未计算则自动调用 compute_scd()。
        """
        # 如果 SCD 尚未计算，先计算
        if self._scd is None:
            self.compute_scd()

        scd = self._scd
        # PSD 幅值（α=0 的 SCD 取模）
        S0 = np.abs(scd[0])  # PSD 幅值
        N_freq = len(self._freq)

        # 获取所有非零的谐频阶次
        orders = sorted(k for k in scd.keys() if k > 0)
        K_max = max(orders)

        # 初始化相干度数组
        coherence = np.zeros((K_max + 1, N_freq))
        coherence[0, :] = 1.0  # α=0 处相干度恒为 1

        # 对每个非零循环频率计算相干度
        for k in orders:
            alpha = k * self.bpf
            # α/2 对应的 bin 偏移量
            k_half = int(round(alpha / 2 / self._df))
            gamma = np.zeros(N_freq)

            for i in range(N_freq):
                i_plus = i + k_half
                i_minus = abs(i - k_half)

                if i_plus < N_freq and i_minus < N_freq:
                    # 分母: S0(f+α/2) * S0(f-α/2)
                    denom = S0[i_plus] * S0[i_minus]

                    if denom > 1e-20:
                        # 分子: |SCD|²
                        gamma[i] = np.abs(scd[k][i]) ** 2 / denom

            # 裁剪到 [0, 1]（数值误差可能导致微小越界）
            gamma = np.clip(gamma, 0.0, 1.0)
            coherence[k, :] = gamma

        # 缓存结果
        self._coherence = coherence
        return coherence

    def compute_integrated_cyclic_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算积分循环谱: I(α) = ∫ |S_x^α(f)| df.

        积分循环谱将 SCD 沿频率轴积分，得到每个循环频率 α 对应的
        总循环能量。I(α) 的大小反映了周期性（循环平稳性）在各个
        循环频率上的强弱分布。

        Returns
        -------
        alphas : np.ndarray
            循环频率值 (Hz)，包括 α=0。
        integrated : np.ndarray
            对应的积分循环能量值，长度与 alphas 相同。

        Notes
        -----
        - 积分使用梯形法则 (np.trapezoid)。
        - 如果 BPF 处出现明显的峰值，说明信号在该周期上具有强循环平稳性。
        """
        if self._scd is None:
            self.compute_scd()

        orders = sorted(self._scd.keys())
        alphas = np.array([k * self.bpf for k in orders])
        # 对每个 α 的 SCD 幅值沿频率积分
        integrated = np.array([
            np.trapezoid(np.abs(self._scd[k]), self._freq) for k in orders
        ])
        return alphas, integrated

    def reconstruct_steady_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """从循环相干度重建定常分量的连续 PSD.

        重建策略
        --------
        S_steady(f) = max_{α≠0} γ_x^α(f) · S_x^0(f)

        直觉：对于频率 f 处的每个频率 bin，取所有非零循环频率上
        最大的循环相干度，将其乘以该频率处的总 PSD，得到定常（周期）
        分量的 PSD。剩余的为非定常（随机）分量的 PSD：
            S_unsteady(f) = S_x^0(f) - S_steady(f)

        这种策略基于以下推理：
        - γ_x^α(f) 代表了周期为 1/α 的过程在频率 f 处贡献的功率比例。
        - 取 max 可以捕获最强周期贡献，无论其具体来自哪次谐频。
        - 不同阶次的循环频率可能在不同频段有不同贡献，取 max 避免了
          低估（如果取平均值）。

        Returns
        -------
        steady_psd : np.ndarray
            定常分量的功率谱密度 (Pa²)，一维实数数组。
        unsteady_psd : np.ndarray
            非定常分量的功率谱密度 (Pa²)，一维实数数组，
            被裁剪为非负值。

        Notes
        -----
        - 要求先计算 SCD 和循环相干度；若未计算则自动计算。
        - unsteady_psd 被 clip 到 [0, +inf)，因为数值误差可能导致
          略微的负值。
        - 这种重建方法对低 SNR 场景可能高估定常分量（因相干度估计偏差）。
        """
        # 确保前置计算完成
        if self._coherence is None:
            self.compute_cyclic_coherence()

        # 总 PSD
        S0 = np.abs(self._scd[0])

        # 对每个频率 bin，取所有 k >= 1 中最大的循环相干度
        # self._coherence 形状 (K_max+1, N_freq)，索引 1: 对应 k>=1
        max_gamma = np.max(self._coherence[1:, :], axis=0)

        # 定常 PSD = 最大相干度 × 总 PSD
        steady_psd = max_gamma * S0

        # 非定常 PSD = 总 PSD - 定常 PSD
        unsteady_psd = S0 - steady_psd

        # 裁剪避免数值误差导致的负值
        unsteady_psd = np.clip(unsteady_psd, 0, None)

        return steady_psd, unsteady_psd

    def compute_metrics(
        self,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict:
        """计算所有定量指标，提供完整的定常/非定常贡献评估.

        计算的指标包括:
        - 全局能量: 总能量、定常能量、非定常能量
        - 全局比例: 定常比、非定常比
        - 全局 SPL: 定常 SPL、非定常 SPL
        - 分频段统计: low (0-250 Hz), mid (250-2000 Hz), high (>2000 Hz)
        - 各阶谐频处的循环相干度和 SCD 幅值
        - 积分循环谱的峰值位置

        Parameters
        ----------
        freq_bands : Dict[str, Tuple[float, float]], optional
            自定义频段字典，格式如 {'low': (0, 250), 'mid': (250, 2000)}。
            默认使用标准三频段: low/mid/high。

        Returns
        -------
        metrics : Dict
            包含所有定量指标的字典。主要键包括:
            - 'total_energy', 'steady_energy', 'unsteady_energy'
            - 'steady_ratio', 'unsteady_ratio'
            - 'steady_spl', 'unsteady_spl' (dB)
            - '{band}_total_energy', '{band}_steady_energy', ...
            - 'harmonic_cyclic_coherence': 各阶谐频的详细相干度
            - 'integrated_cyclic_max_alpha': 积分循环谱峰值 α
        """
        # 确保所有前置计算完成
        if self._scd is None:
            self.compute_scd()
        if self._coherence is None:
            self.compute_cyclic_coherence()

        # ---- 全局能量 ----
        S0 = np.abs(self._scd[0])
        total_energy = np.trapezoid(S0, self._freq)

        steady_psd, unsteady_psd = self.reconstruct_steady_spectrum()
        steady_energy = np.trapezoid(steady_psd, self._freq)
        unsteady_energy = np.trapezoid(unsteady_psd, self._freq)

        # 基础指标
        metrics = {
            'total_energy': total_energy,
            'steady_energy': steady_energy,
            'unsteady_energy': unsteady_energy,
            'steady_ratio': (
                steady_energy / total_energy if total_energy > 0 else 0.0
            ),
            'unsteady_ratio': (
                unsteady_energy / total_energy if total_energy > 0 else 0.0
            ),
            # SPL 计算: SPL = 10*log10(energy / P_REF^2)
            'steady_spl': (
                10 * np.log10(steady_energy / (P_REF ** 2) + 1e-12)
                if steady_energy > 0 else -np.inf
            ),
            'unsteady_spl': (
                10 * np.log10(unsteady_energy / (P_REF ** 2) + 1e-12)
                if unsteady_energy > 0 else -np.inf
            ),
        }

        # ---- 分频段定常比 ----
        # 默认三频段: 低频 0-250 Hz, 中频 250-2000 Hz, 高频 >2000 Hz
        if freq_bands is None:
            freq_bands = {
                'low': (0, 250),
                'mid': (250, 2000),
                'high': (2000, np.inf),
            }

        for band_name, (fl, fh) in freq_bands.items():
            # 创建当前频段的频率掩码
            mask = (self._freq >= fl) & (self._freq < fh)

            if np.any(mask):
                # 频段内积分能量
                band_total = np.trapezoid(S0[mask], self._freq[mask])
                band_steady = np.trapezoid(steady_psd[mask], self._freq[mask])
                band_unsteady = np.trapezoid(unsteady_psd[mask], self._freq[mask])

                metrics[f'{band_name}_total_energy'] = band_total
                metrics[f'{band_name}_steady_energy'] = band_steady
                metrics[f'{band_name}_unsteady_energy'] = band_unsteady
                metrics[f'{band_name}_steady_ratio'] = (
                    band_steady / band_total if band_total > 0 else 0.0
                )
                metrics[f'{band_name}_unsteady_ratio'] = (
                    band_unsteady / band_total if band_total > 0 else 0.0
                )

        # ---- 各阶谐频处的定常比（循环相干度） ----
        harmonics = {}
        K_max = max(k for k in self._scd.keys() if k > 0)

        for k in range(1, K_max + 1):
            if k in self._scd:
                target_freq = k * self.bpf
                # 找到最接近目标频率的 bin 索引
                idx = np.argmin(np.abs(self._freq - target_freq))
                harmonics[k] = {
                    'nominal_freq': target_freq,
                    'actual_freq': self._freq[idx],
                    'cyclic_coherence': float(self._coherence[k, idx]),
                    'scd_magnitude': float(np.abs(self._scd[k][idx])),
                }
        metrics['harmonic_cyclic_coherence'] = harmonics

        # ---- 积分循环谱峰值位置 ----
        # 找到 I(α) 最大的 α（排除 α=0），揭示最强的周期分量
        alphas, integrated = self.compute_integrated_cyclic_spectrum()
        metrics['integrated_cyclic_max_alpha'] = (
            float(alphas[np.argmax(integrated[1:]) + 1])
            if len(alphas) > 1 else 0.0
        )

        return metrics

    # ---- 属性 ----

    @property
    def freq(self) -> np.ndarray:
        """单边频率轴 (Hz)."""
        return self._freq

    @property
    def scd_data(self) -> Optional[Dict[int, np.ndarray]]:
        """缓存的 SCD 数据，若未计算则为 None."""
        return self._scd

    @property
    def coherence_data(self) -> Optional[np.ndarray]:
        """缓存的循环相干度数据，若未计算则为 None."""
        return self._coherence

    # ---- 导出方法 ----

    def get_scd_3d_export(self) -> Dict[str, np.ndarray]:
        """导出 SCD 为 3D 可视化格式.

        返回的数据结构适合绘制 waterfall / surface 图，
        其中 x 轴为频率 f，y 轴为循环频率 α，z 轴为 |SCD|。

        Returns
        -------
        data : Dict[str, np.ndarray]
            包含以下键的字典:
            - 'f': 频率轴 (Hz), shape (N_freq,)
            - 'alpha': 循环频率轴 (Hz), shape (K_max+1,)
            - 'scd_magnitude': |SCD| 幅值, shape (K_max+1, N_freq)
        """
        if self._scd is None:
            self.compute_scd()

        orders = sorted(self._scd.keys())
        alpha_values = np.array([k * self.bpf for k in orders])
        scd_mag = np.array([np.abs(self._scd[k]) for k in orders])

        return {
            'f': self._freq,
            'alpha': alpha_values,
            'scd_magnitude': scd_mag,
        }
