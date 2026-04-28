"""
相位约束载荷分离模块 (Phase-Constrained Load Separation).

本模块实现基于厚度噪声相位的载荷噪声分离方法，将总载荷噪声分解为
定常载荷分量（与厚度噪声同相）和非定常载荷分量（与厚度噪声正交）。

核心算法
--------
利用厚度噪声作为相位参考，在每一谐频带内：
1. 提取厚度噪声的平均相位作为参考相位 ref_phase。
2. 计算载荷噪声与参考相位之间的相位差 phase_diff。
3. 将载荷幅值投影到同相方向得到定常分量：
   steady_amp = load_amp * cos(phase_diff)
4. 将载荷幅值投影到正交方向得到非定常分量：
   unsteady_amp = load_amp * sin(phase_diff)
5. 非谐频区域（宽频）全部归为非定常载荷。

重要警告 (同相假设)
-------------------
该方法的核心假设是：定常载荷产生的声压在观测者处与厚度噪声同相。
这一假设在物理上并不总是成立，原因包括：
- 载荷噪声的传播路径与厚度噪声不同（折射、散射效应）。
- 旋翼尾迹对载荷分布的调制可能引入额外相位差。
- 远场/近场差异可能导致相位关系变化。

**务必使用 CyclicSpectrumAnalyzer 进行交叉验证**，尤其是当：
- BPF 处非定常占比异常高（>50%）。
- 相位差方差显著（说明同相假设不成立）。
- 结论对物理参数敏感时。

参考文献
--------
该方法参考了旋翼噪声中间相/正交分解的思想，在频域将载荷投影到
厚度噪声相位方向以实现定常/非定常分离。
"""

import numpy as np
from typing import Dict, List, Tuple


class PhaseConstraintSeparator:
    """相位约束分离器 -- 以厚度噪声相位为基准，将载荷噪声投影分离为定常/非定常分量.

    在频域中对每一 BPF 谐频带，以厚度噪声的平均相位作为参考方向，
    将载荷噪声的复频谱投影到同相（定常）和正交（非定常）两个方向上。

    .. warning::
        **同相假设 (In-Phase Assumption)**: 本方法假定定常载荷声压
        与厚度噪声在观测者处同相。该假设在物理上不一定成立。强烈
        建议使用 `CyclicSpectrumAnalyzer` 进行交叉验证。若各阶谐频
        的相位差方差较大（> 0.5 rad²），说明同相假设可能不满足。

    Parameters
    ----------
    frequencies : np.ndarray
        频谱对应的频率轴 (Hz)，一维数组。

    Attributes
    ----------
    freqs : np.ndarray
        存储的频率轴。

    Notes
    -----
    算法限制:
    - 仅对谐频带内的载荷噪声进行投影分解，宽频区域全部归为非定常。
    - 投影基于"同相假设"，即定常载荷与厚度噪声相位一致。当此假设
      不成立时，定常/非定常的划分会产生系统偏差。
    - 该方法依赖厚度噪声的复频谱作为相位参考，因此要求厚度噪声的
      信噪比足够高（通常在 BPF 谐频处成立）。
    """

    def __init__(self, frequencies: np.ndarray):
        """初始化相位约束分离器.

        Parameters
        ----------
        frequencies : np.ndarray
            频谱对应的频率轴 (Hz)，一维数组。
        """
        self.freqs = frequencies

    def identify_harmonic_bands(
        self,
        fundamental_freq: float,
        max_harmonic_order: int = 30,
        bandwidth_ratio: float = 0.03
    ) -> List[np.ndarray]:
        """识别频谱中的谐频带索引.

        根据基频 (BPF) 和最高谐频阶次，在频率轴上找出各次谐频
        周围的频带索引。每个频带的宽度由 bandwidth_ratio 控制。

        Parameters
        ----------
        fundamental_freq : float
            基频 (Hz)，通常为叶片通过频率 (BPF)。
        max_harmonic_order : int, optional
            最高谐频阶次，默认 30。
        bandwidth_ratio : float, optional
            带宽与谐频频率的比值，默认 0.03（3% 相对带宽）。

        Returns
        -------
        harmonic_indices_list : List[np.ndarray]
            列表，每个元素是一个整数数组，包含对应阶次谐频带内
            的频率索引（在 self.freqs 中的位置）。
        """
        harmonic_indices_list = []

        for order in range(1, max_harmonic_order + 1):
            # 计算第 order 次谐频的精确频率
            hf = fundamental_freq * order

            # 以 hf 为中心，建立带宽为 hf * bandwidth_ratio 的频带
            bw = hf * bandwidth_ratio

            # 查找落在 [hf - bw/2, hf + bw/2] 内的所有频率索引
            indices = np.where(
                (self.freqs >= hf - bw / 2) & (self.freqs <= hf + bw / 2)
            )[0]

            if len(indices) > 0:
                harmonic_indices_list.append(indices)

        return harmonic_indices_list

    def separate(
        self,
        load_complex: np.ndarray,
        thickness_complex: np.ndarray,
        fundamental_freq: float,
        max_harmonic_order: int = 30,
        bandwidth_ratio: float = 0.03,
        check_phase_consistency: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """分离定常和非定常载荷噪声.

        算法步骤
        --------
        1. 识别谐频带 (调用 identify_harmonic_bands)。
        2. 对每个谐频带：
           a. 取厚度噪声的加权平均相位作为参考相位 ref_phase。
           b. 计算载荷噪声各频率点与 ref_phase 的相位差 phase_diff。
           c. 同相投影：steady_amp = load_amp * cos(phase_diff)。
           d. 正交投影：unsteady_amp = load_amp * sin(phase_diff)。
           e. 重建复频谱：稳态分量相位 = ref_phase，非稳态分量
              相位 = ref_phase + pi/2（正交）。
        3. 将非谐频区域（宽频）的载荷全部归入非定常分量。
        4. 能量守恒验证（误差阈值 0.1%）。
        5. 可选的相位一致性统计。

        Parameters
        ----------
        load_complex : np.ndarray
            载荷噪声的复频谱（FFT 输出），一维复数数组。
        thickness_complex : np.ndarray
            厚度噪声的复频谱（FFT 输出），一维复数数组，用作相位参考。
        fundamental_freq : float
            基频 (Hz)，通常为 BPF。
        max_harmonic_order : int, optional
            最高谐频阶次，默认 30。
        bandwidth_ratio : float, optional
            谐频带宽比，默认 0.03。
        check_phase_consistency : bool, optional
            是否计算相位一致性统计量。若为 True，会在返回的
            phase_stats 中包含相位差均值、方差等信息。默认 False。

        Returns
        -------
        steady_load_amp : np.ndarray
            定常载荷分量的幅值频谱（Pa），一维实数数组。
        unsteady_load_amp : np.ndarray
            非定常载荷分量的幅值频谱（Pa），一维实数数组。
        phase_stats : Dict
            相位统计字典，包含以下键（仅当 check_phase_consistency=True
            时填充完整，否则仅包含空列表）：
            - 'phase_diffs': 所有谐频点的相位差列表 (rad)
            - 'phase_diff_variances': 各阶谐频的相位差方差列表
            - 'mean_phase_diff': 全局相位差均值 (rad)
            - 'overall_phase_diff_variance': 全局相位差方差 (rad²)
            - 'max_phase_diff_variance': 各阶最大相位差方差 (rad²)
            - 以及对应的角度制 (deg) 版本

        Notes
        -----
        投影公式的几何解释:
        - 将载荷复矢量 Load 投影到厚度噪声相位方向，
          同相分量 = |Load| * cos(Δφ)，正交分量 = |Load| * sin(Δφ)。
        - 能量关系：|Load|² = |steady|² + |unsteady|²（在谐频带内严格成立，
          因为 cos² + sin² = 1）。
        - 宽频区域由于没有明确的厚度噪声相位参考，全部归入非定常分量。
        """
        # Step 1: 识别所有谐频带的频率索引
        harmonic_indices_list = self.identify_harmonic_bands(
            fundamental_freq, max_harmonic_order, bandwidth_ratio
        )

        # 初始化：定常和非定常分量的复频谱全为零
        steady_complex = np.zeros_like(load_complex)
        unsteady_complex = np.zeros_like(load_complex)

        # 相位统计字典
        phase_stats: Dict = {'phase_diffs': [], 'phase_diff_variances': []}

        # Step 2: 对每个谐频带执行相位投影
        if harmonic_indices_list:
            for indices in harmonic_indices_list:
                # 取当前谐频带内的厚度噪声和载荷噪声复频谱切片
                thick_band = thickness_complex[indices]
                load_band = load_complex[indices]

                # ---- 相位参考提取 ----
                # 计算厚度噪声带内的加权平均相位（unwrap 后取均值，
                # 避免跨越 -π/+π 边界时的相位跳变问题）
                ref_phase = np.mean(np.unwrap(np.angle(thick_band)))

                # 计算载荷噪声各频率点的相位（unwrap 保证连续性）
                load_phases = np.unwrap(np.angle(load_band))

                # 相位差：归一化到 [-π, π) 范围
                # 公式: ((load_phases - ref_phase + π) mod 2π) - π
                phase_diffs = (
                    np.mod(load_phases - ref_phase + np.pi, 2 * np.pi) - np.pi
                )

                # ---- 可选：相位一致性统计 ----
                if check_phase_consistency:
                    phase_stats['phase_diffs'].extend(phase_diffs.tolist())
                    phase_stats['phase_diff_variances'].append(
                        np.var(phase_diffs)
                    )

                # ---- 核心投影：将载荷幅值分解为同相和正交分量 ----
                #   Load_vector = |Load| * e^{i*φ_load}
                #   ref_direction = e^{i*φ_ref}
                #   同相投影: |Load| * cos(φ_load - φ_ref) = |Load| * cos(Δφ)
                #   正交投影: |Load| * sin(φ_load - φ_ref) = |Load| * sin(Δφ)
                load_amps = np.abs(load_band)
                steady_amps = load_amps * np.cos(phase_diffs)
                unsteady_amps = load_amps * np.sin(phase_diffs)

                # ---- 重建复频谱 ----
                # 定常分量的相位 = 参考相位（与厚度同相）
                steady_complex[indices] = steady_amps * np.exp(1j * ref_phase)
                # 非定常分量的相位 = 参考相位 + π/2（正交方向）
                unsteady_complex[indices] = (
                    unsteady_amps * np.exp(1j * (ref_phase + np.pi / 2))
                )

        # Step 3: 宽频区域处理 -- 非谐频区域全部归为非定常载荷
        # 构建所有谐频索引的并集
        all_harmonic = (
            np.concatenate(harmonic_indices_list)
            if harmonic_indices_list else np.array([], dtype=int)
        )
        # 创建布尔掩码：True 表示宽频（非谐频）位置
        broadband_mask = np.ones(len(self.freqs), dtype=bool)
        broadband_mask[all_harmonic] = False
        # 宽频区域的载荷直接复制到非定常分量
        unsteady_complex[broadband_mask] = load_complex[broadband_mask]

        # Step 4: 能量守恒验证
        # 使用 |amplitude|^2 作为能量度量，验证分解不损失能量
        original_energy = np.sum(np.abs(load_complex) ** 2)
        separated_energy = (
            np.sum(np.abs(steady_complex) ** 2) +
            np.sum(np.abs(unsteady_complex) ** 2)
        )
        energy_error = (
            abs(separated_energy - original_energy) / original_energy
            if original_energy > 0 else 0
        )
        # 误差阈值 0.1%（比频域分离更严格，因为投影应保持能量守恒）
        if energy_error > 0.001:
            print(
                f"Warning: Energy conservation error {energy_error:.2%} "
                f"in phase constraint separation"
            )

        # Step 5: 汇总相位一致性统计量（仅在启用时）
        if check_phase_consistency and phase_stats['phase_diffs']:
            phase_diffs_array = np.array(phase_stats['phase_diffs'])
            phase_stats['mean_phase_diff'] = np.mean(phase_diffs_array)
            phase_stats['overall_phase_diff_variance'] = np.var(
                phase_diffs_array
            )
            phase_stats['max_phase_diff_variance'] = (
                max(phase_stats['phase_diff_variances'])
                if phase_stats['phase_diff_variances'] else 0.0
            )
            # 同时输出角度制 (deg) 版本，便于人工判读
            phase_stats['mean_phase_diff_deg'] = np.rad2deg(
                phase_stats['mean_phase_diff']
            )
            phase_stats['overall_phase_diff_variance_deg'] = np.rad2deg(
                phase_stats['overall_phase_diff_variance']
            )
            phase_stats['max_phase_diff_variance_deg'] = np.rad2deg(
                phase_stats['max_phase_diff_variance']
            )

        # 返回定常和非定常分量的幅值谱（取模），以及相位统计
        return np.abs(steady_complex), np.abs(unsteady_complex), phase_stats
