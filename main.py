"""
旋翼气动噪声分析工具主入口
支持多种分析功能：
  peak   - 峰值频率与谐频分析
  band   - 频带能量贡献分析
  source - 源项频域贡献量化分析（相位约束法）
  cyclic - 循环平稳谱分析（定常/非定常定量评估）
  full   - 完整分析流程（peak + band + source）
"""
import argparse
import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def run_peak_analysis(file_path: str, filename_prefix: List[str],
                      has_reflection: bool = False):
    if has_reflection:
        from pipelines.spectral_ffsr import run_peak_analysis as _run
    else:
        from pipelines.spectral_ff import run_peak_analysis as _run
    print(f"Running peak frequency analysis on {file_path}...")
    _run(file_path, filename_prefix)
    print("Peak analysis completed!")


def run_band_analysis(file_path: str, filename_prefix: List[str],
                      group_prefixes: List[str] = None, has_reflection: bool = False):
    if has_reflection:
        from pipelines.spectral_ffsr import run_band_analysis as _run
    else:
        from pipelines.spectral_ff import run_band_analysis as _run
    print(f"Running band contribution analysis on {file_path}...")
    _run(file_path, filename_prefix, group_prefixes)
    print("Band analysis completed!")


def run_source_analysis(
    file_path: str, filename_prefix: List[str],
    group_prefixes: List[str] = None, has_reflection: bool = False,
    fundamental_freq: float = None, harmonic_bandwidth_ratio: float = 0.03,
    max_harmonic_order: int = 30, band_type: str = 'octave',
    band_fraction: int = 3, band_f_low: float = 10, band_f_high: float = 20000,
    check_phase_consistency: bool = False
):
    if has_reflection:
        from pipelines.decomposition_ffsr import run_decomposition_analysis as _run
    else:
        from pipelines.decomposition_ff import run_decomposition_analysis as _run

    print(f"Running source contribution analysis on {file_path}...")
    if fundamental_freq:
        print(f"Using specified fundamental frequency: {fundamental_freq:.2f} Hz")
    _run(file_path=file_path, filename_prefix=filename_prefix,
         group_prefixes=group_prefixes, fundamental_freq=fundamental_freq,
         harmonic_bandwidth_ratio=harmonic_bandwidth_ratio,
         max_harmonic_order=max_harmonic_order,
         band_type=band_type, band_fraction=band_fraction,
         band_f_low=band_f_low, band_f_high=band_f_high,
         check_phase_consistency=check_phase_consistency)
    print("Source contribution analysis completed!")


def run_cyclic_analysis(
    file_path: str, filename_prefix: List[str],
    group_prefixes: List[str] = None, has_reflection: bool = False,
    bpf: float = None, max_harmonic_order: int = 30,
    output: str = 'ics,spectrum,summary'
):
    if has_reflection:
        from pipelines.cyclic_spectrum_ffsr import run_cyclic_analysis as _run
    else:
        from pipelines.cyclic_spectrum_ff import run_cyclic_analysis as _run

    print(f"Running cyclic spectrum analysis on {file_path}...")
    if bpf:
        print(f"Using specified BPF: {bpf:.2f} Hz")
    output_list = [s.strip() for s in output.split(',')]
    _run(file_path=file_path, filename_prefix=filename_prefix,
         group_prefixes=group_prefixes, bpf=bpf,
         max_harmonic_order=max_harmonic_order, output=output_list)
    print("Cyclic spectrum analysis completed!")


def run_full_analysis(file_path: str, filename_prefix: List[str],
                      group_prefixes: List[str] = None, has_reflection: bool = False,
                      **kwargs):
    print("Running full analysis pipeline...")
    run_peak_analysis(file_path, filename_prefix, has_reflection)
    run_band_analysis(file_path, filename_prefix, group_prefixes, has_reflection)
    run_source_analysis(file_path, filename_prefix, group_prefixes, has_reflection, **kwargs)
    print("Full analysis pipeline completed!")


def main():
    parser = argparse.ArgumentParser(description="旋翼气动噪声分析工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    def add_common(subp):
        subp.add_argument('file_path', help='数据文件所在目录路径')
        subp.add_argument('filename_prefix', nargs='+',
                          help='要处理的文件名前缀（不包含_FreqDomain.csv后缀）')
        subp.add_argument('--group-prefixes', nargs='+',
                          help='分组保存汇总数据的前缀列表')
        subp.add_argument('--has-reflection', action='store_true',
                          help='数据是否包含表面反射分量（FF/SR/merged）')

    def add_source_args(subp):
        subp.add_argument('--fundamental-freq', type=float, help='基频/BPF (Hz)')
        subp.add_argument('--harmonic-bandwidth-ratio', type=float, default=0.03)
        subp.add_argument('--max-harmonic-order', type=int, default=30)
        subp.add_argument('--band-type', choices=['octave', 'custom'], default='octave')
        subp.add_argument('--band-fraction', type=int, default=3)
        subp.add_argument('--band-f-low', type=float, default=10)
        subp.add_argument('--band-f-high', type=float, default=20000)

    # 峰值频率与谐频分析
    p = subparsers.add_parser('peak', help='峰值频率与谐频分析')
    add_common(p)

    # 频带能量贡献分析
    p = subparsers.add_parser('band', help='频带能量贡献分析')
    add_common(p)

    # 源项频域贡献量化分析（相位约束法）
    p = subparsers.add_parser('source', help='源项频域贡献量化分析（相位约束法）')
    add_common(p)
    add_source_args(p)
    p.add_argument('--check-phase-consistency', action='store_true',
                   help='检查相位一致性')

    # 循环平稳谱分析（定常/非定常定量评估）
    p = subparsers.add_parser('cyclic', help='循环平稳谱分析（定常/非定常定量评估）')
    add_common(p)
    p.add_argument('--bpf', type=float, help='叶片通过频率 (Hz)')
    p.add_argument('--max-harmonic-order', type=int, default=30,
                   help='最大谐频阶数 (默认30)')
    p.add_argument('--output', type=str, default='ics,spectrum,summary',
                   help='输出文件选择: scd,coherence,ics,spectrum,summary,all '
                        '(逗号分隔, 默认 ics,spectrum,summary)')

    # 完整分析流程（peak+band+source）
    p = subparsers.add_parser('full', help='完整分析流程（peak+band+source）')
    add_common(p)
    add_source_args(p)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'peak':
        run_peak_analysis(args.file_path, args.filename_prefix, args.has_reflection)
    elif args.command == 'band':
        run_band_analysis(args.file_path, args.filename_prefix,
                          args.group_prefixes, args.has_reflection)
    elif args.command == 'source':
        run_source_analysis(
            args.file_path, args.filename_prefix, args.group_prefixes,
            args.has_reflection, args.fundamental_freq,
            args.harmonic_bandwidth_ratio, args.max_harmonic_order,
            args.band_type, args.band_fraction, args.band_f_low, args.band_f_high,
            args.check_phase_consistency)
    elif args.command == 'cyclic':
        run_cyclic_analysis(
            args.file_path, args.filename_prefix, args.group_prefixes,
            args.has_reflection, args.bpf, args.max_harmonic_order, args.output)
    elif args.command == 'full':
        run_full_analysis(
            args.file_path, args.filename_prefix, args.group_prefixes,
            args.has_reflection,
            fundamental_freq=args.fundamental_freq,
            harmonic_bandwidth_ratio=args.harmonic_bandwidth_ratio,
            max_harmonic_order=args.max_harmonic_order,
            band_type=args.band_type, band_fraction=args.band_fraction,
            band_f_low=args.band_f_low, band_f_high=args.band_f_high)


if __name__ == "__main__":
    main()
