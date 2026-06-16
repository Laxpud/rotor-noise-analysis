"""基于真实 Case05 数据验证 merged OASPL Tecplot 导出。"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.preprocess_ffsr import write_merged_oaspl_dat  # noqa: E402
from signal_utils import SPL  # noqa: E402


class MergedOasplDatTest(unittest.TestCase):
    """检查 merged 时域总声压能否汇总成与现有 SPL_*.dat 一致的空间文件。"""

    def test_writes_case05_merged_oaspl_dat_from_real_data(self):
        data_dir = ROOT / "data" / "Case05"
        prefixes = [
            f"Case05_Rotor_OBS{obs_number:04d}"
            for obs_number in range(1, 13)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "Case05_SPL_merged.dat"
            write_merged_oaspl_dat(data_dir, prefixes, output_path=output_path)

            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines[0], 'title="plot"')
        self.assertEqual(lines[1], 'variables="X","Y","Z","SPL(dB)","IOBS"')
        self.assertEqual(lines[2], "zone,i=12,datapacking=point")
        self.assertEqual(len(lines), 15)

        first_data = lines[3].split()
        self.assertEqual(first_data[:3], ["0.00000", "4.50000", "0.00000"])
        self.assertEqual(first_data[4], "1")

        merged = pd.read_csv(data_dir / "Case05_Rotor_OBS0001_merged.csv")
        expected_spl = SPL(np.vstack([merged["Time"].values, merged["Total"].values]))
        self.assertAlmostEqual(float(first_data[3]), expected_spl, places=5)


if __name__ == "__main__":
    unittest.main()
