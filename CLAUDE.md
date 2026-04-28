# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Rotor noise analysis toolkit for Farassat 1A FW-H numerical simulation data. Quantifies noise source contributions (thickness, steady load, unsteady load) across harmonic and broadband frequency components. Supports free-field (FF) and free-field + surface-reflection (FF+SR) scenarios, typically 12 observer points per case.

## Code style

**Strictly follow PEP standards** (PEP 8 for layout, PEP 257 for docstrings, type annotations where practical).

**Language split**:
- **Chinese (中文)**: all comments and docstrings
- **English**: all code — identifiers, variable names, function names, class names, log/print messages

Rationale: the project owner is a native Chinese speaker working in a Chinese academic context. Chinese comments improve readability and maintenance speed. English code ensures compatibility with the Python ecosystem (no non-ASCII identifiers).

## Shell & Python environment

All commands use **Unix-style paths** (forward slashes). The venv is at `.venv/` — always use `.venv/Scripts/python.exe` (not bare `python`).

## Common commands

```bash
# Run any pipeline directly (edit hardcoded params in if __name__ == "__main__" block first)
.venv/Scripts/python.exe src/pipelines/cyclic_spectrum_ff.py

# CLI entry point (convenience layer over pipeline scripts)
.venv/Scripts/python.exe main.py source Case04 Case04_Rotor --has-reflection --bpf 46.97 --max-harmonic-order 50
.venv/Scripts/python.exe main.py cyclic Case01 Case01_Rotor --bpf 46.97 --output spectrum,summary

# CLI subcommands: peak | band | source | cyclic | full
.venv/Scripts/python.exe main.py --help
```

## Architecture

### Package layering (top-down)

```
main.py                          # CLI dispatcher (argparse) — thin layer, delegates to pipelines/
  └── pipelines/                 # Orchestration: read CSVs → call library → write output CSVs
        └── spectral/            # Spectrum analysis: peak detection, octave-band energy
        └── decomposition/       # Noise decomposition: harmonic/broadband, steady/unsteady, cyclic spectrum
              └── signal_utils   # Leaf utilities: SPL(), rfft()
```

### Two analysis approaches for steady/unsteady load separation

1. **Phase-constraint method** (`decomposition/phase_constraint.py` — `PhaseConstraintSeparator`):
   Projects load noise onto thickness-noise phase axis. In-phase = steady, quadrature = unsteady.
   **Limitation**: assumes steady load is in-phase with thickness noise at the observer — this is often physically false. When `phase_diff ≈ ±90°`, the method systematically misclassifies periodic energy as "unsteady".
   pipeline: `decomposition_ff.py` / `decomposition_ffsr.py`

2. **Cyclic spectrum method** (`decomposition/cyclic_spectrum.py` — `CyclicSpectrumAnalyzer`):
   FAM-based cyclostationary analysis. Separates periodic (cyclostationary, α ≠ 0) from random (stationary, α = 0 only) components without needing a reference signal or phase assumption. Physically more reliable.
   pipeline: `cyclic_spectrum_ff.py` / `cyclic_spectrum_ffsr.py`

### Pipeline script pattern

Every pipeline script follows the same dual-use pattern:
- Exposes a `run_*()` function taking explicit parameters (called by `main.py` or direct import)
- Has an `if __name__ == "__main__"` block where the user **hardcodes** parameters directly
- Contains `sys.path.insert(0, ...)` at the top so it can be run from any working directory

### Data flow

```
Preprocessing:           Time-domain CSV  →  FreqDomain CSV + SPLs CSV
                          (ff_signal / merge_signal equivalent in pipelines/)

Spectral analysis:       FreqDomain CSV   →  Harmonics CSV, BandContribution CSV
                          (spectral_ff.py / spectral_ffsr.py)

Source decomposition:    FreqDomain CSV   →  SourceContribution_Detail/Band/Harmonic/Summary CSVs
 (phase constraint)      + Time-domain CSVs  (decomposition_ff.py / decomposition_ffsr.py)

Cyclic spectrum:         Time-domain CSV  →  SCD_3D.npz, CyclicCoherence CSV,
                          (Load noise)        IntegratedCyclicSpectrum CSV, SteadySpectrum CSV,
                                              CyclicSummary CSV
                                              (cyclic_spectrum_ff.py / cyclic_spectrum_ffsr.py)
```

### Key data file naming

- `{prefix}_FF.csv` — free-field time-domain (Time, Thickness, Load, Total)
- `{prefix}_SR.csv` — surface-reflection time-domain (same columns)
- `{prefix}_FreqDomain.csv` — frequency-domain amplitudes/SPLs
- `{prefix}_FF/SR/merged` prefix in column names distinguishes signal types

## Output control for cyclic spectrum

The `output` parameter in cyclic pipeline scripts controls which files are written:

| Value | File | When to use |
|-------|------|-------------|
| `scd` | `*_SCD_3D.npz` | For 3D waterfall/surface plots (large file) |
| `coherence` | `*_CyclicCoherence.csv` | For colormap heatmap of γ(f,α) |
| `ics` | `*_IntegratedCyclicSpectrum.csv` | Check α-peak positions |
| `spectrum` | `*_SteadySpectrum.csv` | Full continuous steady/unsteady PSD for plotting |
| `summary` | `*_CyclicSummary.csv` | Just the numbers (steady_ratio, band ratios) |
| `all` | All above | |
| default | `ics, spectrum, summary` | Balanced: key metrics + plottable spectrum |
