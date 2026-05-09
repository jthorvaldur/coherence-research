# coherence-research

Delta.72 coherence framework — applied mathematics for instability detection.

## Framework

**Δ = (P · A · R) / (D + N)**

Where P = Pattern Retention, A = Phase Alignment, R = Recovery, D = Drift, N = Noise Amplification.

Extended operators: M (Memory-of-Attractor), W (Windowed Recovery), F (Flower Return), B (Bounded Distortion), L (Loss-of-Flower).

## Experiments

| # | Domain | Dataset | Key Result |
|---|--------|---------|------------|
| 1-6 | Synthetic | GPU suite | 100% detection, 406-step mean lead |
| 7 | Energy | Office buildings | 348 coherence-only alerts |
| 8 | Mixed | NAB benchmark | Structural degradation detector |
| 8 | Aerospace | NASA C-MAPSS | 100% detection, 186 cycles lead |
| 9 | Industrial | SKAB valves | 97.1% detection, 34 experiments |
| 10 | Energy | UCI Household | 432 vs 1 variance alert |
| 11 | Biomedical | PhysioNet ECG | 100% recall, F1=0.386 |
| 12 | Financial | Credit Card | F1=0.557, 88.7% recall |
| 13 | Biomedical | EEG seizure | In progress (GPU) |

## Mathematical Modules (14)

All in `src/delta72/`:

| Tier | Module | Concept |
|------|--------|---------|
| Core | `engine.py` | Δ equation + M, W operators |
| 1 | `rqa.py` | Recurrence Quantification Analysis |
| 1 | `wavelet.py` | Wavelet Coherence |
| 1 | `lyapunov.py` | Lyapunov Exponents |
| 1 | `bocpd.py` | Bayesian Change Point Detection |
| 2 | `transfer_entropy.py` | Directed Information Flow |
| 2 | `phase_space.py` | Phase Space Reconstruction |
| 2 | `rmt.py` | Random Matrix Theory |
| 2 | `tda.py` | Persistent Homology / TDA |
| 2 | `granger.py` | Granger Causality |
| 3 | `koopman.py` | Koopman Operator Theory |
| 3 | `info_geometry.py` | Information Geometry |
| 3 | `ergodic.py` | Ergodic Theory |
| 3 | `stochastic_resonance.py` | Stochastic Resonance |
| Canon | `extended_operators.py` | F, B, L + regime classifier |

## Published Results

- [jthorvaldur.github.io/r/research/](https://jthorvaldur.github.io/r/research/) — experiment dashboards
- [jthorvaldur.github.io/r/d72/](https://jthorvaldur.github.io/r/d72/) — canonical framework

## Stack

- Python, uv
- NumPy, SciPy, Pandas, Matplotlib
- Pure numpy math implementations (no external ML/stats libraries)
