\# 🧠 EEG Resting-State Analysis Pipeline

Master's Final Thesis - Universidad Carlos III de Madrid

\*\*Thesis title:\*\* Non-parametric EEG Spectral Decomposition via Whittle-EM: Effects of tDCS and Exercise

\*\*Author:\*\* Alberto Peces Contreras  

\*\*Supervisors:\*\* Juan Miguel Marín Diazaraque and Cristina Nombela Otero  

\*\*Institution:\*\* Universidad Carlos III de Madrid  

\*\*Year:\*\* 2026  




\---

\## 🧾 Description

This repository contains the complete source code for the analytical pipeline of the master's thesis. The project analyses resting-state EEG recordings from a three-arm intervention study (Control, Sham, Experimental) at three time points (PRE, POST, Follow-Up) and two recording conditions (Eyes Open, Eyes Closed).



The pipeline is divided into four sequential modules:



```

EDF recordings

&#x20;     │

&#x20;     ▼

┌─────────────────────────┐

│  01  Preprocessing      │  ICA-based artefact removal          MATLAB R2024a

│      (MATLAB)           │  → preprocessed CSVs (20 ch × N)

└────────────┬────────────┘

&#x20;            │

&#x20;            ▼

┌─────────────────────────┐

│  02  Connectivity       │  Spectral coherence (C) and           Python 3.12.2

│      (Python)           │  squared Pearson correlation (ρ²)

└────────────┬────────────┘

&#x20;            │

&#x20;            ▼

┌─────────────────────────┐

│  03  Spectral           │  Xi-Pi decomposition → periodic (Π)  Python 3.12.2

│      (Python)           │  and aperiodic (Ξ) components

└────────────┬────────────┘

&#x20;            │

&#x20;            ▼

┌─────────────────────────┐

│  04  Statistics         │  Bayesian LMMs with full posterior    Python 3.12.2

│      (Python)           │  uncertainty quantification

└─────────────────────────┘

```



\---



\## 📁 Repository Structure



```

.

├── 01\_preprocessing/

│   ├── 01\_fitGlobalICA.m          # Fit group-level ICA model on concatenated dataset

│   ├── 02\_runPreprocessing.m      # Apply ICA per recording, remove blink artefacts

│   ├── utils/

│   │   ├── readEDF.m              # Read .edf files into MATLAB structs

│   │   ├── applyPassBand.m        # Zero-phase Butterworth bandpass filter \[0.1–30 Hz]

│   │   ├── StructToDouble.m       # Convert EEG struct → \[channels × samples] matrix

│   │   ├── DoubleToStruct.m       # Convert matrix → EEG struct

│   │   └── extractDirectory.m     # Build folder tree structure from a root path

│   ├── pca\_ica/                   # FastICA implementation (third-party)

│   └── demo/

│       └── demo\_eyeBlinkRemoval.m # Diagnostic: visualise ICA blink removal on one file

│

├── 02\_connectivity/

│   ├── config.py                  # Shared constants (regions, groups, Sham IDs)

│   ├── 01\_compute\_coherence.py    # Magnitude-squared coherence per region pair

│   ├── 02\_compute\_correlation.py  # Pearson ρ² via Fisher's Z averaging

│   └── connectivity\_analysis.ipynb# LMMs + Δ connectivity heatmaps

│

├── 03\_spectral/

│   ├── xipi.py                    # Xi-Pi algorithm (Python port of Hu et al., 2024)

│   └── 01\_extract\_components.py   # Batch Xi-Pi decomposition + time-domain reconstruction

│

├── 04\_statistics/

│   └── bayesian\_lmm.ipynb         # Bayesian LMMs for Π and Ξ oscillatory energy

│

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\## 📚 Requirements



\### MATLAB (module 01)



\- MATLAB R2024a

\- Signal Processing Toolbox

\- The `pca\_ica/` folder must be on the MATLAB path (handled automatically by the scripts via `addpath`)



\### Python (modules 02–04)



Python 3.12.2. Key packages:



| Package | Version | Purpose |

|---|---|---|

| `numpy` | ≥ 1.26 | Numerical computation |

| `pandas` | ≥ 2.2 | Data wrangling |

| `scipy` | ≥ 1.13 | Signal processing (Welch, CSD, filters) |

| `joblib` | ≥ 1.4 | Parallel computation |

| `pyarrow` | ≥ 16 | Parquet I/O |

| `pymc` | ≥ 5.10 | Bayesian inference |

| `arviz` | ≥ 0.18 | MCMC diagnostics and HDI |

| `scikit-learn` | ≥ 1.4 | Isotonic regression (Xi-Pi) |

| `matplotlib` | ≥ 3.8 | Figures |

| `seaborn` | ≥ 0.13 | Heatmaps |



\---



\## Usage



Scripts must be run in module order. Each module reads the outputs of the previous one.



\### 01 · Preprocessing (MATLAB)



Set the data paths at the top of each script, then run in order:



```matlab

% Step 1 — Fit the global ICA unmixing matrix across all subjects

run('01\_preprocessing/01\_fitGlobalICA.m')



% Step 2 — Apply ICA per recording, remove eye-blink artefacts, save CSVs

run('01\_preprocessing/02\_runPreprocessing.m')

```



\*\*Output:\*\* one CSV per subject per condition, format `\[N × 24]` (4 metadata columns + 20 EEG channels).



\### 02 · Connectivity (Python)



```bash

cd 02\_connectivity



\# Compute spectral coherence for all subjects

python 01\_compute\_coherence.py



\# Compute Pearson correlation for all subjects

python 02\_compute\_correlation.py



\# Open the notebook for LMM modelling and visualisation

jupyter notebook connectivity\_analysis.ipynb

```



\*\*Output:\*\* `coherence\_results\_by\_regions.parquet`, `pearson\_results\_by\_regions.parquet`.



\### 03 · Spectral Decomposition (Python)



```bash

cd 03\_spectral



\# Run Xi-Pi decomposition on all preprocessed recordings

python 01\_extract\_components.py

```



\*\*Output:\*\* one CSV per subject with `\_xi` and `\_pi` columns per channel, plus `xipi\_metrics\_summary.csv` with reconstruction quality metrics.



\### 04 · Statistics (Python)



```bash

cd 04\_statistics

jupyter notebook bayesian\_lmm.ipynb

```



Run the \*\*Periodic (Π)\*\* section first, then the \*\*Aperiodic (Ξ)\*\* section. The notebook fits four Bayesian LMMs (2 components × 2 analysis windows) and generates publication-ready posterior distribution figures.



\---



\## ⚒️ Methods Overview



\### EEG recording



Resting-state EEG recorded at \*f\*s = 500 Hz, 20 scalp electrodes (10–20 system), two 3-minute conditions: Eyes Open (EO) and Eyes Closed (EC).



\### Preprocessing



Eye-blink artefacts are removed using Independent Component Analysis (ICA). A group-level unmixing matrix \*\*W\*\* is first estimated on the full concatenated dataset; individual recordings are then decomposed with warm-start initialisation from \*\*W\*\*. The two components most correlated with the averaged Fp1/Fp2 signal are zeroed out before signal reconstruction. A zero-phase Butterworth bandpass filter \[0.1–30 Hz, 4th-order effective] is applied to all channels.



\### Functional Connectivity (module 02)



Two complementary metrics capture inter-regional coupling:



\- \*\*Spectral coherence C\*\* — magnitude-squared coherence computed electrode-pair by electrode-pair and averaged within region pairs to prevent phase cancellation.

\- \*\*Squared Pearson correlation ρ²\*\* — electrode-level correlations are transformed to Fisher's Z, averaged within region pairs, and back-transformed to obtain a bounded |r| and ρ².



\### Spectral Decomposition (module 03)



The Xi-Pi model (Hu et al., 2024) decomposes each channel's power spectrum into a monotonically decreasing \*\*aperiodic component Ξ\*\* (1/\*f\* noise) and \*\*unimodal periodic components Π\*\* (oscillatory peaks) via an Expectation-Maximisation loop using the Whittle likelihood. Components are reconstructed in the time domain via a frequency-domain Wiener filter applied to the raw FFT coefficients.



\### Statistical Modelling (module 04)



Linear Mixed-Effects Models are specified in a Bayesian framework using PyMC. The outcome variable is the log-RMS amplitude (dB) averaged per anatomical region. Subject-level random effects and region effects use non-centred parameterisations to improve sampler efficiency. Uncertainty is reported as 95 % Highest Density Intervals (HDI).



Two analysis windows are modelled:



| Window | Groups | Fixed effects |

|---|---|---|

| PRE → POST | CTRL, SHAM, EXP | group + instant + group×instant + region |

| POST → FU | SHAM, EXP | group + instant + group×instant + region |



\---



\## References



Hu, S., Zhang, Z., Zhang, X., Wu, X., \& Valdes-Sosa, P. A. (2024). Xi-Pi: A nonparametric model for neural power spectra decomposition. \*IEEE Journal of Biomedical and Health Informatics\*, \*28\*(5), 2624–2635. https://doi.org/10.1109/JBHI.2024.3364499



Original Xi-Pi MATLAB implementation: https://github.com/ShiangHu/Xi-Pi



\---

