# =============================================================================
# compute_eeg_coherence_by_regions.py
# =============================================================================
# Description  : Computes spectral coherence matrices for all subjects across
#                all experimental conditions (PRE / POST / SEGUIMIENTO) and
#                study arms (CONTROL / EXP / SHAM).
#
#                Methodology: coherence is computed pair-by-pair for every
#                individual electrode combination between regions, then the
#                raw values are averaged within region pairs to prevent phase
#                cancellation caused by premature temporal averaging.
#
# Input        : full_eeg_stacked.parquet  (produced by build_eeg_dataset.py)
#
# Output       : coherence_results_by_regions.parquet
#                Columns: instant, id, cond, group, ch_i, ch_j, coherence
# =============================================================================

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import csd, welch

from config import EEG_REGIONS, SHAM_IDS, FS, FREQ_MIN, FREQ_MAX

# ---------
# Constants
# ---------
FREQ_RES    = 0.25   # Spectral resolution (Hz) → nperseg = FS / FREQ_RES = 2000
OVERLAP     = 0.5    # Hann window overlap fraction

INPUT_FILE  = Path("full_eeg_stacked.parquet")
OUTPUT_FILE = Path("coherence_results_by_regions.parquet")

# --------------------------------------------------
# Core: Robust Channel-by-Channel Regional Coherence
# --------------------------------------------------
def compute_regional_coherence_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute magnitude-squared coherence for every individual electrode pair
    across regions, then average those values to obtain region-to-region
    connectivity.

    Averaging raw coherence values (rather than pre-averaging signals)
    prevents phase cancellation.
    """
    nperseg  = int(FS / FREQ_RES)       # 2000 samples → 0.25 Hz resolution
    noverlap = int(nperseg * OVERLAP)    # 1000 samples → 50 % overlap

    # Pre-compute auto-spectra (PSD) for all valid electrodes
    valid_channels = list({
        ch for chs in EEG_REGIONS.values() for ch in chs if ch in df.columns
    })

    psds: dict[str, np.ndarray] = {}
    freqs: np.ndarray | None = None
    for ch in valid_channels:
        freqs, pxx = welch(df[ch].values, fs=FS, window='hann',
                           nperseg=nperseg, noverlap=noverlap)
        psds[ch] = pxx

    freq_mask    = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    region_names = list(EEG_REGIONS.keys())
    rows: list[dict] = []

    # Iterate over region pairs (upper triangle)
    for r_i, r_j in combinations(region_names, 2):
        chs_i = [ch for ch in EEG_REGIONS[r_i] if ch in df.columns]
        chs_j = [ch for ch in EEG_REGIONS[r_j] if ch in df.columns]

        if not chs_i or not chs_j:
            continue

        pair_coherences: list[float] = []

        for ch_i in chs_i:
            for ch_j in chs_j:
                _, pxy = csd(df[ch_i].values, df[ch_j].values, fs=FS,
                             window='hann', nperseg=nperseg, noverlap=noverlap)

                # Magnitude-squared coherence, clipped denominator to avoid /0
                denom        = psds[ch_i][freq_mask] * psds[ch_j][freq_mask] + 1e-12
                coh_spectrum = (np.abs(pxy[freq_mask]) ** 2) / denom
                pair_coherences.append(float(np.mean(coh_spectrum)))

        mean_coh = float(np.mean(pair_coherences))

        # Store both directions for the long-format output
        rows.append({'ch_i': r_i, 'ch_j': r_j, 'coherence': mean_coh})
        rows.append({'ch_i': r_j, 'ch_j': r_i, 'coherence': mean_coh})

    # Diagonal: a region is perfectly coherent with itself
    for r in region_names:
        rows.append({'ch_i': r, 'ch_j': r, 'coherence': 1.0})

    return pd.DataFrame(rows)

# -------------------------------------------------------
# Worker: one (instant × subject × condition) combination
# -------------------------------------------------------
def _compute_single(df_indexed: pd.DataFrame, instant: str,
                    subj_id: int, cond: str) -> pd.DataFrame | None:
    """
    Extract one subject's segment, compute regional coherence, and attach
    structural metadata.
    """
    try:
        subset = df_indexed.loc[(instant, subj_id, cond)].reset_index()
    except KeyError:
        return None

    group = 'Sham' if subj_id in SHAM_IDS else subset['group'].iloc[0]
    long  = compute_regional_coherence_long(subset)

    long['instant'] = instant
    long['id']      = subj_id
    long['cond']    = cond
    long['group']   = group

    return long

# ------------
# Orchestrator
# ------------
def compute_and_save_coherences(
        df_all: pd.DataFrame,
        output_path: Path = OUTPUT_FILE,
        n_jobs: int = max(1, (os.cpu_count() or 4) - 1),
) -> pd.DataFrame:
    """
    Orchestrate parallel regional coherence computation across all
    (instant × subject × condition) combinations.

    Results are cached: if the output file already exists it is loaded
    directly, skipping all computation.
    """
    if output_path.exists():
        print(f"Output already exists at '{output_path}'. Loading from disk...")
        return pd.read_parquet(output_path)

    # Pre-index once for O(1) lookups inside workers
    df_indexed = df_all.set_index(['instant', 'id', 'cond']).sort_index()

    combos = [
        (instant, subj_id, cond)
        for instant  in df_all['instant'].unique()
        for subj_id  in df_all['id'].unique()
        for cond     in df_all['cond'].unique()
    ]
    print(f"Dispatching {len(combos)} tasks across {n_jobs} workers...\n")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_compute_single)(df_indexed, instant, subj_id, cond)
        for instant, subj_id, cond in combos
    )

    df_coh = pd.concat([r for r in results if r is not None], ignore_index=True)

    # Optimise storage dtypes
    for col in ['instant', 'cond', 'group', 'ch_i', 'ch_j']:
        df_coh[col] = df_coh[col].astype('category')
    df_coh['coherence'] = df_coh['coherence'].astype('float32')

    df_coh.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df_coh):,} rows → '{output_path}'")

    return df_coh

# ----
# Main
# ----
def main() -> None:
    df_all = pd.read_parquet(INPUT_FILE)
    print(f"Loaded '{INPUT_FILE}': {df_all.shape[0]:,} rows × {df_all.shape[1]} columns")
    print(f"Subjects : {df_all['id'].nunique()}")
    print(f"Instants : {sorted(df_all['instant'].unique())}")
    print(f"Conditions: {sorted(df_all['cond'].unique())}\n")

    df_coh = compute_and_save_coherences(df_all)
    print(f"\nFinal coherence table: {df_coh.shape[0]:,} rows × {df_coh.shape[1]} columns")


if __name__ == "__main__":
    main()
