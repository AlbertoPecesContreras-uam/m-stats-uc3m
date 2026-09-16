# =============================================================================
# compute_eeg_correlation_by_regions.py
# =============================================================================
# Description  : Computes regional Pearson connectivity metrics rigorously.
#                1. Electrode-to-electrode Pearson correlation.
#                2. Absolute Fisher's Z transform to prevent phase cancellation.
#                3. Average Z-values within region pairs.
#                4. Derive |r| and R² from the averaged Z values.
#
# Input        : full_eeg_stacked.parquet  (produced by build_eeg_dataset.py)
#
# Output       : pearson_results_by_regions.parquet
#                Columns: instant, id, cond, group, ch_i, ch_j,
#                         abs_r, pearson_z, r_squared
# =============================================================================

from __future__ import annotations

import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import EEG_REGIONS, SHAM_IDS

# ---------
# Constants
# ---------
INPUT_FILE  = Path("full_eeg_stacked.parquet")
OUTPUT_FILE = Path("pearson_results_by_regions.parquet")

# --------------------------------------------------
# Core: Rigorous Regional Pearson Connectivity
# --------------------------------------------------
def compute_regional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute channel-to-channel Pearson correlation, transform to absolute
    Fisher's Z space, average within region pairs, then derive |r| and R².

    Averaging in Z space (rather than r space) gives a statistically valid
    summary of regional connectivity.
    """
    valid_channels = [
        ch for chs in EEG_REGIONS.values() for ch in chs if ch in df.columns
    ]
    if not valid_channels:
        raise KeyError("No valid EEG channels found in the DataFrame.")

    # Channel-to-channel Pearson correlation matrix
    corr_mat = df[valid_channels].corr(method='pearson')

    # Absolute Fisher's Z (clip to avoid arctanh(1) = ∞)
    r_abs  = np.clip(np.abs(corr_mat.to_numpy()), 0, 0.9999)
    z_vals = np.arctanh(r_abs)
    df_z   = pd.DataFrame(z_vals, index=corr_mat.index, columns=corr_mat.columns)

    # Average in Z-space by region pair and derive bounded metrics
    regions = list(EEG_REGIONS.keys())
    records: list[dict] = []

    for r_i, r_j in product(regions, regions):
        chs_i = [ch for ch in EEG_REGIONS[r_i] if ch in df_z.index]
        chs_j = [ch for ch in EEG_REGIONS[r_j] if ch in df_z.columns]

        if not (chs_i and chs_j):
            continue

        regional_z     = df_z.loc[chs_i, chs_j].to_numpy().mean()
        regional_abs_r = float(np.tanh(regional_z))
        regional_r2    = regional_abs_r ** 2

        records.append({
            'ch_i':      r_i,
            'ch_j':      r_j,
            'abs_r':     regional_abs_r,
            'pearson_z': regional_z,
            'r_squared': regional_r2,
        })

    return pd.DataFrame(records)

# -------------------------------------------------------
# Worker: one (instant × subject × condition) combination
# -------------------------------------------------------
def _compute_single(df_indexed: pd.DataFrame, instant: str,
                    subj_id: int, cond: str) -> pd.DataFrame | None:
    """
    Extract one subject's segment, compute regional metrics, and attach
    structural metadata.
    """
    try:
        subset = df_indexed.loc[(instant, subj_id, cond)].reset_index()
    except KeyError:
        return None

    group = 'Sham' if subj_id in SHAM_IDS else subset['group'].iloc[0]
    long  = compute_regional_metrics(subset)

    long['instant'] = instant
    long['id']      = subj_id
    long['cond']    = cond
    long['group']   = group

    return long

# ------------
# Orchestrator
# ------------
def compute_and_save_correlations(
        df_all: pd.DataFrame,
        output_path: Path = OUTPUT_FILE,
        n_jobs: int = max(1, (os.cpu_count() or 4) - 1),
) -> pd.DataFrame:
    """
    Orchestrate parallel computation across all (instant × subject × condition)
    combinations.

    Results are cached: if the output file already exists it is loaded
    directly, skipping all computation.
    """
    if output_path.exists():
        print(f"Output already exists at '{output_path}'. Loading from disk...")
        return pd.read_parquet(output_path)

    df_indexed = df_all.set_index(['instant', 'id', 'cond']).sort_index()

    combos = list(product(
        df_all['instant'].unique(),
        df_all['id'].unique(),
        df_all['cond'].unique(),
    ))
    print(f"Dispatching {len(combos)} tasks across {n_jobs} workers...\n")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_compute_single)(df_indexed, instant, subj_id, cond)
        for instant, subj_id, cond in combos
    )

    df_corr = pd.concat([r for r in results if r is not None], ignore_index=True)

    # Remove self-connections (diagonal)
    df_corr = df_corr[df_corr['ch_i'] != df_corr['ch_j']].reset_index(drop=True)

    # Optimise storage dtypes
    for col in ['instant', 'cond', 'group', 'ch_i', 'ch_j']:
        df_corr[col] = df_corr[col].astype('category')
    for col in ['abs_r', 'pearson_z', 'r_squared']:
        df_corr[col] = df_corr[col].astype('float32')

    df_corr.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df_corr):,} rows → '{output_path}'")

    return df_corr

# ----
# Main
# ----
def main() -> None:
    df_all = pd.read_parquet(INPUT_FILE)
    print(f"Loaded '{INPUT_FILE}': {df_all.shape[0]:,} rows × {df_all.shape[1]} columns")
    print(f"Subjects : {df_all['id'].nunique()}")
    print(f"Instants : {sorted(df_all['instant'].unique())}")
    print(f"Conditions: {sorted(df_all['cond'].unique())}\n")

    df_corr = compute_and_save_correlations(df_all)
    print(f"\nFinal Pearson table: {df_corr.shape[0]:,} rows × {df_corr.shape[1]} columns")


if __name__ == "__main__":
    main()
