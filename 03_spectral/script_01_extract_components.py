"""
=================================================================================
EEG COMPONENT RECONSTRUCTION PIPELINE: XI-PI DECOMPOSITION & TIME-DOMAIN SYNTHESIS
=================================================================================

AUTHOR: Alberto Peces Contreras
DATE: 2026
VERSION: 2.0
FRAMEWORK: ScMEM (Spectral Component Mixture EM Model)

OVERVIEW:
This pipeline automates the extraction of Periodic (Pi) and Aperiodic (Xi) 
components from raw EEG signals. Utilizing a frequency-domain Wiener Filter 
approach, the system decomposes multi-channel EEG records and reconstructs 
them back into the time domain while preserving the original phase information.

METHODOLOGICAL STEPS:
1. SPECTRAL ESTIMATION: Computes PSD via Welch's method (0-30 Hz).
2. COMPONENT MODELING: Applies the Xi-Pi algorithm to isolate rhythmic peaks 
   from the 1/f background activity.
3. WIENER RECONSTRUCTION: Generates amplitude masks to filter the complex 
   Fourier spectrum of the raw signal.
4. TIME SYNTHESIS: Applies Inverse FFT and energy normalization based on 
   Parseval's Theorem to ensure high-fidelity reconstruction (Corr ~ 0.97).
5. VALIDATION: Computes RMSE and cross-correlation between the original 
   and the reconstructed time-series per channel.

HARDWARE OPTIMIZATION & PERFORMANCE:
The architecture is designed for high-throughput batch processing using 
asynchronous multithreading (ThreadPoolExecutor). 

    - TARGET HARDWARE: Intel Core i9-14900K (14th Gen) | 14 Cores | 32 GB RAM.
    - WORKER CONFIGURATION: 12 Concurrent threads.
    - THROUGHPUT: ~125 records per hour.
    - TOTAL DATASET ESTIMATION: ~5-6 hours for a full cohort (~500-600 subjects).

OUTPUT STRUCTURE:
- Individual CSVs: Reconstructed Xi and Pi signals per channel (44-column format).
- Metrics Summary: 'xipi_metrics_summary.csv' containing global error stats.
=================================================================================
"""
# -------
# MODULES
# -------
from concurrent.futures import ThreadPoolExecutor, as_completed

import os

import numpy as np
import pandas as pd
from scipy.signal import welch, correlate
from tqdm import tqdm

from xipi import XiPi

# ------------------------------
# 1. DIRECTORY & FILE MANAGEMENT
# ------------------------------
def get_files(path_in):
    """
    Retrieve all file paths from a given list of directories.

    Parameters
    ----------
    path_in : list of str
        Directory paths to scan (e.g. ["E:\\...\\POST\\CTRL", "E:\\...\\PRE\\EXP"]).

    Returns
    -------
    list of str
        Absolute paths of all files found across the input directories.
    """
    all_files = []
    for path in path_in:
        # os.scandir is significantly faster than os.listdir + os.path.isdir
        # because it caches file properties, avoiding extra stat() calls.
        folder_files = [
            entry.path for entry in os.scandir(path) if entry.is_file()
        ]
        all_files.extend(folder_files)
    return all_files


def setup_directory_structure(base_path, timepoints, groups, verbose=0):
    """
    Initialise the output folder hierarchy.

    Creates nested subdirectories for each time point and experimental group,
    ensuring a standardised structure for the output CSV files.

    Parameters
    ----------
    base_path  : str   Root directory for all output files.
    timepoints : list  Time-point folder names (e.g. ["PRE", "POST", "SEGUIMIENTO"]).
    groups     : list  Group folder names (e.g. ["CONTROL", "EXP"]).
    verbose    : int   If non-zero, prints a confirmation message.
    """
    for tp in timepoints:
        for grp in groups:
            os.makedirs(os.path.join(base_path, tp, grp), exist_ok=True)
    if verbose:
        print(f"Directory structure verified at: {base_path}")

# -------------------------------------
# 2. SPECTRAL ANALYSIS (Welch's Method)
# -------------------------------------
def compute_psd(signal, fs=500, window_sec=4, overlap_pct=0.5, f_max=30):
    """
    Estimate the Power Spectral Density (PSD) using Welch's method.

    Provides the frequency-domain foundation for the Xi-Pi decomposition
    by filtering the spectrum within the physiologically relevant band (0-30 Hz).

    Parameters
    ----------
    signal      : array-like  Input time-series.
    fs          : int         Sampling frequency in Hz.
    window_sec  : float       Window length in seconds.
    overlap_pct : float       Fractional window overlap (0–1).
    f_max       : float       Upper frequency bound in Hz.

    Returns
    -------
    freqs : np.ndarray  Frequency vector (Hz).
    psd   : np.ndarray  Power spectral density.
    """
    nperseg  = int(window_sec * fs)
    noverlap = int(nperseg * overlap_pct)

    freqs, psd = welch(signal, fs=fs, window='hann',
                       nperseg=nperseg, noverlap=noverlap)

    mask = (freqs >= 0) & (freqs <= f_max)
    return freqs[mask], psd[mask]

# ---------------------------------------------
# 3. COMPONENT RECONSTRUCTION (Inverse Fourier)
# ---------------------------------------------
def time_domain_reconstruction(eeg_raw, f_scmem, complete_spectrum, peaks_pi, pink_noise_xi, fs):
    """
    Synthesise Xi and Pi time-domain signals via a Wiener filter.

    Applies a Wiener filter mask to the complex Fourier coefficients of the
    raw EEG, preserving the original phase while isolating the Aperiodic (Xi)
    and Periodic (Pi) components. A final scaling step based on Parseval's
    Theorem ensures that time-domain energy matches the theoretical spectral power.

    Parameters
    ----------
    eeg_raw           : np.ndarray  Raw EEG time-series.
    f_scmem           : np.ndarray  Frequency vector from the Xi-Pi model (Hz).
    complete_spectrum  : np.ndarray  Total fitted PSD (Xi + Pi + noise).
    peaks_pi          : np.ndarray  (N x K) periodic component matrix.
    pink_noise_xi     : np.ndarray  Aperiodic component vector.
    fs                : int         Sampling frequency in Hz.

    Returns
    -------
    eeg_pi_final : np.ndarray  Reconstructed periodic (Pi) signal.
    eeg_xi_final : np.ndarray  Reconstructed aperiodic (Xi) signal.
    """
    # 1. Fourier transform of the raw EEG
    # Decomposes the original signal into amplitude and phase components.
    n        = len(eeg_raw)
    fft_raw  = np.fft.rfft(eeg_raw, n=n)
    freqs_fft = np.fft.rfftfreq(n, d=1/fs)

    # 2. Compute Wiener filter ratios (power proportion per component)
    # Ratio = PSD_Component / Total_PSD
    ratio_pi = np.sum(peaks_pi, axis=1) / (complete_spectrum + 1e-12)
    ratio_xi = pink_noise_xi / (complete_spectrum + 1e-12)

    # 3. Power-to-amplitude conversion (square root)
    # Yields the multiplicative factor applied to the FFT amplitude.
    amp_mask_pi = np.sqrt(np.clip(ratio_pi, 0, 1))
    amp_mask_xi = np.sqrt(np.clip(ratio_xi, 0, 1))

    # 4. FFT interpolation (cut-off at 30 Hz)
    # Xi-Pi uses 0.25 Hz resolution; FFT of a long signal has thousands of
    # points. Linear interpolation fills the intermediate values.
    mask_pi_interp = np.interp(freqs_fft, f_scmem, amp_mask_pi, left=0, right=0)
    mask_xi_interp = np.interp(freqs_fft, f_scmem, amp_mask_xi, left=0, right=0)

    # 5. Inverse FFT — original phase is preserved via the complex FFT coefficients
    eeg_pi_res = np.fft.irfft(fft_raw * mask_pi_interp, n=n)
    eeg_xi_res = np.fft.irfft(fft_raw * mask_xi_interp, n=n)

    # 6. Energy scaling (Parseval's Theorem)
    # Adjusts time-domain amplitude so that energy matches the spectral power estimate.
    df = f_scmem[1] - f_scmem[0]

    def adjust_scale(signal, pot_thery):
        if np.std(signal) > 0:
            return (signal / np.std(signal)) * np.sqrt(np.sum(pot_thery) * df)
        return signal

    eeg_pi_final = adjust_scale(eeg_pi_res, np.sum(peaks_pi, axis=1))
    eeg_xi_final = adjust_scale(eeg_xi_res, pink_noise_xi)

    return eeg_pi_final, eeg_xi_final

# ---------------------------------------
# 4. SIGNAL ALIGNMENT & CROSS-CORRELATION
# ---------------------------------------
def align_signals(ref, est, fs):
    """
    Identify and correct temporal lag between a reference and an estimate signal.

    Computes the cross-correlation to find the best-lag offset, then shifts
    the estimate to align it with the reference. Used to verify phase accuracy
    and to compute lag-corrected precision metrics.

    Parameters
    ----------
    ref : np.ndarray  Reference (original) signal.
    est : np.ndarray  Estimated (reconstructed) signal.
    fs  : int         Sampling frequency in Hz.

    Returns
    -------
    corr_est : np.ndarray  Lag-corrected estimate.
    best_lag : int          Optimal lag in samples.
    dt       : float        Optimal lag in seconds.
    """
    n        = len(ref)
    corr     = correlate(ref - np.mean(ref), est - np.mean(est), mode='full')
    lags     = np.arange(-n + 1, n)
    best_lag = lags[np.argmax(corr)]
    if best_lag > 0:
        corr_est = np.pad(est, (best_lag, 0), mode='edge')[:n]
    elif best_lag < 0:
        corr_est = np.pad(est, (0, abs(best_lag)), mode='edge')[abs(best_lag):]
    else:
        corr_est = est
    return corr_est, best_lag, best_lag / fs

# --------------------------------------------
# 5. CORE PROCESSING PIPELINE (Single Subject)
# --------------------------------------------
def process_single_record(file_path, base_path):
    """
    Orchestrate the end-to-end Xi-Pi workflow for one EEG recording.

    Extracts metadata from the filename, runs spectral decomposition and
    time-domain reconstruction per channel, computes error metrics, and
    saves the result as a 44-column CSV.

    Parameters
    ----------
    file_path : str  Full path to the preprocessed EEG CSV file.
    base_path : str  Root directory for saving output files.

    Returns
    -------
    dict or None
        Row of quality metrics for this recording, or None on failure.
    """
    try:
        # 1. Metadata extraction from filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        parts     = base_name.split("_")
        instant, group, id_subj, name, cond = parts[:5]

        # Normalise for directory matching
        tp_folder  = "SEGUIMIENTO" if instant.upper() == "SEG" else instant.upper()
        grp_folder = group.upper()

        # 2. Load data and identify EEG channels (exclude metadata columns)
        df_raw      = pd.read_csv(file_path, engine='pyarrow')
        info_cols   = ['cond', 'instant', 'group', 'id']
        eeg_channels = [col for col in df_raw.columns if col not in info_cols]
        num_samples  = len(df_raw)

        # 3. Initialise output containers
        row_metrics = {'cond': cond, 'instant': instant,
                       'group': group, 'id': id_subj,
                       'filepath': file_path}

        df_out = pd.DataFrame({'cond':    [cond]     * num_samples,
                               'instant': [instant]  * num_samples,
                               'group':   [group]    * num_samples,
                               'id':      [id_subj]  * num_samples})

        # 4. Compute PSD + XiPi + Reconstruction per channel
        for ch in eeg_channels:

            # --- A. Welch's method ---
            eeg_raw = df_raw[ch].values
            freqs, psd_values = compute_psd(eeg_raw)

            # --- B. Initialisation ---
            sigk_ini, _, _ = XiPi.initialize_scmem(freqs, psd_values)

            # --- C. Xi-Pi decomposition ---
            results = XiPi.scmem_em_loop(freqs, psd_values, sigk_ini,
                                          max_iter=1000, verbose=False)

            # --- D. Extract components ---
            complete_spectrum = results['psd_ftd']
            pink_noise_est_xi = results['sigk'][:, 0]
            peaks_est_pi      = results['sigk'][:, 1:]

            # --- E. Time-domain reconstruction ---
            eeg_pi_est, eeg_xi_est = time_domain_reconstruction(
                eeg_raw, freqs, complete_spectrum,
                peaks_est_pi, pink_noise_est_xi, fs=500,
            )
            eeg_estimate = eeg_xi_est + eeg_pi_est

            # --- F. Phase correction ---
            eeg_global_corr, lag_global, dt_global = align_signals(
                eeg_raw, eeg_estimate, fs=500,
            )

            # --- G. Error metrics ---
            # Raw vs. reconstructed signal
            corr_global = np.corrcoef(eeg_raw, eeg_global_corr)[0, 1]
            rmse_global = np.sqrt(np.mean((eeg_raw - eeg_global_corr) ** 2))
            # Xi vs. Pi cross-check
            corr_xi_pi = np.corrcoef(eeg_xi_est, eeg_pi_est)[0, 1]
            rmse_xi_pi = np.sqrt(np.mean((eeg_xi_est - eeg_pi_est) ** 2))

            row_metrics[f'corr_global_{ch}'] = corr_global
            row_metrics[f'rmse_global_{ch}'] = rmse_global
            row_metrics[f'corr_xi_pi_{ch}']  = corr_xi_pi
            row_metrics[f'rmse_xi_pi_{ch}']  = rmse_xi_pi

            df_out[f'{ch}_xi'] = eeg_xi_est.astype(np.float32)
            df_out[f'{ch}_pi'] = eeg_pi_est.astype(np.float32)

        # Save reconstructed signals (one CSV per subject)
        fname     = f"{instant}_{group}_{id_subj}_{name}_{cond}.csv"
        save_path = os.path.join(base_path, tp_folder, grp_folder, fname)
        df_out.to_csv(save_path, index=False)

        return row_metrics

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

# ------------
# MAIN PROGRAM
# ------------
post_ctrl_path = r'E:\TFM_UC3M\PREPROCESADO\EEG\POST\CONTROL'
post_exp_path  = r'E:\TFM_UC3M\PREPROCESADO\EEG\POST\EXP'
pre_ctrl_path  = r'E:\TFM_UC3M\PREPROCESADO\EEG\PRE\CONTROL'
pre_exp_path   = r'E:\TFM_UC3M\PREPROCESADO\EEG\PRE\EXP'
seg_ctrl_path  = r'E:\TFM_UC3M\PREPROCESADO\EEG\SEGUIMIENTO\CONTROL'
seg_exp_path   = r'E:\TFM_UC3M\PREPROCESADO\EEG\SEGUIMIENTO\EXP'
paths     = [pre_ctrl_path, post_ctrl_path, seg_ctrl_path,
             pre_exp_path,  post_exp_path,  seg_exp_path]
base_path = r'E:\TFM_UC3M\PREPROCESADO\EEG_Reconstructed_Components'

if __name__ == "__main__":

    path_list = get_files(paths)
    setup_directory_structure(base_path, ["PRE", "POST", "SEGUIMIENTO"], ["CONTROL", "EXP"])

    print(f"Starting reconstruction pipeline for {len(path_list)} records...")

    results = []

    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_file = {
            executor.submit(process_single_record, path, base_path): path
            for path in path_list
        }
        with tqdm(total=len(path_list), desc="Processing EEG records", unit="file") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                file_name = os.path.basename(file_path)
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                    pbar.set_postfix_str(f"Last: {file_name[:25]}...")
                    pbar.update(1)
                except Exception as e:
                    print(f"\n[ERROR] Thread failed for file {file_name}: {e}")
                    pbar.update(1)

    if results:
        df_metrics = pd.DataFrame(results)
        df_metrics.to_csv("xipi_metrics_summary.csv", index=False)
        print(f"\nPIPELINE SUCCESS! {len(results)} subjects processed successfully.")
        print(f"Global summary saved as: 'xipi_metrics_summary.csv'")
    else:
        print("\nPIPELINE FAILED: No data was successfully processed.")
