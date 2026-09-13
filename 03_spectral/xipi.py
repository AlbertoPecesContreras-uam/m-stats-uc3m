"""
===============================================================================
Xi-Pi Module: Nonparametric Neural Power Spectra Decomposition
===============================================================================

DESCRIPTION:
This tool is a Python implementation of the nonparametric Xi-Pi model. 
Its main objective is to decompose the power spectrum of neural signals 
(such as EEG), cleanly separating the aperiodic component (1/f background 
noise or Xi) from the periodic components (oscillations/peaks or Pi).

The algorithm is based on an Expectation-Maximization (EM) loop utilizing 
the Whittle likelihood approximation. To model the signal without relying on 
strict parametric assumptions, it uses Isotonic Regression and Gaussian 
smoothing (Shape Language Modeling), enforcing shape constraints (monotonic 
decrease for the aperiodic noise, unimodal shapes for the periodic peaks).

ORIGINAL PAPER REFERENCE:
S. Hu, Z. Zhang, X. Zhang, X. Wu and P. A. Valdes-Sosa, "Xi-Pi: A Nonparametric 
Model for Neural Power Spectra Decomposition," in IEEE Journal of Biomedical 
and Health Informatics, vol. 28, no. 5, pp. 2624-2635, May 2024.
DOI: 10.1109/JBHI.2024.3364499

ORIGINAL REPOSITORY (MATLAB):
https://github.com/ShiangHu/Xi-Pi

PYTHON IMPLEMENTATION BY:
Alberto Peces Contreras
Translated and adapted from the original MATLAB source code.

CLASS STRUCTURE (XiPi):
- initialize_scmem : Robust log-linear initialization of noise and peaks.
- mstep_unim       : Maximization step (Geometric shape fitting).
- scmem_em_loop    : Main EM loop for model convergence.
===============================================================================
"""
# -------
# MODULES
# -------
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

class XiPi:
    # --------------------------
    # 1. M-STEP (Github + Paper)
    # --------------------------
    @staticmethod
    def mstep_unim(sigk_sdo, freqs, smooth=0.5):
        """
        Transcript of 'mstep_unim.m' using Isotonic Regression + Smoothing Gaussian
        to simulate Shape Language Modeling (SLM) from MATLAB. For each component,
        this function fits a non-parametric model and returns the resulting values.

        Parameters
        ----------
        sigk_sdo : (N x K) matrix with Aperiodic (0) + Periodic (1:K) components by columns.
        freqs    : (1 x N) vector with frequency values.
        smooth   : smoothing factor for Gaussian filter (0.5, by default).
        
        Returns
        -------
        sigk_up : (N x K) matrix with updated Aperiodic + Periodic components by columns.
        """

        _, ank = sigk_sdo.shape  # Column 0 = Pink Noise; Columns 1:K = Periodic Components
        sigk_up = np.zeros_like(sigk_sdo)

        for k in range(ank):
            if k == 0:  # Pink Noise (Aperiodic component): it only can go downwards

                # 1) Find the INITIAL MAXIMUM PEAK of low frequency
                # In EEG recordings, the maximum peak is always in the first frequencies.
                max_idx = np.argmax(sigk_sdo[:, k])
                
                # -> [This part could be improved]
                # From 0 to MaxPeak, all values are replaced with the maximum PSD, so that
                # all values are flat at the beginning. This method prevents the Aperiodic
                # Component from disappearing and its energy from being redistributed.
                y_safe = sigk_sdo[:, k].copy()
                y_safe[:max_idx] = y_safe[max_idx]
                y_safe[max_idx:] = y_safe[max_idx:]

                # 2) ISOTONIC REGRESSION for new data
                iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
                yp = iso.fit_transform(freqs, y_safe)

                # 3) SMOOTHING
                yp = gaussian_filter1d(yp, sigma=smooth) 
                
            else:  # Periodic components: From left-upwards to right-downwards
                
                # PCs start going upwards, they reach a summit, and then they go downwards.
                loc = np.argmax(sigk_sdo[:, k])  # Find the peak
                
                # Left side
                freqs_left  = freqs[:loc+1]
                signal_left = sigk_sdo[:loc+1, k]
                if len(freqs_left) > 0:
                    iso_inc = IsotonicRegression(increasing=True, out_of_bounds='clip')
                    y1 = iso_inc.fit_transform(freqs_left, signal_left)
                else:
                    y1 = np.array([])

                # Right side
                freqs_right  = freqs[loc+1:]
                signal_right = sigk_sdo[loc+1:, k]
                if len(freqs_right) > 0:
                    iso_dec = IsotonicRegression(increasing=False, out_of_bounds='clip')
                    y2 = iso_dec.fit_transform(freqs_right, signal_right)
                else:
                    y2 = np.array([])

                yp = np.concatenate([y1, y2])

                # Ensure there are no negative PSD values
                yp = np.clip(yp, 1e-10, None)
            
            sigk_up[:, k] = yp

        return sigk_up

    # --------------------------------
    # 2. MAIN EM-LOOP (Github + Paper)
    # --------------------------------
    @staticmethod
    def scmem_em_loop(freqs, psd_real, sigk_ini, max_iter=100, tol=1e-7, verbose=True):
        """
        Transcript of 'scmem_unim.m' with the original Expectation-Maximization algorithm
        from MATLAB.
        
        Parameters
        ----------
        freqs    : np.array (N,) -> Frequencies in Hz.
        psd_real : np.array (N,) -> Raw periodogram (Welch's method in linear scale).
        sigk_ini : np.ndarray (N, K) -> Initialization [Pink noise, PC1, PC2, ..., PCK].
        max_iter : maximum number of iterations until "convergence".
        tol      : minimum tolerance for convergence.

        Returns
        -------
        results : {"psd_ftd": psd_ftd, "sigk": sigk, "sige": sige, "log_lik": log_lik}
        psd_ftd : np.array (N,) -> Total PSD fit (Final model).
        sigk    : (N x K) matrix with Aperiodic (0) + Periodic (1:K) components by columns.
        sige    : np.array (N,) -> constant white noise.
        log_lik : log likelihood (Whittle likelihood)
        """

        # Column 0 = Pink Noise; Columns 1:K = Periodic Components
        nf, ank = sigk_ini.shape 
        
        # Working copies for iteration
        sigk = sigk_ini.copy()
        sige = np.ones(nf) * 1e-2 * np.mean(psd_real)  # scale relative to the PSD
        
        lh_prev, log_lik = -np.inf, []
        
        for i in range(1, max_iter + 1):
            # -------------------------------------------------
            # E-STEP: Calculate weights with Whittle Likelihood
            # -------------------------------------------------
            # Compute the theoretical PSD as: c = pink noise + Periodic Components
            c = np.sum(sigk, axis=1) + sige 
            
            # w(f) = [I(f) - S(f)] / S(f)**2 (Github's formula)
            w = (psd_real - c) / (c ** 2)

            # This is the same mathematical foundation extracted from the paper.
            # Weight's vector (w) is transformed into a projection matrix. All components
            # are projected onto weights space to compute new pseudo-data. Those values
            # greater or lower than real values are penalized with a negative or positive 
            # factor proportional to the error, respectively.
            w_mat    = np.tile(w[:, None], (1, ank))
            sigk_sdo = sigk + w_mat * (sigk ** 2)  # Pink Noise + Periodic Components
            sige_sdo = sige + w * (sige ** 2)       # Basal Noise
            
            # --------------------------------------------
            # M-STEP: Update components and compute errors
            # --------------------------------------------
            # 1) Geometrically adjust the Pink Noise and Peaks
            sigk = XiPi.mstep_unim(sigk_sdo, freqs)
            
            # 2) White noise is reduced to a flat constant (Github's formula)
            sige = np.mean(sige_sdo) * np.ones(nf)
            
            # -----------
            # CONVERGENCE
            # -----------
            # Whittle approximation is used for the total model (c)
            c_new = np.sum(sigk, axis=1) + sige
            lh = -np.sum(np.log(c_new) + psd_real / c_new)
            log_lik.append(lh)
            
            delta = abs(lh - lh_prev)
            if verbose:
                print(f"Iteration {i:03d} | Log-Likelihood: {lh:.4f} | Tol: {delta:.2e}")
                
            if i > 1 and delta < tol:
                if verbose:
                    print(f">>> Reached convergence successfully in {i} epochs.")
                break
                
            lh_prev = lh
            
        # Build the final model
        psd_ftd = np.sum(sigk, axis=1) + sige
        results = {"psd_ftd": psd_ftd, "sigk": sigk, "sige": sige, "log_lik": log_lik}
        return results

    # -----------------
    # 3. INITIALIZATION
    # -----------------
    @staticmethod
    def initialize_scmem(freqs, psd_real):
        """
        Generate the initial estimates for Pink Noise, White Noise, and peaks.

        Parameters
        ----------
        freqs    : np.array (N,) -> Frequencies in Hz.
        psd_real : np.array (N,) -> Raw periodogram (Welch's method in linear scale).

        Returns
        -------
        sigk_ini  : (N x K) matrix with Aperiodic (0) + Periodic (1:K) components by columns.
        sige_ini  : np.array (N,) -> constant white noise.
        peaks_idx : np.array (N,) -> Frequencies at which peaks were detected.
        """

        # 1) Basal noise (E): The floor of the signal
        # In the PSD, this noise is represented as a constant baseline and it
        # is strictly positive across frequencies.
        # -> [This part could be improved]
        sige_ini = np.min(psd_real) * 0.8

        # 2) Pink noise (Xi): Robust Log-Linear initialization 
        # The starting and ending PSD points are connected to ensure 
        # a pure 1/f base (under the real PSD), so it does not "eat" the peaks.
        
        # 2.1. Find the first maximum peak
        peak_mask = (freqs >= 0.1) & (freqs <= 10) 
        idx_max   = np.argmax(psd_real[peak_mask])
        f_peak    = freqs[peak_mask][idx_max]
        p_peak    = psd_real[peak_mask][idx_max]

        # 2.2. Ascending trend: Isotonic Regression (from 0 to f_peak)
        mask_asc = (freqs >= 0) & (freqs <= f_peak)
        # the curve can only go upwards until the first peak
        iso_asc          = IsotonicRegression(increasing=True, out_of_bounds='clip')
        p_asc_segment    = iso_asc.fit_transform(freqs[mask_asc], psd_real[mask_asc])

        # 2.3. Descending trend: Log-Linear (from f_peak to 30Hz)
        mask_desc = (freqs > f_peak)
        f_last    = freqs[mask_desc][-1]
        p_last    = psd_real[mask_desc][-1]

        # 2.4. Compute the slope in the Log-Log space
        log_f_coords = np.log10([f_peak, f_last])
        log_p_coords = np.log10([p_peak, p_last])

        slope_asc_hib     = (log_p_coords[1] - log_p_coords[0]) / (log_f_coords[1] - log_f_coords[0])
        intercept_asc_hib = log_p_coords[0] - slope_asc_hib * log_f_coords[0]

        # Build the descending trend
        p_desc_segment = 10 ** (intercept_asc_hib + slope_asc_hib * np.log10(freqs[mask_desc]))

        # 2.5. Concatenate both segments
        pink_noise_ini = np.concatenate([p_asc_segment, p_desc_segment])

        # 3) Find the periodic components (Pi) in the residuals. Remove the initial 
        # estimate of Xi and the basal noise. The residual is smoothed to facilitate 
        # the search for possible periodic peaks.
        residual        = (psd_real - sige_ini) - pink_noise_ini
        smooth_residual = gaussian_filter1d(residual, sigma=1.0)
        
        # Change ('prominence') if it detects many false peaks or ignores the real ones.
        # The peak must stand out at least 10% with respect to the maximum peak.
        # Peaks must be separated at least N points ('distance').
        # This last setting is highly influenced by the frequency resolution.

        # -> [Be careful about these settings]
        min_prominence = np.max(smooth_residual) * 0.1
        peaks_idx, _   = find_peaks(smooth_residual, prominence=min_prominence, distance=5)

        # 4) Build initial matrix (sigk_ini)
        n_peaks  = len(peaks_idx)
        sigk_ini = np.zeros((len(freqs), 1 + n_peaks))
        
        # Column 0 is the pink noise
        sigk_ini[:, 0] = pink_noise_ini

        # Compute the real width at half height
        # This returns the FWHM in terms of sample size
        sample_widths, _, _, _ = peak_widths(smooth_residual, peaks_idx, rel_height=0.5)

        # Center a Gaussian on each detected peak
        for i, p_idx in enumerate(peaks_idx):
            center_hz = freqs[p_idx]
            A         = smooth_residual[p_idx]  # Amplitude 
            # Compute sigma (Full Width at Half Maximum method)
            width_hz  = sample_widths[i] * (freqs[1] - freqs[0])
            sigma     = width_hz / 2.355
            Kh        = A * np.exp(-0.5 * ((freqs - center_hz) / sigma) ** 2)
            sigk_ini[:, i + 1] = np.clip(Kh, 1e-10, None)
            
        return sigk_ini, sige_ini, peaks_idx
