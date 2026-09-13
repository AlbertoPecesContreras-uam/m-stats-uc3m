function out = applyPassBand(signal, fs, f_cutoff)
% Apply a zero-phase Butterworth bandpass filter (2nd order per stage,
% 4th order effective due to filtfilt).
%
%   Syntax:
%       out = applyPassBand(signal, fs, {f_low, f_high})
%
%   Input:
%       signal   : Data matrix [n_channels x n_samples].
%       fs       : Sampling frequency (Hz).
%       f_cutoff : Cell array with cut-off frequencies {f_low, f_high}.
%
%   Output:
%       out      : Filtered signal in the same format [n_channels x n_samples].

    f_low  = f_cutoff{1};  % e.g. 0.1 Hz
    f_high = f_cutoff{2};  % e.g. 30 Hz

    % --- 0. Orientation detection ---
    % filtfilt operates column-wise, so signal must be [samples x channels].
    % Assumption: the time dimension is always the longer one.
    [rows, cols]    = size(signal);
    needs_transpose = cols > rows;
    if needs_transpose
        signal = signal.';  % non-conjugate transpose
    end
    % signal is now [samples x channels]

    % --- 1. High-pass filter ---
    if f_low ~= 0
        Wn_high           = f_low / (fs / 2);  % normalised cut-off
        [b_high, a_high]  = butter(2, Wn_high, 'high');
        signal_hp         = filtfilt(b_high, a_high, signal);
    else
        signal_hp = signal;
    end

    % --- 2. Low-pass filter ---
    Wn_low          = f_high / (fs / 2);
    [b_low, a_low]  = butter(2, Wn_low, 'low');
    signal_lp       = filtfilt(b_low, a_low, signal_hp);

    % --- 3. Restore original orientation ---
    if needs_transpose
        out = signal_lp.';
    else
        out = signal_lp;
    end
end
