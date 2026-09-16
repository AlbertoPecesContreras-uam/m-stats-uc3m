# =============================================================================
# config.py
# =============================================================================
# Shared constants used across the connectivity analysis pipeline.
# Import from here instead of redefining in each script.
# =============================================================================

# EEG anatomical region definitions
EEG_REGIONS = {
    "Frontal":   ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "Central":   ["C3", "C4", "Cz"],
    "Parietal":  ["P3", "P4", "P7", "P8", "Pz"],
    "Occipital": ["O1", "O2", "Oz"],
    "Temporal":  ["T7", "T8"],
}

REGION_ORDER = ["Frontal", "Central", "Parietal", "Occipital", "Temporal"]

# Subject IDs assigned to the Sham (placebo) arm
SHAM_IDS = frozenset({
     1,  4,  7,  9, 11, 13, 14, 15,
    17, 18, 19, 24, 25, 27, 28,
    31, 35, 40, 41, 42, 43, 44,
    47, 48, 49, 50, 52, 61, 63, 65, 105,
})

# Experimental design
CONDITIONS  = ["OA", "OC"]
COND_LABELS = {"OA": "Eyes Open (EO)", "OC": "Eyes Closed (EC)"}
GROUPS      = ["Control", "Sham", "Exp"]

# Signal parameters
FS       = 500.0   # Sampling frequency (Hz)
FREQ_MIN = 0.1     # Lower bound of filtered band (Hz)
FREQ_MAX = 30.0    # Upper bound of filtered band (Hz)
