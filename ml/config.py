"""
Single source of truth for all constants.
Both ml/ Python code and arduino/ C headers derive from this file.
Never hand-edit generated headers; run ml/export_c.py instead.
"""

# Sampling / window constants
SR             = 9600       # Uno analogRead rate (Hz); update & retrain if probe differs >3%
WIN            = 256        # samples per window
WINS_PER_CHUNK = 37         # windows per 1-second classification chunk  (~9600/256 ≈ 37.5)

# ADC simulation
ADC_BITS       = 10         # Uno is 10-bit
ADC_MID        = 512        # centre value

# Feature extraction (integer, see siren-feature-parity skill)
HYST           = 6          # zero-crossing hysteresis in ADC counts
DC_SHIFT       = 6          # EMA shift: dc16 >>= DC_SHIFT  (alpha ≈ 1/64)
LEVEL_GATE     = 40         # lvl < LEVEL_GATE → classify as NOISE without running tree

# Class labels (must match CLASSES order everywhere)
CLASSES        = ["traffic", "ambulance", "firetruck"]   # label 0, 1, 2
EXCLUDE_CLASSES= []         # e.g. ["police"] — excluded during data prep

# Training
RANDOM_SEED    = 42
MAX_DEPTH_GRID = [3, 4, 5]          # tree depths to try
MIN_SAMPLES_GRID = [2, 4, 8]       # min_samples_leaf to try
MAX_LEAVES     = 16                 # Uno size limit

# Augmentation (applied to ADC-converted samples during training)
GAIN_LOW       = 0.03       # log-uniform gain range
GAIN_HIGH      = 0.90
ADC_NOISE_LOW  = 1          # Gaussian noise sigma (ADC counts)
ADC_NOISE_HIGH = 3
N_AUG          = 3          # augmentation copies per training chunk
SNR_MIX_DB    = [20, 10, 5, 0]    # SNR levels for traffic-noise mixing at evaluation

# Paths (use pathlib in code, never hardcode elsewhere)
DATA_RAW       = "data/raw"
DATA_PROCESSED = "data/processed"
MANIFEST_CSV   = "data/processed/manifest.csv"
MODELS_DIR     = "ml/models"
TREE_JSON      = "ml/models/siren_tree.json"
METRICS_JSON   = "ml/models/metrics.json"
DOCS_DIR       = "docs"
FIGS_DIR       = "docs/figures"
ARDUINO_DIR    = "arduino/signal_controller"
PROBE_DIR      = "arduino/siren_probe"
TESTS_DIR      = "tests"
