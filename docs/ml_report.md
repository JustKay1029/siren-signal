# Machine Learning Report: Siren-Aware Adaptive Traffic Signal

## 1. Problem Statement & Architecture
The goal of this project is to build an embedded acoustic emergency vehicle classifier running on an Arduino Uno (ATmega328P microcontroller with 2 KB SRAM, 32 KB flash, and no floating-point unit). The system continuously samples an analog microphone at ~9.6 kHz (`SR = 9600 Hz`), extracts integer-only pitch and loudness features over streaming windows, and classifies the sound into three categories:
1. `traffic` (road/background noise, label 0)
2. `ambulance` (wailing/yelping siren, label 1)
3. `firetruck` (hi-lo/steady siren, label 2)

When an emergency siren is recognized, the traffic signal controller adjusts its phase timers to grant green priority clearance.

## 2. Dataset & Quality Audit
- **Source**: DynamicSuperb Emergency Traffic Detection (ETD) dataset (Hugging Face).
- **Raw Audio**: 600 audio files downloaded to `data/raw/{traffic,ambulance,firetruck}/` (200 files per class).
- **Validation Audit**:
  - **591 valid files accepted** into `data/processed/manifest.csv` (197 traffic, 197 ambulance, 197 firetruck).
  - **9 problematic files excluded** (4 exact duplicates, 4 severely clipped, 1 near-silent) and fully documented in `docs/data_report.md`.
  - Data directory is strictly excluded from version control (`data/` in `.gitignore`).

## 3. Feature Extraction & Hardware Parity
- **Streaming Integer Routine**: Matches the ATmega328P C routine (`arduino/signal_controller/siren_features.h`) bit-for-bit:
  - Exponential moving average DC offset removal with Q4 fixed-point arithmetic (`dc16 >>= 6`).
  - Hysteresis zero-crossing counter (`HYST = 6` ADC counts) to track fundamental pitch without floating-point division.
  - Per-window aggregation over 37 windows (~1 second classification chunk at 9.6 kHz / 256 samples).
  - 4 core integer decision tree features: `pmin` (minimum pitch), `pmax` (maximum pitch), `pmean` (mean pitch), and `sweep` (mean adjacent window frequency delta in Hz/window).
- **Loudness Gating**: Quiet chunks (`lvl < LEVEL_GATE`) are assigned directly to traffic/noise without evaluating the decision tree, saving CPU cycles on the microcontroller.
- **Unit Testing**: Verified in `tests/test_features.py` (8 unit tests passed).

## 4. Model Training & Evaluation Metrics
The classifier was trained using scikit-learn's `DecisionTreeClassifier` with `class_weight="balanced"` and constrained to $\le 16$ leaves to fit Uno RAM and flash limits.

- **Split Strategy**: Grouped by source audio file (5-fold cross-validation) so no file's chunks appear in both train and test folds, preventing data leakage.
- **Dataset Size**: 6,606 total chunks extracted (6,169 training chunks with gain/noise augmentation, 437 clean held-out test chunks across 147 test files).
- **Optimal Hyperparameters**: `max_depth = 3`, `min_samples_leaf = 2`, random state = 42.
- **Tree Complexity**: **8 leaves** (15 nodes total), requiring < 100 bytes of flash.

### Script-Produced Held-Out Test Results (`ml/models/metrics.json`)
- **Held-Out Accuracy**: **80.32%** (351 / 437 correct)
- **Held-Out Macro-F1**: **0.7957** (CV Macro-F1: 0.7952)
- **Class Breakdown**:
  - `traffic`: Precision = **0.99**, Recall = 0.97, F1 = **0.98** (150 test chunks)
  - `ambulance`: Precision = 0.81, Recall = 0.55, F1 = 0.66 (140 test chunks)
  - `firetruck`: Precision = 0.66, Recall = 0.88, F1 = 0.75 (147 test chunks)
- **Confusion Matrix** (Held-Out Test Set):
  ```
                   Predicted
                   traffic   ambulance   firetruck
    Actual traffic     145           0           5
         ambulance       1          77          62
         firetruck       0          18         129
  ```
- **Robustness vs Noise (SNR)**:
  - 20 dB SNR: 70.4% accuracy
  - 10 dB SNR: 67.6% accuracy
  - 5 dB SNR: 65.7% accuracy
  - 0 dB SNR: 67.6% accuracy

## 5. Exported Arduino C Logic
The trained decision tree is exported into `arduino/signal_controller/siren_model.h` as nested integer conditional statements with zero floating-point operations, dynamic allocations, or recursion:
```c
static inline uint8_t siren_classify(int16_t pmin, int16_t pmax,
                                      int16_t pmean, int16_t sweep) {
  if (pmax <= 646) {
    if (pmin <= 9) {
      if (pmax <= 177) {
        return 0; /* traffic */
      } else {
        return 1; /* ambulance */
      }
    } else {
      if (pmax <= 552) {
        return 0; /* traffic */
      } else {
        return 0; /* traffic */
      }
    }
  } else {
    if (sweep <= 60) {
      if (pmax <= 1452) {
        return 1; /* ambulance */
      } else {
        return 1; /* ambulance */
      }
    } else {
      if (sweep <= 193) {
        return 2; /* firetruck */
      } else {
        return 1; /* ambulance */
      }
    }
  }
}
```
**Equivalence Verification**: Evaluated across 2,000 random integer feature vectors in `tests/test_export.py`; yielded **0 mismatches** between scikit-learn predictions and exported C logic.

## 6. Limitations & Known Weaknesses
1. **Ambulance vs Firetruck Confusion**: While traffic separation is near-perfect (99% precision, only 1 false positive on traffic), ambulance and firetruck sirens share overlapping pitch bands (62 ambulance chunks misclassified as firetruck; 18 vice versa). In practice, detecting either triggers emergency green clearance, minimizing traffic disruption impact.
2. **Harmonic & Polyphonic Distortion**: Zero-crossing pitch estimation assumes a single dominant frequency. Simultaneous engine roar or honking horns at high volume may distort zero-crossing counts.
3. **Sampling Rate Sensitivity**: `SR = 9600 Hz` assumes Uno `analogRead` timing. Actual clock drift must be verified using `arduino/siren_probe/siren_probe.ino`.
4. **Host Compilation Check**: `arduino-cli` was not installed on the host system; exact flash and SRAM usage must be confirmed via Arduino IDE/CLI during board upload.

## 7. Artifacts & Figures
- Confusion Matrix: `docs/figures/confusion_matrix.png`
- Accuracy vs. SNR: `docs/figures/accuracy_vs_snr.png`
- Feature Pitch Scatter: `docs/figures/pitch_scatter.png`
- Model Metadata: `ml/models/siren_tree.json`, `ml/models/metrics.json`
