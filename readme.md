# Interturn Fault Detection in BLDC Motor using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

A data-driven fault diagnosis system for **Permanent Magnet Brushless DC (PMBLDC) motors**, combining a custom hardware test bench with a deep learning classifier to detect and grade the severity of **stator inter-turn short-circuit faults** using only raw three-phase current waveforms — with **no hand-crafted feature engineering** anywhere in the pipeline.

> Submitted as part of the Summer Internship Program (SIP) 2026, Department of Electrical Engineering, National Institute of Technology Calicut.


---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Hardware Test Bench](#-hardware-test-bench)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Ablation & Comparative Study](#-ablation--comparative-study)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Future Scope](#-future-scope)
- [References](#-references)
- [Authors](#-authors)
- [License](#-license)

---

## 🔍 Overview

Inter-turn short-circuit faults in stator windings are one of the most common and progressively damaging failure modes in PMBLDC motors. Left undetected, they generate localized heating, accelerate insulation breakdown, and can escalate into phase-to-phase or phase-to-ground failures.

This project builds a full pipeline — from **controlled fault induction hardware** to a **purpose-built dual-branch deep learning model** — for reliable, single-sensor, severity-graded diagnosis of these faults:

- 🔧 A PMBLDC motor modified with tapped windings to induce inter-turn shorts of **2–6 turns**
- ⚡ Three-phase stator current capture under **2 loading conditions** and **4 frequency ranges (20–100 Hz)**
- 🧠 A **dual-branch fusion model**: Temporal Convolutional Network (TCN) with Squeeze-and-Excitation + Temporal Attention, fused with an autoencoder-compressed global waveform summary
- 📊 A structured **ablation study** and **comparative study** against 4 baseline models and published literature

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **93.07%** |
| **Macro F1-Score** | **0.900** |
| No-Fault F1 | 0.99 |
| Dataset Size | 1,501 samples (4 classes) |
| Inference Latency | ~3.5 ms/sample |

The proposed model outperforms all evaluated baselines (ANN, KNN, Neuro-Fuzzy, Attention-LSTM) and comparable published methods on BLDC fault diagnosis.

<!-- 📌 SUGGESTED: Add Figure 5.5 (ablation accuracy bar chart) here -->
<!-- ![Ablation Study Results](assets/ablation_accuracy_chart.png) -->

---

## 🛠 Hardware Test Bench

| Component | Detail |
|---|---|
| Motor | PMBLDC, C16-NY+, 48 V, 10 A, 1000 W, 4-pole, 3-phase |
| Fault Induction | Tapped windings + physical switches (2–6 point shorts, all combinations) |
| Loading | Prony-brake / spring-balance rig (No-load, Half-load) |
| Drive | NY+ Intelligent Controller (Hall-sensor feedback, twist-grip throttle) |
| Instrumentation | 3× current probes (10 mV/A) → Measuring device → CSV export |

**Signal flow:** Power Supply → Controller → Motor (tapped windings) → Current Probes → Measuring Device → Laptop (CSV logging)

<!-- 📌 SUGGESTED IMAGES to add under an /assets folder:
- Fig 2.1: PMBLDC motor nameplate
- Fig 2.2 / 2.3: Terminal board tap points
- Fig 2.4: Prony-brake loading rig
- Fig 2.8: Complete hardware setup (labeled)
- Fig 2.9: Block diagram
-->
![Block Diagram](assets/block_diagram.png)
![Complete Setup](assets/complete_setup.jpg)

---

## 📊 Dataset

Fault conditions were grouped into **four severity classes**:

| Class | Label | Description |
|---|---|---|
| 0 | No Fault | Healthy motor |
| 1 | Low | 2-point inter-turn short |
| 2 | Medium | 3-point and 4-point shorts |
| 3 | High | 5-point and 6-point shorts |

| | Collected | Retained (final) |
|---|---|---|
| No Fault | 240 | 240 |
| Low | 360 | 358 |
| Medium | 840 | 738 |
| High | 168 | 165 |
| **Total** | **1608** | **1501** |

Each capture: **1,999 samples × 3 channels** (one per phase), 80/20 stratified train/test split.

> ⚠️ **Data-quality note:** The No-Fault class was originally recorded with an inconsistent timebase and too few samples (24). Re-collection under consistent acquisition settings raised the No-Fault F1-score from 0.00 → ~0.99.

<!-- 📌 SUGGESTED: Add representative waveform plots -->
<!-- ![Sample Waveforms](assets/waveform_panel.png) — combine Figs 5.1–5.4 (No Fault / Low / Medium / High) -->

---

## 🧠 Model Architecture

The proposed model — **Autoencoder + TCN (SE + Temporal Attention)** — processes the **same raw waveform** through two complementary branches, with no hand-crafted statistical features anywhere in the pipeline:

1. **Branch 1 — Autoencoder (global summary):**
   Flattened waveform (5997-D) → `5997 → 256 → 64 → 24` (unsupervised pretraining, frozen) → `24 → 16 → 8` (supervised refinement)

2. **Branch 2 — Attention-TCN (local/temporal features):**
   Raw waveform (1999×3) → 3× Residual Blocks (dilated TCN + Squeeze-and-Excitation + Temporal Attention) → Conv1D refinement → GAP → 64-D feature vector

3. **Fusion Head:**
   `[64-D TCN ; 8-D AE] → Dense(128, ReLU) → Dropout(0.4) → Dense(4, Softmax)`

<!-- 📌 SUGGESTED: Add Figure 4.4 (full architecture diagram) — this is the most important image for the README -->
![Model Architecture](assets/model_architecture.png)

| Stage | Component | Parameters | Trainable |
|---|---|---|---|
| 1 | Autoencoder (5997↔24) | 3,112,965 | ❌ (frozen) |
| 2 | Latent refinement (24→16→8) | 536 | ✅ |
| 3 | Attention-TCN branch | 800,287 | ✅ |
| 4 | Fusion head | 9,860 | ✅ |
| **Total trainable** | | **810,683** | |

---

## 🔬 Ablation & Comparative Study

### Ablation Ladder (test accuracy)

| # | Model | Accuracy | Macro F1 |
|---|---|---|---|
| A | Vanilla CNN | 78.22% | 0.748 |
| B | Vanilla TCN | 88.78% | 0.847 |
| C | AE + CNN | 81.85% | 0.735 |
| D | AE + Vanilla TCN | 88.78% | 0.842 |
| **E** | **Proposed (AE + TCN, SE+TA)** | **93.07%** | **0.900** |

### Comparative Study (vs. classical/baseline paradigms)

| Model | Accuracy | F1-Score |
|---|---|---|
| Neuro-Fuzzy Network | 52.81% | 50.32 |
| ANN (ReLU) | 73.60% | 70.95 |
| KNN (k=3) | 83.50% | 76.93 |
| Attention-LSTM | 87.79% | 84.44 |
| **Proposed** | **93.07%** | **91.35** |

<!-- 📌 SUGGESTED: Add confusion matrix and training curve images -->
<!-- ![Confusion Matrix](assets/confusion_matrix.png) -->
<!-- ![Training Curves](assets/training_curves.png) -->

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                # Raw CSV captures from measuring device
│   └── processed/          # Cleaned/aligned train-test splits
├── notebooks/               # Exploratory analysis & waveform plots
├── src/
│   ├── data_preprocessing.py
│   ├── autoencoder.py
│   ├── models/
│   │   ├── ann.py
│   │   ├── knn.py
│   │   ├── neuro_fuzzy.py
│   │   ├── alstm.py
│   │   └── ae_tcn_se_ta.py   # Proposed model
│   ├── train.py
│   └── evaluate.py
├── assets/                  # Images used in this README
├── results/                 # Confusion matrices, plots, logs
├── requirements.txt
└── README.md
```

> Update this section to match your actual repo layout.

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/interturn-fault-detection-bldc.git
cd interturn-fault-detection-bldc
pip install -r requirements.txt
```

**Core dependencies:** `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`

---

## ▶️ Usage

```bash
# Preprocess raw CSV captures into train/test splits
python src/data_preprocessing.py --input data/raw --output data/processed

# Pretrain the autoencoder branch
python src/autoencoder.py --epochs 250

# Train the proposed model
python src/train.py --model ae_tcn_se_ta --epochs 100

# Evaluate on the held-out test split
python src/evaluate.py --model ae_tcn_se_ta --checkpoint results/best_model.h5
```

> Update commands/flags to match your actual scripts.

---

## 📈 Results

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Fault | 1.00 | 0.98 | 0.99 | 48 |
| Low | 0.93 | 0.95 | 0.94 | 74 |
| Medium | 0.93 | 0.97 | 0.95 | 148 |
| High | 0.79 | 0.67 | 0.72 | 33 |
| **Accuracy** | | | **0.9307** | 303 |

High-severity faults remain the hardest class to classify, primarily due to limited training samples (165) and waveform overlap with extreme-Medium-severity captures.

---

## 🔮 Future Scope

- Automated relay/microcontroller-based fault switching
- Additional loading points + direct torque measurement
- Continuous frequency sweep instead of 4 discrete bands
- Multi-modal sensing (vibration, temperature, acoustic)
- Real-time embedded deployment (Raspberry Pi / Jetson Nano)
- 5-fold cross-validation + Monte Carlo Dropout uncertainty estimates
- Transfer learning from large public motor-fault datasets

---

## 📚 References

1. Awadallah et al., "A Neuro-Fuzzy Approach to Automatic Diagnosis and Location of Stator Inter-Turn Faults in CSI-Fed PM Brushless DC Motors," *IEEE Trans. Energy Conversion*, 2005.
2. Duraisamy et al., "A Quick and Effective Current-Based Approach for BLDC Motor Interturn Fault Identification Using ANN," *JESA*, 2025.
3. Sindapati et al., "Attention-Based LSTM for RUL Estimation in Electric BLDC Motors with Stator Inter-Turn Faults," *ICA*, 2025.
4. Khan et al., "Securing Electric Vehicle Performance: ML-Driven Fault Detection and Classification," *IEEE Access*, 2024.
5. Kim, Lee, Hur, "Diagnosis Technique Using a Detection Coil in BLDC Motors With Interturn Faults," *IEEE Trans. Magnetics*, 2014.
6. Lee, Kim, Hur, "Diagnosis Technique for Stator Winding Inter-Turn Fault in BLDC Motor Using Detection Coil," *ICPE-ECCE Asia*, 2015.

---

## 👥 Authors

- **M S Samiksha** (SIP260463)
- **Mohd Luqmaan** (SIP262126)

**Guide:** Dr. Shihabudheen K V, Dept. of Electrical Engineering, NIT Calicut

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
