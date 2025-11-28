# West Coast Swing Dance Analysis

## Project Title

**Analyzing West Coast Swing Patterns Using Video Classification**

## Authors

* Ania Niedzialek
* Nguyen Pham

## Research Topic

This project applies machine learning and computer vision to analyze movement patterns in West Coast Swing (WCS) dance videos. We focus on detecting and comparing specific patterns—such as Sugar Push and Sugar Tag—across competition divisions (Newcomer, Intermediate, Advanced, All-Star, Champion). Our goals include identifying stylistic differences across experience levels and exploring whether motion features can support data‑driven feedback for dancers.

---

## Project Outline

### 1. Data Preparation

* Collect short, well‑defined video clips of Sugar Push and Sugar Tag patterns.
* Extract pose keypoints using MediaPipe for temporal analysis.

### 2. Feature Extraction

* Extract 33 pose keypoints with (X, Y, Z, confidence).
* Standardize sequences to 32 frames.

### 3. Model Training

* LSTM neural network for classifying dance divisions.
* **Input:** 32 frames × (33 keypoints × 3 coordinates).
* **Output:** 5 division classes.
* Use data augmentation, cross‑validation, and regularization.

### 4. Evaluation

* Stratified cross‑validation.
* Temporal importance analysis.
* Sequence‑level attention visualization.

### 5. Deliverables

* Final report
* Trained LSTM model
* Visualization notebook for movement analysis

---

## Data Collection Plan

### Sources

* Public WCS competition footage on YouTube.

### Preprocessing

* Download using `yt-dlp`.
* Trim clips to 3–6 seconds using `ffmpeg`.
* Extract MediaPipe pose keypoints.
* Save standardized `.npy` sequences.

### Ethics

* Use only public or self‑recorded videos.

---

## Model Architecture

### LSTM Temporal Model

* **Input:** 99 features per frame (33 keypoints × 3 coordinates)
* **LSTM:** 2 layers × 128 units, dropout 0.3
* **Output:** 5 division classes
* **Training:** LR scheduling, early stopping, augmentation
* **Performance:** ~35% CV accuracy

---

## Project Timeline

| Week  | Milestone              | Description                                                       |
| ----- | ---------------------- | ----------------------------------------------------------------- |
| 10/27 | Topic Approval & Setup | Finalize research question, confirm tools, gather initial videos. |
| 11/03 | Data Collection        | Download and trim clips, build labels.csv.                        |
| 11/10 | Dataset Prep           | Extract 3D keypoints, standardize sequences, augment data.        |
| 11/17 | Model Training         | Train LSTM with cross‑validation.                                 |
| 11/24 | Evaluation & Report    | Visualizations, temporal analysis, final updates.                 |
| 12/01 | Presentation           | Presentation and project prerecorded demo.                        |

------|------------|-------------|
| 10/13 | Topic Approval & Setup | Finalize research question, confirm tools, gather initial videos. |
| 10/20 | Data Collection | Download and trim clips, build labels.csv. |
| 10/27 | Dataset Prep | Extract 3D keypoints, standardize sequences, augment data. |
| 11/03 | Model Training | Train LSTM with cross‑validation. |
| 11/10 | Evaluation & Report | Visualizations, temporal analysis, final report. |

---

## Current Updates

* Collected videos for 5 divisions (4 per class)
* Extracted 3D MediaPipe keypoints
* Implemented LSTM classification model
* Achieved **35% CV accuracy**
* Added temporal importance visualizations

---

# Notebooks Overview

## Notebook 1 — `01_data_collection.ipynb`

**Focus:** Build and organize the dataset.

* Download YouTube clips, trim with time ranges.
* Maintain `labels.csv` (division, pattern, timestamps).
* Extract frames + MediaPipe keypoints.
* Organize videos into structured folders.

## Notebook 2 — `02_train.ipynb`

**Focus:** Train the LSTM division classifier.

* Load 3D keypoints (X, Y, Z).
* Apply augmentation + fixed 32-frame sequences.
* Train 2-layer LSTM using 4-fold stratified cross-validation.
* Save final model with metadata.

## Notebook 3 — `03_analyze_keypoints.ipynb`

**Focus:** Traditional ML + engineered motion features.

* Clean & normalize sequences.
* Compute angles, distances, and velocity features.
* PCA/t-SNE visualization of Sugar Push vs Sugar Tag.
* Train a Decision Tree classifier (near-perfect accuracy).
* Save deployable scikit-learn pipeline.

## Notebook 4 — `04_lstm_analysis.ipynb`

**Focus:** Interpret and evaluate the LSTM model.

* Predict divisions on new sequences.
* Generate confusion matrix + misclassification analysis.
* Compute **temporal importance curves** from LSTM hidden states.
* Visualize motion trajectories (XY & Z).
* Assess per-division accuracy and confidence.

---


# .gitignore

```
venv
.DS_Store
*.mp4
data/raw/videos/
.env
notebooks/.ipynb_checkpoints/
__pycache__/
models/*
```

# License

MIT License — open for academic and research use.
