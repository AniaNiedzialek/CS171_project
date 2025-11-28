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
| 10/13 | Topic Approval & Setup | Finalize research question, confirm tools, gather initial videos. |
| 10/20 | Data Collection        | Download and trim clips, build labels.csv.                        |
| 10/27 | Dataset Prep           | Extract 3D keypoints, standardize sequences, augment data.        |
| 11/03 | Model Training         | Train LSTM with cross‑validation.                                 |
| 11/10 | Evaluation & Report    | Visualizations, temporal analysis, final report.                  |

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

**Purpose:** Automates downloading, labeling, organizing, and preprocessing raw video data.

### Main Components

#### **Imports**

Handles filesystem ops, YouTube downloads, URL parsing, and keypoint extraction helpers.

#### **Download Videos**

`download_video(url, name, output_path, start_time, duration)`

* Downloads specific segments using `yt-dlp`'s `download_ranges`.
* Saves clips to a structured dataset folder.

#### **Extract YouTube ID**

`extract_youtube_id(url)` handles:

* Standard YouTube links
* Shorts
* Embed formats
* youtu.be links

#### **Create Labels**

`add_label(...)`:

* Creates or appends rows to `labels.csv`.
* Fields include: youtube_id, start_time, duration, division, pattern, labels.

#### **Batch Download Function**

`download_all_from_csv(csv_path, folder)`:

* Iterates through CSV rows
* Creates division‑specific folders
* Downloads trimmed clips
* Supports multiple datasets (`videos_1`, `videos_2`)

#### **Frame & Keypoint Extraction**

After downloading:

* `extract_frames(...)` extracts RGB frames.
* `extract_keypoints(...)` extracts MediaPipe pose data.

---

## Notebook 2 — `02_train.ipynb`

Training of LSTM classifier using 3D pose keypoints.

---

## Notebook 3 — `03_analyze_keypoints.ipynb`

Statistical feature analysis, PCA/t‑SNE, traditional ML comparison.

---

## Notebook 4 — `04_lstm_analysis.ipynb`

Model interpretation: temporal importance, confusion matrix, trajectory plots.

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
