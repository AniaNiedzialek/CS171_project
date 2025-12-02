# West Coast Swing Dance Analysis

## Project Title

**Analyzing West Coast Swing Patterns Using Video Classification**

## Authors & Project Roles

| Author | Role | Notebooks |
|--------|------|-----------|
| **Nguyen Pham** | Decision Tree Model Development | `01a_data_process.ipynb`, `02a_train.ipynb` |
| **Ania Niedzialek** | LSTM Model Development | `01b_data_process.ipynb`, `02b_train.ipynb` |
| **Both** | Analysis & Visualization | `03_visualization_analysis.ipynb` |

## Research Topic

This project applies machine learning and computer vision to analyze movement patterns in West Coast Swing (WCS) dance videos. We focus on detecting and comparing specific patterns—such as Sugar Push and Sugar Tag—across competition divisions (Newcomer, Intermediate, Advanced, All-Star, Champion). Our goals include identifying stylistic differences across experience levels and exploring whether motion features can support data‑driven feedback for dancers.

---

## Installation Instructions

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/AniaNiedzialek/CS171_project.git
cd CS171_project

# Create virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

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
* Implemented LSTM classification model (Ania)
* Implemented Decision Tree classifier (Nguyen)
* Added temporal importance visualizations

---

## Notebooks Overview

### Data Preprocessing (2 notebooks)

| Notebook | Author | Description |
|----------|--------|-------------|
| `01a_data_process.ipynb` | Nguyen Pham | Data preprocessing for Decision Tree model - downloads videos, extracts keypoints, engineers features |
| `01b_data_process.ipynb` | Ania Niedzialek | Data preprocessing for LSTM model - downloads videos, extracts temporal keypoint sequences |

### Model Construction (2 notebooks)

| Notebook | Author | Description |
|----------|--------|-------------|
| `02a_train.ipynb` | Nguyen Pham | Decision Tree training - feature engineering, model training, evaluation |
| `02b_train.ipynb` | Ania Niedzialek | LSTM training - sequence modeling, cross-validation, temporal classification |

### Analysis & Visualization (1 collaborative notebook)

| Notebook | Authors | Description |
|----------|---------|-------------|
| `03_visualization_analysis.ipynb` | Both | Model analysis, confusion matrices, temporal importance, model comparison |

---
## .gitignore

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
