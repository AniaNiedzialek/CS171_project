# West Coast Swing Dance Classification

## CS 171 Final Project

---

## Research Question

**Can we classify West Coast Swing dance performances by skill division and dance pattern using pose estimation keypoints extracted from video?**

Specifically, we investigate:
1. **Division Classification**: Can an LSTM model learn temporal movement patterns that distinguish dancers across skill levels (Novice → Champion)?
2. **Pattern Classification**: Can a Decision Tree identify which body part movements (e.g., wrist position variability) differentiate Sugar Push from Sugar Tag patterns?

---

## Authors & Project Roles

| Author | Role | Responsibilities |
|--------|------|------------------|
| **Nguyen Pham** | Decision Tree Model Development | Data preprocessing (`01a_data_process.ipynb`), Decision Tree training (`02a_train.ipynb`), feature engineering, pattern classification |
| **Ania Niedzialek** | LSTM Model Development | Data preprocessing (`01b_data_process.ipynb`), LSTM training (`02b_train.ipynb`), temporal sequence modeling, division classification |
| **Both (Collaborative)** | Analysis & Visualization | `03_visualization_analysis.ipynb` - model evaluation, temporal importance analysis, feature importance visualization |

---

## Key Findings

### Decision Tree - Pattern Classification (Sugar Push vs Sugar Tag)
- **Top Feature**: `std(right_wrist_Y)` accounts for nearly 100% of classification importance
- **Interpretation**: The variability in vertical wrist movement is the primary distinguishing factor between Sugar Push and Sugar Tag patterns
- **Why it makes sense**: Sugar Push involves pushing motions while Sugar Tag involves tagging/reaching motions - both heavily involve wrist positioning

### LSTM - Division Classification
- **Temporal Importance**: Middle frames (around frame 15-20 of 32) show highest importance for division classification
- **Interpretation**: The core execution phase of dance patterns contains the most skill-differentiating information
- **Finding**: Higher-level dancers show more consistent temporal patterns compared to novice dancers

---

## Installation Instructions


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

## Data Access Statement

### Data Sources
- **Training Data**: Public West Coast Swing competition videos from YouTube
- **Extraction Method**: Videos downloaded using `yt-dlp`, trimmed with `ffmpeg`
- **Pose Estimation**: MediaPipe Pose extracts 33 body landmarks with (X, Y, Z, visibility) per frame

---

## Notebooks Overview

### Data Preprocessing 

| Notebook | Author | Description |
|----------|--------|-------------|
| `01a_data_process.ipynb` | Nguyen Pham | Downloads videos, extracts MediaPipe keypoints, engineers statistical features (angles, velocities, pairwise distances) for Decision Tree |
| `01b_data_process.ipynb` | Ania Niedzialek | Downloads videos, extracts temporal keypoint sequences, standardizes to 32 frames for LSTM |

### Model Construction 

| Notebook | Author | Description |
|----------|--------|-------------|
| `02a_train.ipynb` | Nguyen Pham | Decision Tree classifier for pattern classification (Sugar Push vs Sugar Tag). Features: aggregated mean/std of keypoints, angles, distances, velocities |
| `02b_train.ipynb` | Ania Niedzialek | LSTM neural network for division classification. Architecture: 2-layer LSTM (128 units), cross-validation, early stopping |

### Analysis & Visualization

| Notebook | Authors | Description |
|----------|---------|-------------|
| `03_visualization_analysis.ipynb` | Both | Confusion matrices, temporal importance analysis (LSTM), feature importance with body part names (Decision Tree), performance comparison |

---

## Model Architectures

### Decision Tree (Pattern Classification)
- **Task**: Classify Sugar Push vs Sugar Tag patterns
- **Input**: 460 features (mean + std of: 99 keypoint coords, 4 joint angles, 28 pairwise distances, 99 velocities)
- **Output**: Binary classification (sugar_push / sugar_tag)
- **Key Finding**: Right wrist Y-position variability is the dominant feature

### LSTM (Division Classification)
- **Task**: Classify dancer skill level (5 divisions)
- **Input**: 32 frames × 99 features (33 keypoints × 3 coordinates)
- **Architecture**: 2-layer LSTM, 128 hidden units, dropout 0.3
- **Output**: 5 classes (novice, intermediate, advanced, allstar, champion)
- **Training**: 4-fold stratified cross-validation, learning rate scheduling

## Ethics

* Use only public or self‑recorded videos.

## License

MIT License — open for academic and research use.
