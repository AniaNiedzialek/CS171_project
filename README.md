# West Coast Swing Dance Analysis

## Project Title
**Analyzing West Coast Swing Patterns Using Video Classification**

## Authors
- Ania Niedzialek
- Nguyen Pham

## Research Topic
This project focuses on applying machine learning and computer vision to analyze movement patterns in West Coast Swing (WCS) dance videos. The goal is to detect and compare specific dance patterns—such as Sugar Push and Sugar Tag—across different competition divisions (e.g., Newcomer, Intermediate, Advanced, All-Star, Champion). We aim to identify stylistic differences and explore whether measurable motion features correlate with dancer experience level, ultimately suggesting data-driven feedback for improvement.

## Project Outline
1. **Data Preparation**
   - Collect short, clearly defined video clips of Sugar Push and Sugar Tag patterns from publicly available YouTube footage or self-recorded examples.
   - Extract pose keypoints using MediaPipe for temporal analysis.
2. **Feature Extraction**
   - Extract 3D pose keypoints (X, Y, Z coordinates + confidence) from video frames using MediaPipe.
   - Process keypoints into sequences for temporal pattern analysis.
3. **Model Training**
   - Use LSTM neural network to classify dance divisions based on temporal keypoint sequences:
     - Input: 32-frame sequences of 33 keypoints × 3 coordinates (99 features per frame)
     - Output: 5 division classes (advanced, allstar, champion, intermediate, novice)
   - Implement data augmentation, cross-validation, and regularization for robust training.
4. **Evaluation**
   - Evaluate using stratified cross-validation with balanced class distributions.
   - Analyze temporal importance patterns and division-specific movement characteristics.
   - Visualize model attention and sequence-level predictions.
5. **Deliverables**
   - Final report with classification accuracy analysis
   - Trained LSTM model with temporal analysis capabilities
   - Comprehensive visualization notebook with temporal importance analysis

## Data Collection Plan
- **Sources:** Publicly available WCS competition clips on YouTube, with credit and citation.  
- **Preprocessing:**
  - Download video segments using `yt-dlp`
  - Trim each clip to 3–6 seconds around the pattern using `ffmpeg`
  - Extract pose keypoints using MediaPipe (33 keypoints × 4 dimensions: X, Y, Z, confidence)
  - Standardize sequences to 32 frames with data augmentation
- **Ethics:**  
  - Use videos that are publicly available or self-recorded.
  - Avoid private or monetized content.

---
## Model Architecture
### Final Model — Keypoints + LSTM (Temporal Analysis)
- **Architecture:**  
  - **Input Layer**: 99 features per frame (33 keypoints × 3 coordinates)
  - **LSTM Layers**: 2-layer LSTM with 128 hidden units, dropout=0.3
  - **Classification Head**: Linear layer → 5 division classes
  - **Regularization**: Data augmentation, weight decay, early stopping
- **Training Strategy:**
  - 4-fold stratified cross-validation (balanced class distribution)
  - Learning rate scheduling with ReduceLROnPlateau
  - Data augmentation with random noise injection
- **Performance:** 35% cross-validation accuracy
## Project Timeline
| Week | Milestone | Description |
|------|------------|-------------|
| 10/13 | Topic Approval & Setup | Finalize the research question, confirm tools (PyTorch, MediaPipe, ffmpeg), and collect initial reference videos for WCS patterns. |
| 10/20 | Data Collection & Labeling | Download and trim selected WCS clips using `yt-dlp` and `ffmpeg`. Extract pose keypoints and create labels.csv with division information. |
| 10/27 | Keypoint Extraction & Dataset Preparation | Process videos to extract 3D pose sequences, standardize to 32-frame sequences, and implement data augmentation. |
| 11/03 | Model Development & Training | Implement LSTM architecture for temporal keypoint analysis, train with cross-validation and regularization techniques. |
| 11/10 | Evaluation & Report | Generate temporal analysis results, visualize importance patterns, and write the final report + presentation slides. |
---
Updates:
- Collected dance videos for 5 divisions (4 samples each)
- Extracted 3D pose keypoints using MediaPipe (33 keypoints × 4 dimensions)
- Implemented LSTM model with temporal sequence analysis
- Achieved 35% cross-validation accuracy
- Developed comprehensive temporal importance visualization

## Notebooks Description
#### Notebook 1 - `01_data_collection.ipynb`
**Purpose:** Collects raw dance clips from YouTube and organizes them for modeling.

The notebook:
- Extracts YouTube video IDs from different URL formats
- Downloads trimmed video segments
- Creates and updates `labels.csv` with division, pattern, start time, duration, metadata
- Organizes dataset folders by division
- Batch-downloads all labeled clips

#### Notebook 2 - `02_train.ipynb`
**Purpose:** Train LSTM model on 3D pose keypoints for division classification.

The notebook:
- Loads 3D pose keypoints from `.npy` files (33 keypoints × 4 dimensions)
- Implements `KeypointsDataset` class with data augmentation
- Builds LSTM architecture for temporal sequence analysis
- Uses 4-fold stratified cross-validation with balanced class distribution
- Applies advanced training techniques (LR scheduling, early stopping, regularization)
- Achieves 35% cross-validation accuracy
- Saves final model with temporal analysis capabilities

#### Notebook 3 - `03_analyze_keypoints.ipynb`
**Purpose:** Analyze dancer movements using pose/keypoint data with statistical features.

The notebook:
- Loads keypoint `.npy` files from each video
- Cleans missing keypoints via interpolation
- Calculates engineered features (angles, distances, velocities)
- Aggregates features for traditional ML classification
- Performs PCA and t-SNE visualization of movement patterns
- Implements Decision Tree classifier for comparison with LSTM approach
- Achieves 100% accuracy on keypoints_2 data for sugar_push vs sugar_tag classification

#### Notebook 4 - `04_lstm_analysis.ipynb`
**Purpose:** Analyze the trained LSTM model's temporal behavior and predictions.

This notebook:
- Loads trained LSTM model with 3D keypoint processing
- Makes predictions on test sequences with temporal preprocessing using keypoints_2 data
- Generates confusion matrix and division-level performance metrics
- Performs temporal importance analysis (which time steps matter most)
- Visualizes keypoint trajectories and Z-axis movement patterns
- Provides comprehensive sequence-level interpretation and analysis

## .gitignore and License
**.gitignore**
venv
.DS_Store
*.mp4
data/raw/videos/
.env
notebooks/.ipynb_checkpoints/
__pycache__/
models/*


**License:** MIT License — open for academic and research use only.
