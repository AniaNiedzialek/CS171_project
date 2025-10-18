# CS171_project

### Project Title
**Analyzing West Coast Swing Patterns Using Video Classification**
### Authors
-   Ania Niedzialek
-   Nguyen Pham
### Research Topic
This project focuses on applying machine learning and computer vision to analyze movement patterns in West Coast Swing (WCS) dance videos.  
The goal is to detect and compare specific dance patterns—such as Sugar Push and Sugar Tag—across different competition divisions (e.g., Newcomer, Intermediate, Advanced, All-Star, Champion).  
We aim to identify stylistic differences and explore whether measurable motion features correlate with dancer experience level, ultimately suggesting data-driven feedback for improvement.
---
### Project Outline/Plan
1. **Data Preparation**
   - Collect short, clearly defined video clips of Sugar Push and Sugar Tag patterns from publicly available YouTube footage or self-recorded examples.
   - Trim and sample frames from each video for analysis.
2. **Feature Extraction**
   - We will extract raw image frames for CNN-based spatial pattern recognition.
3. **Model Training**
   - We plan to use a Convolutional Neural Network (CNN) to classify:
     1. The dance pattern (Sugar Push vs. Sugar Tag)
     2. The competition division (Newcomer → Champion)
4. **Evaluation**
   - Evaluate accuracy, confusion matrices, and cross-division comparison results.
   - Visualize key differences and discuss potential improvements dancers could apply.
5. **Deliverables**
   - Final report
   - Trained model
   - Visualization notebook (showing prediction and attention maps)

### Data Collection Plan
- **Sources:** Publicly available WCS competition clips on YouTube, with credit and citation.  
- **Preprocessing:**
  - Download video segments using `yt-dlp`
  - Trim each clip to 3–6 seconds around the pattern using `ffmpeg`
  - Extract uniformly spaced frames (e.g., 12–16 per clip, resized to 224×224)
- **Ethics:**  
  - Use videos that are publicly available or self-recorded.
  - Avoid private or monetized content.

---
### Model Plans
#### Baseline Model — CNN (Frame-based)
- **Architecture:**  
  - CNN
  - Temporal pooling 
  - Two linear classification heads:
    - Head 1 → Pattern classification (2 classes)
    - Head 2 → Division classification (5 classes)
### Project Timeline
| Week | Milestone | Description |
|------|------------|-------------|
| 1 | **Topic Approval & Setup** | Finalize the research question, confirm tools (PyTorch, torchvision, ffmpeg), and collect initial reference videos for Sugar Push and Sugar Tag patterns. |
| 2 | **Data Collection & Labeling** | Download and trim selected WCS clips using `yt-dlp` and `ffmpeg`. Create `labels.csv` with pattern and division labels. |
| 3 | **Frame Extraction & Dataset Preparation** | Sample frames (8–16 per clip), organize datasets into train/val/test splits, and verify class balance. |
| 4 | **Model Development & Training** | Implement and train the CNN (ResNet18 backbone) for dual classification tasks: pattern and division. Optionally test a small MLP on pose-based features. |
| 5 | **Evaluation & Report** | Generate accuracy and confusion matrix results, visualize feature attention (Grad-CAM), and write the final report + presentation slides. |
---
### .gitignore and License
**.gitignore**
pycache/
*.mp4
*.jpg
*.csv
*.npy
*.pth
*.ipynb_checkpoints

**License:** MIT License — open for academic and research use only.
