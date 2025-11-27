import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import math

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
DATA_DIR = Path('data/keypoints_1')
LABELS_CSV = Path('data/hp.csv')
OUT_DIR = Path('data/processed')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_labels_mapping():
    if not LABELS_CSV.exists():
        print(f"Warning: {LABELS_CSV} not found")
        return None
    
    labels_df = pd.read_csv(LABELS_CSV)
    mapping = {}
    for _, row in labels_df.iterrows():
        video_name = f"{row['division']}_{row['id']}"
        mapping[video_name] = row['labels']
    return mapping

def find_videos(root=DATA_DIR):
    records = []
    if not root.exists():
        raise FileNotFoundError(f"Data dir not found: {root}")
    
    labels_mapping = load_labels_mapping()
    
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        division = label_dir.name
        
        for video_dir in sorted(label_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            
            np_path = video_dir / 'keypoints.npy'
            if not np_path.exists():
                continue
            
            video_name = video_dir.name
            trick_label = labels_mapping.get(video_name, division) if labels_mapping else division
            
            records.append({
                'label': trick_label,
                'division': division,
                'video': video_name,
                'np_path': np_path
            })
    
    return pd.DataFrame(records)

def load_keypoints(np_path: Path):
    arr = np.load(np_path)
    if arr.ndim != 3: 
        raise ValueError(f"Unexpected shape {np_path}: {arr.shape}")
    return arr.astype(np.float32)

def interpolate_missing(arr, visibility_threshold=0.1):
    out = arr.copy()
    for lm in range(out.shape[1]):
        vis = out[:, lm, 3]
        bad = vis < visibility_threshold
        if bad.all(): continue
        for d in range(3):
            vals = out[:, lm, d].astype(float)
            vals[bad] = np.nan
            s = pd.Series(vals)
            vals_filled = s.interpolate(limit_direction='both').bfill().ffill().values
            out[:, lm, d] = vals_filled.astype(np.float32)
    return out

def torso_scale_single(lm):
    left_sh = lm[11][:2]
    right_sh = lm[12][:2]
    left_hip = lm[23][:2]
    right_hip = lm[24][:2]
    mid_sh = (left_sh + right_sh) / 2.0
    mid_hip = (left_hip + right_hip) / 2.0
    d = np.linalg.norm(mid_sh - mid_hip)
    return max(d, 1e-6)

def normalize_by_torso(arr):
    out = arr.copy()
    for t in range(out.shape[0]):
        lm = out[t]
        hip_center = (lm[23][:2] + lm[24][:2]) / 2.0
        scale = torso_scale_single(lm)
        out[t, :, :2] = (lm[:, :2] - hip_center) / scale
    return out

def pad_or_truncate(arr, target_len=64, strategy='center'):
    T = arr.shape[0]
    if T == target_len: return arr
    if T > target_len:
        if strategy == 'center':
            start = max(0, (T - target_len) // 2)
            return arr[start:start+target_len]
        return arr[:target_len]
    pad = np.zeros((target_len - T, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)

def angle_at(a, b, c):
    ba = a - b
    bc = c - b
    lena = np.linalg.norm(ba)
    lenb = np.linalg.norm(bc)
    if lena < 1e-6 or lenb < 1e-6: return 0.0
    cosang = np.dot(ba, bc) / (lena * lenb)
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.acos(cosang)

ANGLE_TRIPLETS = [(11, 13, 15), (12, 14, 16), (23, 25, 27), (24, 26, 28)]
def compute_frame_features(seq):
    T = seq.shape[0]
    flat = seq[:, :, :3].reshape(T, -1)
    angles = np.zeros((T, len(ANGLE_TRIPLETS)), dtype=np.float32)
    for t in range(T):
        for i, (a, b, c) in enumerate(ANGLE_TRIPLETS):
            angles[t, i] = angle_at(seq[t, a, :2], seq[t, b, :2], seq[t, c, :2])
    
    important = [11, 12, 23, 24, 13, 14, 25, 26]
    pdists = []
    for t in range(T):
        coords = seq[t, important, :2]
        dists = []
        for i in range(coords.shape[0]):
            for j in range(i+1, coords.shape[0]):
                dists.append(np.linalg.norm(coords[i] - coords[j]))
        pdists.append(dists)
    pdists = np.array(pdists)
    
    vel = np.vstack((np.zeros((1, flat.shape[1])), np.diff(flat, axis=0)))
    feats = np.concatenate([flat, angles, pdists, vel], axis=1)
    return feats

def aggregate_video_features(seq, target_len=64):
    seq = interpolate_missing(seq)
    seq = normalize_by_torso(seq)
    seq = pad_or_truncate(seq, target_len=target_len)
    frame_feats = compute_frame_features(seq)
    mean = frame_feats.mean(axis=0)
    std = frame_feats.std(axis=0)
    return np.concatenate([mean, std])

def build_dataset(index_df, target_len=64):
    X = []
    y = []
    for _, r in index_df.iterrows():
        seq = load_keypoints(r['np_path'])
        vec = aggregate_video_features(seq, target_len=target_len)
        X.append(vec)
        y.append(r['label'])
    return np.array(X), np.array(y)

# --- Main Analysis ---
def main():
    print("Loading data...")
    df = find_videos()
    print(f"Found {len(df)} videos.")
    print(df['label'].value_counts())
    
    X, y = build_dataset(df)
    
    # Filter for target classes only
    target_classes = ['sugar_push', 'sugar_tag']
    mask = np.isin(y, target_classes)
    X = X[mask]
    y = y[mask]
    
    print(f"Filtered Dataset shape: {X.shape}")
    print(pd.Series(y).value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models and grids
    models = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'classifier__n_neighbors': [3, 5, 7, 9],
                'classifier__weights': ['uniform', 'distance']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'classifier__max_depth': [None, 5, 10, 15],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['rbf', 'linear']
            }
        }
    }
    
    results = []
    best_overall_model = None
    best_overall_score = -1
    best_model_name = ""
    
    print("\nStarting model comparison...")
    
    for name, config in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', config['model'])
        ])
        
        grid = GridSearchCV(
            pipeline, 
            config['params'], 
            cv=StratifiedKFold(n_splits=3), 
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        
        best_est = grid.best_estimator_
        y_pred = best_est.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"  Best Params: {grid.best_params_}")
        print(f"  Test Accuracy: {acc:.2%}")
        
        results.append({
            'Model': name,
            'Best Params': str(grid.best_params_),
            'Test Accuracy': acc
        })
        
        if acc > best_overall_score:
            best_overall_score = acc
            best_overall_model = best_est
            best_model_name = name
            
    # Save results
    results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
    print("\n--- Final Results ---")
    print(results_df)
    results_df.to_csv(OUT_DIR / 'model_comparison_results.csv', index=False)
    
    # Save best model
    if best_overall_model:
        model_path = OUT_DIR / 'best_model.joblib'
        joblib.dump(best_overall_model, model_path)
        print(f"\nSaved best model ({best_model_name}) to {model_path}")
        
        # Detailed report for best model
        y_pred = best_overall_model.predict(X_test)
        print(f"\nClassification Report for {best_model_name}:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
