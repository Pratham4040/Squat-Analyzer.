# SquatsModelTest

SquatsModelTest is a squat posture classification project with:
1. A training notebook pipeline in [SquatsModelTest/Colab/squat_classifier_final_V2.ipynb](SquatsModelTest/Colab/squat_classifier_final_V2.ipynb)
2. Live inference scripts for webcam demo in [SquatsModelTest/live2_For_Only_Angles_Config.py](SquatsModelTest/live2_For_Only_Angles_Config.py) and [SquatsModelTest/live_For_Coords+Angles.py](SquatsModelTest/live_For_Coords+Angles.py)

It predicts the following 5 classes:
1. Incorrect Posture
2. Legs too Narrow
3. Legs too Wide
4. Not a Squat
5. Perfect Squats

## Dataset

Dataset link: [Add link here]

Dataset folder structure expected by the V2 notebook:
1. Root folder containing one subfolder per class
2. Subfolder names should match class names (case-insensitive fallback is included)
3. Supported video extensions: .mp4, .mov, .avi, .mkv, .webm, .m4v

Expected class folders:
1. Incorrect Posture
2. Legs too Narrow
3. Legs too Wide
4. Not a Squat
5. Perfect Squats

## V2 Notebook Pipelines

Reference notebook: [SquatsModelTest/Colab/squat_classifier_final_V2.ipynb](SquatsModelTest/Colab/squat_classifier_final_V2.ipynb)

### 1) Environment and config pipeline

1. Install dependencies (MediaPipe, TensorFlow, scikit-learn, matplotlib, seaborn, tqdm)
2. Import training + visualization + Colab utilities
3. Mount Google Drive
4. Set config values:
1. DATASET_ROOT
2. OUTPUT_DIR
3. CLASS_NAMES
4. WINDOW_SIZE, STRIDE, TARGET_FPS, MAX_FRAMES
5. AUG_ROTATIONS, AUG_NOISE_STD

### 2) Landmark and feature-definition pipelines

All variants use the same 12 kept landmarks (elbows/wrists/hands excluded):
1. 11, 12 (shoulders)
2. 23, 24 (hips)
3. 25, 26 (knees)
4. 27, 28 (ankles)
5. 29, 30 (heels)
6. 31, 32 (foot index)

Notebook defines these feature pipelines:
1. Coords + 8 angles:
1. COORD_DIM = 48
2. ANGLE_DIM = 8
3. FEAT_DIM = 56
2. Coords + 10 angles:
1. COORD_DIM = 48
2. ANGLE_DIM = 10
3. FEAT_DIM = 58
3. Only 10 angles:
1. ANGLE_DIM = 10
2. FEAT_DIM = 10

### 3) Geometry and normalization pipelines

Core geometry helpers in notebook:
1. Vector extraction
2. 3-point joint angle computation
3. Trunk angle from vertical
4. Optional Y-axis coordinate rotation for augmentation

Normalization pipeline:
1. Hip-center translation (center body at pelvis)
2. Y-axis alignment using hip vector in XZ plane (camera-angle robustness)
3. Scale normalization using shoulder-center distance (size robustness)

Angle sets used:
1. 8-angle set:
1. Left/right knee flexion
2. Left/right hip flexion
3. Left/right ankle angle
4. Trunk lean
5. Foot-width ratio feature
2. 10-angle set:
1. All 8-angle features
2. Knee-tracking ratio (knee width vs ankle width)
3. Hip symmetry feature

### 4) Video processing and augmentation pipelines

Shared video processing steps:
1. Read video with OpenCV
2. Sample to TARGET_FPS
3. Run MediaPipe Pose per sampled frame
4. Keep frame only when critical joints pass visibility threshold
5. Convert valid frames to normalized features
6. Build sliding windows of length WINDOW_SIZE with STRIDE

Augmentation pipeline for Coords + 8 angles:
1. Y-axis rotations with additive noise
2. Coordinate noise augmentation
3. Foot-width scale jitter
4. Left-right mirror
5. Speed jitter

Augmentation pipeline for Coords + 10 angles:
1. Y-axis rotations with additive noise
2. Coordinate noise augmentation
3. Left-right mirror
4. Speed jitter
5. Foot-width jitter
6. Global scale jitter

Augmentation pipeline for Only 10 angles:
1. Y-axis rotations with additive noise
2. Coordinate noise augmentation before angle conversion
3. Left-right mirror
4. Speed jitter
5. Foot-width jitter
6. Global scale jitter
7. Angle-space noise injection
8. Temporal dropout (random dropped/occluded frames)
9. Time reversal

### 5) Dataset build and split pipeline

1. Scan class folders and collect video paths + labels
2. Split at video level to avoid leakage
1. Train split
2. Validation split
3. Test split
3. Process each split separately (augmentation enabled only for training)
4. Shuffle training samples after augmentation

### 6) Dataset caching pipeline

Notebook includes both save and load cache flows:
1. Save:
1. X_train, y_train, X_val, y_val, X_test, y_test as .npy
2. Save cache metadata JSON (shapes, class distribution, config)
2. Load:
1. Validate required cache files
2. Validate config compatibility (WINDOW_SIZE, STRIDE, FEAT_DIM)
3. Load arrays only when compatible

### 7) Scaling and label-prep pipeline

1. Fit StandardScaler on flattened train frames only
2. Transform val/test with same scaler
3. Save scaler parameters and model metadata
4. Apply MixUp on training set
5. Build one-hot labels for val/test where needed
6. Compute class weights

### 8) Training and evaluation pipeline

1. Optimizer: AdamW
2. Learning-rate schedule: cosine decay
3. Callbacks:
1. EarlyStopping with best-weight restore
2. ModelCheckpoint saving best validation model
4. Metrics tracking: train/val accuracy and loss
5. Evaluation:
1. Classification report
2. Normalized confusion matrix plot
6. Exports:
1. best_model.keras
2. scaler_params and metadata JSON
3. training curves and confusion matrix images

## Model Architectures (Three Main Variants in V2 Notebook)

### Model 1: 1D CNN + BiLSTM + Multi-Head Self-Attention (Coords + 8 Angles)

Input:
1. Window shape: (45, 56)
2. Stream split:
1. Coordinates stream (48 dims)
2. Angle stream (8 dims, projected)

Backbone:
1. Conv1D blocks + batch norm + pooling + dropout on coordinates
2. Concatenate downsampled angle projection
3. BiLSTM sequence modeling
4. Multi-head self-attention with residual normalization
5. Global average pooling
6. Dense classifier head

### Model 2: 1D CNN + BiLSTM + Multi-Head Self-Attention (Coords + 10 Angles, deeper head)

Input:
1. Window shape: (45, 58)
2. Stream split:
1. Coordinates stream (48 dims)
2. Angle stream (10 dims, projected)

Backbone:
1. Conv1D feature extractor on coordinates
2. Merge with projected/downsampled angle stream
3. BiLSTM temporal encoder
4. Multi-head self-attention
5. Deeper dense head (multiple fully connected layers)
6. Softmax output for 5-class classification

### Model 3: BiLSTM + Multi-Head Self-Attention + Dense NN (Angles Only)

Input:
1. Window shape: (45, 10)
2. Features: angle-only sequence

Backbone:
1. Stacked BiLSTM layers (no coordinate CNN branch)
2. Multi-head self-attention for key-frame emphasis
3. Global pooling
4. Dense classification head
5. Softmax output for 5 classes

Note:
1. The notebook also includes an additional DD-Net-inspired TCN fusion experiment for angle-only input.

## Execution Steps

### A) Notebook execution (training + evaluation)

Notebook: [SquatsModelTest/Colab/squat_classifier_final_V2.ipynb](SquatsModelTest/Colab/squat_classifier_final_V2.ipynb)

1. Open notebook in Colab
2. Run install cell, then restart runtime
3. Run import cell
4. Mount Drive and set DATASET_ROOT and OUTPUT_DIR in config cell
5. Select the feature pipeline and corresponding model variant you want to train
6. Run geometry + video pipeline cells
7. Build dataset from class folders
8. Optionally save cache
9. Run scaling and label-prep cell
10. Run model-definition cell for selected architecture
11. Run training cell
12. Run training-curves and evaluation cells
13. Export and copy model artifacts to your serving location

### B) Live demo execution

Primary script: [SquatsModelTest/live2_For_Only_Angles_Config.py](SquatsModelTest/live2_For_Only_Angles_Config.py)

1. Create/activate Python environment
2. Install runtime dependencies:

```bash
pip install opencv-python mediapipe numpy tensorflow
```

3. Update model/scaler/meta paths in the live script to your exported artifacts
4. Ensure webcam access is available
5. Run the script:

```bash
python live2_For_Only_Angles_Config.py
```

Common controls:
1. Q to quit
2. R to reset rep counter

Alternative script for coords+angles flow:
1. [SquatsModelTest/live_For_Coords+Angles.py](SquatsModelTest/live_For_Coords+Angles.py)

## Sample Output Screenshots

### Model 1: DD-Net TCN (Best)

Training Accuracy and Loss:

<img src="Assets/Screenshots/Model_1_DD-Net%20TCN%20(Best)/Acc_and_Loss_Curve.png" alt="Model 1 - Accuracy and Loss" width="640" />

Confusion Matrix:

<img src="Assets/Screenshots/Model_1_DD-Net%20TCN%20(Best)/Confusion_Matrix.png" alt="Model 1 - Confusion Matrix" width="480" />

### Model 2: BiLSTM + Attention

Training Accuracy and Loss:

<img src="Assets/Screenshots/Model_2_BiLSTM%20+%20Attention/Acc_and_Loss_Curve.png" alt="Model 2 - Accuracy and Loss" width="640" />

Confusion Matrix:

<img src="Assets/Screenshots/Model_2_BiLSTM%20+%20Attention/Confusion_Matrix.png" alt="Model 2 - Confusion Matrix" width="480" />

### Model 3: BiLSTM + Attention + Dropout

Training Accuracy and Loss:

<img src="Assets/Screenshots/Model_3_BiLSTM%20+%20Attn%20+%20Dropout/Acc_and_Loss_Curve.png" alt="Model 3 - Accuracy and Loss" width="640" />

Confusion Matrix:

<img src="Assets/Screenshots/Model_3_BiLSTM%20+%20Attn%20+%20Dropout/Confusion_Matrix.png" alt="Model 3 - Confusion Matrix" width="480" />

### Model 4: Coords + Angles

Training Accuracy and Loss:

<img src="Assets/Screenshots/Model4_Cords+Angles/Acc_and_Loss_Curve.png" alt="Model 4 - Accuracy and Loss" width="640" />

Confusion Matrix:

<img src="Assets/Screenshots/Model4_Cords+Angles/Confusion_Matrix.png" alt="Model 4 - Confusion Matrix" width="480" />

### Live Demo Screenshots

Add your live demo screenshots in the same folder and replace the placeholders below:

1. Live prediction screen

<img src="Assets/Screenshots/live_demo_prediction.png" alt="Live Demo - Prediction" width="640" />

2. Live rep counter screen

<img src="Assets/Screenshots/live_demo_rep_counter.png" alt="Live Demo - Rep Counter" width="640" />

3. Live posture correction feedback (optional)

<img src="Assets/Screenshots/live_demo_feedback.png" alt="Live Demo - Feedback" width="640" />

Screenshot size guideline:
1. Prefer image width around 1280 px or less before upload
2. Keep each file ideally under 600 KB (up to 1 MB is fine)

## Project Artifacts

Important folders/files:
1. [SquatsModelTest/Colab](SquatsModelTest/Colab)
2. [SquatsModelTest/Squats_Model](SquatsModelTest/Squats_Model)
3. [SquatsModelTest/All Models](SquatsModelTest/All Models)