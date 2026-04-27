# SquatsModelTest

## How to Run Live Demo V2

### Prerequisites
- Python 3.8+
- Webcam
- Model file: `Squats_Model/best_model (9).keras`
- Scaler file: `Squats_Model/scaler_params (9).json`

### Setup
1. Open terminal in the project folder.
2. Create a virtual environment:

   ```bash
   python -m venv squat_venv
   ```

3. Activate the virtual environment:

   Windows:
   ```bash
   squat_venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   source squat_venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Run

```bash
python live2_For_Only_Angles_Config.py
```

### Controls
- Q: Quit
- R: Reset rep counter

## Dataset Curation

We curated the dataset ourselves using our own phones and recorded 199 videos of ourselves performing squats.

Classes used:
1. Incorrect Posture
2. Legs too Narrow
3. Legs too Wide
4. Not a Squat
5. Perfect Squats

## Evaluation Matrix (Model Parameter Sizes)

| Model | Name | Parameter Size |
|---|---|---|
| Model 1 | DD-Net TCN (Best) | 8.6K params |
| Model 2 | BiLSTM + Attention | 130K params |
| Model 3 | BiLSTM + Attention + Dropout | 130K params |
| Model 4 | Coords + Angles | 470K params |

## Model Evaluation Screenshots

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
