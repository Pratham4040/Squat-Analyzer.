# SquatsModelTest

SquatsModelTest is a real-time squat classification demo built with MediaPipe Pose, OpenCV, and a TensorFlow/Keras model. It uses webcam input to detect body landmarks, normalize the pose, and classify squat form live.

## What the model does

The current pipeline in `live2.py` performs these steps:

1. Detect pose landmarks from the webcam with MediaPipe.
2. Keep 12 key landmarks for the lower and upper body.
3. Normalize the skeleton so the model is less sensitive to camera position and scale.
4. Convert each frame into 10 biomechanical angle features.
5. Build a 45-frame sequence window.
6. Run the sequence through a Keras classifier and smooth the output with an exponential moving average.
7. Display one of these classes:
   - Incorrect Posture
   - Legs too Narrow
   - Legs too Wide
   - Not a Squat
   - Perfect Squats

The squat rep counter is driven by knee-angle thresholds, not by the classifier output.

## Deep Learning architecture

The current model is a sequence classifier:

- Input: `45 x 10` angle features per sample
- Feature source: normalized pose landmarks from MediaPipe
- Model type: trained Keras neural network saved as `.keras`
- Post-processing: EMA smoothing + confidence thresholding

Supporting files:

- `Squats_Model/best_model (9).keras` - trained Keras model
- `Squats_Model/scaler_params (9).json` - feature normalization values

There is also an older variant in `live.py` that uses a different feature format and a different model/scaler pair.

## Setup

1. Create and activate a Python virtual environment.
2. Install the project dependencies:

```bash
pip install opencv-python mediapipe numpy tensorflow
```

3. Make sure the model paths inside `live2.py` point to the files in `Squats_Model/`.
4. Connect a webcam.

## Run

```bash
python live2.py
```

Controls:

- `Q` - quit
- `R` - reset rep counter

## Notes

- `live2.py` is the current angle-only version and is the best starting point.
- `live.py` is an older feature-rich version kept for comparison.
- The `All Models/` and `Colab/` folders contain training artifacts and notebook history.