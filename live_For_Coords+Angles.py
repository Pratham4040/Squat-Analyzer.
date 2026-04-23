import cv2, json, math, time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_styles
import tensorflow as tf
from tensorflow import keras

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG — update these paths to where you saved your model files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KERAS_MODEL_PATH = r'C:\ANYFITCOACH\SquatsModelTest\Squats_Model\best_model (3).keras'
SCALER_JSON_PATH = r'C:\ANYFITCOACH\SquatsModelTest\Squats_Model\scaler_params (3).json'

# Must match exactly what you used in training (your 4 folder names)
CLASS_NAMES = [
    'Incorrect Posture',
    'Legs too Narrow',
    'Legs too Wide',
    'Not a Squat',
    'Perfect Squats',
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants — must match training config exactly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEPT_LM   = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
_i        = {lm: idx for idx, lm in enumerate(KEPT_LM)}
L_SHOULDER= _i[11]; R_SHOULDER= _i[12]
L_HIP     = _i[23]; R_HIP     = _i[24]
L_KNEE    = _i[25]; R_KNEE    = _i[26]
L_ANKLE   = _i[27]; R_ANKLE   = _i[28]
L_HEEL    = _i[29]; R_HEEL    = _i[30]
L_FOOT    = _i[31]; R_FOOT    = _i[32]
N_KEPT    = 12
COORD_DIM = N_KEPT * 4   # 48
ANGLE_DIM = 8
FEAT_DIM  = 56
WINDOW    = 30
NUM_CLS   = len(CLASS_NAMES)

EMA_ALPHA   = 0.15
CONF_THRESH = 0.38
SQUAT_ANGLE = 130.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Geometry helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def vec3(arr, idx):
    return arr[idx, :3]

def angle_between(a, b, c):
    ba, bc = a - b, c - b
    cos_v  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(float(np.clip(cos_v, -1, 1))))

def angle_from_vertical(v):
    down  = np.array([0., -1., 0.])
    cos_v = np.dot(v, down) / (np.linalg.norm(v) * np.linalg.norm(down) + 1e-8)
    return math.degrees(math.acos(float(np.clip(cos_v, -1, 1))))

def compute_joint_angles(arr):
    lk = angle_between(vec3(arr, L_HIP),      vec3(arr, L_KNEE),   vec3(arr, L_ANKLE))
    rk = angle_between(vec3(arr, R_HIP),       vec3(arr, R_KNEE),   vec3(arr, R_ANKLE))
    lh = angle_between(vec3(arr, L_SHOULDER),  vec3(arr, L_HIP),    vec3(arr, L_KNEE))
    rh = angle_between(vec3(arr, R_SHOULDER),  vec3(arr, R_HIP),    vec3(arr, R_KNEE))
    la = angle_between(vec3(arr, L_KNEE),      vec3(arr, L_ANKLE),  vec3(arr, L_HEEL))
    ra = angle_between(vec3(arr, R_KNEE),      vec3(arr, R_ANKLE),  vec3(arr, R_HEEL))
    spine = (vec3(arr, L_SHOULDER) + vec3(arr, R_SHOULDER)) / 2.
    tl    = angle_from_vertical(spine)
    foot_w = np.linalg.norm(vec3(arr, L_FOOT) - vec3(arr, R_FOOT))
    hip_w  = np.linalg.norm(vec3(arr, L_HIP)  - vec3(arr, R_HIP)) + 1e-8
    fw     = float(np.clip(foot_w / hip_w, 0, 3)) / 3. * 180.
    return np.array([lk, rk, lh, rh, la, ra, tl, fw], dtype=np.float32) / 180.

def normalise_skeleton(raw):
    arr   = raw.copy().astype(np.float32)
    hip_c = (arr[L_HIP, :3] + arr[R_HIP, :3]) / 2.
    arr[:, :3] -= hip_c
    hip_vec = arr[R_HIP, :3] - arr[L_HIP, :3]
    hip_xz  = np.array([hip_vec[0], 0., hip_vec[2]], dtype=np.float32)
    mag     = np.linalg.norm(hip_xz)
    if mag > 1e-6:
        hip_xz /= mag
        ct, st = float(hip_xz[0]), float(hip_xz[2])
        Ry = np.array([[ct, 0., st], [0., 1., 0.], [-st, 0., ct]], dtype=np.float32)
        arr[:, :3] = (Ry @ arr[:, :3].T).T
    sc = (arr[L_SHOULDER, :3] + arr[R_SHOULDER, :3]) / 2.
    arr[:, :3] /= (np.linalg.norm(sc) + 1e-8)
    return arr

def frame_to_feature(arr):
    return np.concatenate([arr.flatten(), compute_joint_angles(arr)])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Load scaler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Loading scaler...")
with open(SCALER_JSON_PATH) as f:
    sc = json.load(f)
MEAN  = np.array(sc['mean'],  dtype=np.float32)
SCALE = np.array(sc['scale'], dtype=np.float32)
print("Scaler loaded ✓")
# Add this right after loading the scaler, as a one-time sanity check:
print(f"MEAN shape  : {MEAN.shape}   expected: ({WINDOW * FEAT_DIM},)  = ({30 * 56},)")
print(f"SCALE shape : {SCALE.shape}  expected: ({WINDOW * FEAT_DIM},)  = ({30 * 56},)")
# Should print 1680 for both

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Load Keras model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Loading model...")
keras.config.enable_unsafe_deserialization()
model = keras.models.load_model(KERAS_MODEL_PATH)
print("Model loaded ✓")

# Warm up the model — first prediction is always slow
dummy = np.zeros((1, WINDOW, FEAT_DIM), dtype=np.float32)
model.predict(dummy, verbose=0)
print("Model warmed up ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Colour map per class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLORS = {
    'Perfect Squats'   : (50,  205, 50),
    'Legs too Narrow'  : (0,   165, 255),
    'Legs too Wide'    : (180, 0,   180),
    'Not a Squat'      : (0,   215, 255),
    'Incorrect Posture': (0,   0,   220),
    'Positioning...'   : (150, 150, 150),
    'No pose'          : (80,  80,  80),
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Runtime state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

buffer    = []
ema_probs = np.ones(NUM_CLS, dtype=np.float32) / NUM_CLS
state     = 'IDLE'
rep_count = 0
label     = 'Waiting...'
conf      = 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pose detector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Webcam
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

print("\nRunning — press Q to quit, R to reset rep counter\n")
prev_time = time.time()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Lost webcam feed")
        break

    # ── MediaPipe ────────────────────────────────────────────────────────────
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        # Draw skeleton on frame
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )

        # Extract landmarks
        lm  = results.pose_landmarks.landmark
        raw = np.array(
            [[lm[i].x, lm[i].y, lm[i].z, lm[i].visibility] for i in KEPT_LM],
            dtype=np.float32
        )

        # Visibility gate — skip frame if key joints not clearly visible
        if raw[[L_HIP, R_HIP, L_KNEE, R_KNEE], 3].min() >= 0.40:
            norm = normalise_skeleton(raw)
            feat = frame_to_feature(norm)

            buffer.append(feat)
            if len(buffer) > WINDOW:
                buffer.pop(0)

            if len(buffer) == WINDOW:
                # Scale
                win    = np.stack(buffer)                                        # (30, 56)
                scaled = ((win - MEAN) / SCALE)[np.newaxis].astype(np.float32) 

                # Inference
                probs = model.predict(scaled, verbose=0)[0]

                # EMA smoothing
                ema_probs[:] = EMA_ALPHA * probs + (1 - EMA_ALPHA) * ema_probs
                top_idx      = int(np.argmax(ema_probs))
                conf         = float(ema_probs[top_idx])
                label        = CLASS_NAMES[top_idx] if conf >= CONF_THRESH else 'Positioning...'

                # ── State machine ─────────────────────────────────────────
                knee_ang = (
                    angle_between(vec3(norm, L_HIP), vec3(norm, L_KNEE), vec3(norm, L_ANKLE)) +
                    angle_between(vec3(norm, R_HIP), vec3(norm, R_KNEE), vec3(norm, R_ANKLE))
                ) / 2.

                if   state == 'IDLE'      and knee_ang < SQUAT_ANGLE:
                    state = 'SQUATTING'
                elif state == 'SQUATTING' and knee_ang > SQUAT_ANGLE + 15:
                    state = 'COMPLETE'
                    rep_count += 1
                elif state == 'COMPLETE':
                    state = 'IDLE'

            else:
                label = f'Buffering  {len(buffer)}/{WINDOW}'
                conf  = 0.
                ema_probs[:] = 1. / NUM_CLS 
        else:
            label = 'No pose'
            conf  = 0.
            ema_probs[:] = 1. / NUM_CLS   # ← add this line

    else:
            label = 'No pose'
            conf  = 0.
            ema_probs[:] = 1. / NUM_CLS   # ← add this line

    # ── Draw overlay ─────────────────────────────────────────────────────────
    H, W  = frame.shape[:2]
    color = COLORS.get(label, (200, 200, 200))

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 95), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # Main label
    cv2.putText(frame, label, (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2, cv2.LINE_AA)

    # Confidence bar background
    bar_max = W // 2
    cv2.rectangle(frame, (16, 64), (16 + bar_max, 76), (40, 40, 40), -1)
    # Confidence bar fill
    bar_fill = int(bar_max * conf)
    cv2.rectangle(frame, (16, 64), (16 + bar_fill, 76), color, -1)
    # Confidence percentage
    cv2.putText(frame, f'{conf:.0%}', (16 + bar_max + 8, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Reps + state (top right)
    cv2.putText(frame, f'Reps: {rep_count}   State: {state}',
                (W - 300, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 1, cv2.LINE_AA)

    # All class probabilities bar (bottom of screen)
    bar_h     = 6
    bar_y     = H - 30
    bar_total = W - 32
    bar_chunk = bar_total // NUM_CLS
    for i, cls in enumerate(CLASS_NAMES):
        p     = float(ema_probs[i])
        bx    = 16 + i * bar_chunk
        bcol  = COLORS.get(cls, (150, 150, 150))
        # Track
        cv2.rectangle(frame, (bx, bar_y), (bx + bar_chunk - 4, bar_y + bar_h),
                      (40, 40, 40), -1)
        # Fill
        cv2.rectangle(frame, (bx, bar_y), (bx + int((bar_chunk - 4) * p), bar_y + bar_h),
                      bcol, -1)
        # Class label
        short = cls.replace('Legs too ', '').replace('Not a ', '').replace('Perfect ', '✓ ')
        cv2.putText(frame, short, (bx, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, bcol, 1, cv2.LINE_AA)

    # FPS counter (bottom right)
    fps       = 1.0 / (time.time() - prev_time + 1e-8)
    prev_time = time.time()
    cv2.putText(frame, f'{fps:.0f} fps', (W - 80, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    # ── Show ─────────────────────────────────────────────────────────────────
    cv2.imshow('Squat Classifier  |  Q = quit   R = reset reps', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        rep_count = 0
        buffer.clear()
        ema_probs[:] = 1. / NUM_CLS
        state = 'IDLE'
        print("Reset ✓")

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
pose.close()
cv2.destroyAllWindows()
print("Done.")