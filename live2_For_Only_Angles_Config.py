import cv2, json, math, time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_styles
import tensorflow as tf
from tensorflow import keras

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KERAS_MODEL_PATH = r'C:\ANYFITCOACH\SquatsModelTest\Squats_Model\best_model (9).keras'
SCALER_JSON_PATH = r'C:\ANYFITCOACH\SquatsModelTest\Squats_Model\scaler_params (9).json'

CLASS_NAMES = [
    'Incorrect Posture',
    'Legs too Narrow',
    'Legs too Wide',
    'Not a Squat',
    'Perfect Squats',
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEPT_LM    = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
_i         = {lm: idx for idx, lm in enumerate(KEPT_LM)}
L_SHOULDER = _i[11]; R_SHOULDER = _i[12]
L_HIP      = _i[23]; R_HIP      = _i[24]
L_KNEE     = _i[25]; R_KNEE     = _i[26]
L_ANKLE    = _i[27]; R_ANKLE    = _i[28]
L_HEEL     = _i[29]; R_HEEL     = _i[30]
L_FOOT     = _i[31]; R_FOOT     = _i[32]

N_KEPT    = 12
COORD_DIM = N_KEPT * 4   # 48 — used internally for angle math only
ANGLE_DIM = 10
FEAT_DIM  = 10            # model only sees angles
WINDOW    = 45
NUM_CLS   = len(CLASS_NAMES)

EMA_ALPHA   = 0.15        # smoother than 0.28 for angle-only model
CONF_THRESH = 0.30       # slightly lower since angles-only is less overconfident
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
    """10 biomechanical angles — fully camera invariant."""
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
    knee_w  = np.linalg.norm(vec3(arr, L_KNEE)  - vec3(arr, R_KNEE))
    ankle_w = np.linalg.norm(vec3(arr, L_ANKLE) - vec3(arr, R_ANKLE)) + 1e-8
    ktr     = float(np.clip(knee_w / ankle_w, 0, 3)) / 3. * 180.
    hip_sym = float(abs(vec3(arr, L_HIP)[1] - vec3(arr, R_HIP)[1])) * 180.
    return np.array([lk, rk, lh, rh, la, ra, tl, fw, ktr, hip_sym],
                    dtype=np.float32) / 180.

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
    """Returns angles only — coords used internally but not fed to model."""
    return compute_joint_angles(arr)   # shape (10,)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Load scaler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Loading scaler...")
with open(SCALER_JSON_PATH) as f:
    sc = json.load(f)
MEAN  = np.array(sc['mean'],  dtype=np.float32)
SCALE = np.array(sc['scale'], dtype=np.float32)
print(f"Scaler loaded ✓  — MEAN shape: {MEAN.shape}  expected: ({FEAT_DIM},)")
assert MEAN.shape == (FEAT_DIM,), f"Scaler mismatch! Got {MEAN.shape}, expected ({FEAT_DIM},). Wrong scaler_params.json?"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Load model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Loading model...")
keras.config.enable_unsafe_deserialization()
model = keras.models.load_model(KERAS_MODEL_PATH)
print("Model loaded ✓")

dummy = np.zeros((1, WINDOW, FEAT_DIM), dtype=np.float32)
model(dummy, training=False)
print("Model warmed up ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Colour map
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLORS = {
    'Perfect Squats'   : (50,  205, 50),
    'Legs too Narrow'  : (0,   165, 255),
    'Legs too Wide'    : (180, 0,   180),
    'Not a Squat'      : (0,   215, 255),
    'Incorrect Posture': (0,   0,   220),
    'Ready'            : (120, 200, 120),
    'Squatting...'     : (180, 180, 80),
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

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )

        lm  = results.pose_landmarks.landmark
        raw = np.array(
            [[lm[i].x, lm[i].y, lm[i].z, lm[i].visibility] for i in KEPT_LM],
            dtype=np.float32
        )

        if raw[[L_HIP, R_HIP, L_KNEE, R_KNEE], 3].min() >= 0.40:
            norm = normalise_skeleton(raw)
            feat = frame_to_feature(norm)   # (10,) — angles only

            # Knee angle is used for rep state + display filtering.
            knee_ang = (
                angle_between(vec3(norm, L_HIP), vec3(norm, L_KNEE), vec3(norm, L_ANKLE)) +
                angle_between(vec3(norm, R_HIP), vec3(norm, R_KNEE), vec3(norm, R_ANKLE))
            ) / 2.

            # Keep rep state machine active regardless of classification gating.
            if   state == 'IDLE'      and knee_ang < SQUAT_ANGLE:
                state = 'SQUATTING'
            elif state == 'SQUATTING' and knee_ang > SQUAT_ANGLE + 15:
                state = 'COMPLETE'
                rep_count += 1
            elif state == 'COMPLETE':
                state = 'IDLE'

            buffer.append(feat)
            if len(buffer) > WINDOW:
                buffer.pop(0)

            if len(buffer) == WINDOW:
                win    = np.stack(buffer)                                        # (30, 10)
                scaled = ((win - MEAN) / SCALE)[np.newaxis].astype(np.float32)  # (1, 30, 10)

                probs = model.predict(scaled, verbose=0)[0]

                # Always keep EMA flowing; only gate what is shown.
                ema_probs[:] = EMA_ALPHA * probs + (1 - EMA_ALPHA) * ema_probs
                top_idx      = int(np.argmax(ema_probs))
                conf         = float(ema_probs[top_idx])

                if knee_ang > 155.:
                    label = 'Ready'
                    conf  = 0.
                else:
                    label = CLASS_NAMES[top_idx] if conf >= CONF_THRESH else 'Positioning...'

            else:
                label = f'Buffering  {len(buffer)}/{WINDOW}'
                conf  = 0.
                ema_probs[:] = 1. / NUM_CLS

        else:
            label = 'No pose'
            conf  = 0.
            ema_probs[:] = 1. / NUM_CLS

    else:
        label = 'No pose'
        conf  = 0.
        ema_probs[:] = 1. / NUM_CLS

    # ── Draw overlay ─────────────────────────────────────────────────────────
    H, W  = frame.shape[:2]
    color = COLORS.get(label, (200, 200, 200))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 95), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, label, (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2, cv2.LINE_AA)

    bar_max  = W // 2
    cv2.rectangle(frame, (16, 64), (16 + bar_max, 76), (40, 40, 40), -1)
    bar_fill = int(bar_max * conf)
    cv2.rectangle(frame, (16, 64), (16 + bar_fill, 76), color, -1)
    cv2.putText(frame, f'{conf:.0%}', (16 + bar_max + 8, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    cv2.putText(frame, f'Reps: {rep_count}   State: {state}',
                (W - 300, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 1, cv2.LINE_AA)

    bar_h     = 6
    bar_y     = H - 30
    bar_total = W - 32
    bar_chunk = bar_total // NUM_CLS
    for i, cls in enumerate(CLASS_NAMES):
        p    = float(ema_probs[i])
        bx   = 16 + i * bar_chunk
        bcol = COLORS.get(cls, (150, 150, 150))
        cv2.rectangle(frame, (bx, bar_y), (bx + bar_chunk - 4, bar_y + bar_h),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (bx, bar_y), (bx + int((bar_chunk - 4) * p), bar_y + bar_h),
                      bcol, -1)
        short = cls.replace('Legs too ', '').replace('Not a ', '').replace('Perfect ', '✓ ')
        cv2.putText(frame, short, (bx, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, bcol, 1, cv2.LINE_AA)

    fps       = 1.0 / (time.time() - prev_time + 1e-8)
    prev_time = time.time()
    cv2.putText(frame, f'{fps:.0f} fps', (W - 80, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

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