"""
============================================================================
 Computer Vision Project
============================================================================
 Run locally in VS Code with:
     python cv_project_local.py

 Requirements:
   - Webcam connected
   - OBS Studio installed and Virtual Camera started (click "Start Virtual Camera" in OBS)

 Controls (focus the Preview window first):
   1 stats overlay        2 RGB histogram
   3 equalization         4 Sobel edge filter
   5 face overlay (ON)    6 linear transform
   0 turn everything off  q quit
============================================================================
"""

# ---------------------------------------------------------------------------
# 0) Make sure required packages are present
# ---------------------------------------------------------------------------
import importlib
import subprocess
import sys


def _ensure(packages):
    """Install a package via pip only if it cannot be imported."""
    for module_name, pip_name in packages:
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"[setup] installing {pip_name} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", pip_name])


_ensure([
    ("cv2",          "opencv-python"),
    ("mediapipe",    "mediapipe==0.10.14"),
    ("numpy",        "numpy"),
    ("PIL",          "pillow"),
])

import cv2
import numpy as np


# ===========================================================================
#  IMAGE PROCESSING FUNCTIONS
#  All functions assume RGB uint8 images, shape (H, W, 3).
# ===========================================================================

# ---------------------------------------------------------------------------
#  BASICS
# ---------------------------------------------------------------------------

def compute_stats(image_rgb):
    """
    Compute basic statistics on the grayscale version of the image.
    Returns mean, std, min, max, and mode of pixel brightness values (0-255).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    flat = gray.flatten()   # convert 2D grid of pixels into a 1D list of values
    return {
        "mean": float(np.mean(flat)),           # average brightness
        "std":  float(np.std(flat)),            # how spread out the values are
        "min":  int(np.min(flat)),              # darkest pixel
        "max":  int(np.max(flat)),              # brightest pixel
        "mode": int(np.argmax(np.bincount(flat, minlength=256))),  # most common value
    }


def compute_entropy(image_rgb):
    """
    Shannon entropy (in bits) of the grayscale intensity distribution.
    High entropy  -> lots of detail / information in the image.
    Low entropy   -> flat / uniform image (e.g. blank wall).
    Formula: H = -sum(p * log2(p)) where p = probability of each intensity value.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()          # turn counts into probabilities
    prob = prob[prob > 0]             # drop zero bins (log(0) is undefined)
    return float(-np.sum(prob * np.log2(prob)))


def linear_transform(image_rgb, alpha=1.4, beta=15):
    """
    Linear point transformation: output = alpha * input + beta
      alpha > 1  increases contrast  (stretches the range of values)
      beta  > 0  increases brightness (shifts all values up)
    convertScaleAbs clips result to [0, 255] automatically.
    """
    return cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)


def equalize(image_rgb):
    """
    Histogram equalization: automatically improves contrast by redistributing
    pixel brightness values across the full 0-255 range.

    We equalize only the Y (luminance) channel in YCrCb color space
    so that colors are not distorted. Then convert back to RGB.
    """
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])   # equalize luminance only
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def sobel_filter(image_rgb):
    """
    Sobel edge detection filter.
    Measures how fast pixel values change in horizontal (gx) and vertical (gy)
    directions. The magnitude sqrt(gx^2 + gy^2) is high at edges (object
    boundaries) and low in flat regions.
    NOT an identity filter - visibly transforms the image into an edge map.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)    # horizontal gradient
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)    # vertical gradient
    mag = cv2.magnitude(gx, gy)                         # combined edge strength
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)        # return as 3-channel image


def draw_histogram_overlay(image_rgb, panel_w=320, panel_h=180, margin=12):
    """
    Draw the per-channel (R, G, B) histogram as three colored lines in ONE panel,
    blended into the top-right corner of the frame.

    Requirement: histogram for each channel separately, three lines in one plot.
    Built with OpenCV drawing (not matplotlib) so it runs at 30fps in real time.
    """
    h, w = image_rgb.shape[:2]
    panel = np.full((panel_h, panel_w, 3), 25, dtype=np.uint8)   # dark background

    # Draw one line per channel: R = red, G = green, B = blue
    colors = [(255, 70, 70), (70, 255, 70), (90, 150, 255)]
    for channel, color in enumerate(colors):
        hist = cv2.calcHist([image_rgb], [channel], None, [256], [0, 256]).flatten()
        cv2.normalize(hist, hist, 0, panel_h - 12, cv2.NORM_MINMAX)
        pts = np.array(
            [[int(x * panel_w / 256), panel_h - 1 - int(hist[x])] for x in range(256)],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(panel, [pts], isClosed=False, color=color, thickness=1)

    cv2.putText(panel, "RGB Histogram", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    # Blend panel onto frame (25% original frame + 85% panel)
    x0, y0 = w - panel_w - margin, margin
    roi = image_rgb[y0:y0 + panel_h, x0:x0 + panel_w]
    image_rgb[y0:y0 + panel_h, x0:x0 + panel_w] = cv2.addWeighted(roi, 0.25, panel, 0.85, 0)
    return image_rgb


def draw_stats_overlay(image_rgb, source_rgb):
    """
    Write the statistics (computed from the unprocessed source frame) as text
    in the top-left corner of the frame.
    Each line is drawn twice: thick black outline first, then yellow text on top,
    so it stays readable on any background.
    """
    s = compute_stats(source_rgb)
    e = compute_entropy(source_rgb)
    lines = [
        f"Mean: {s['mean']:.1f}",
        f"Std : {s['std']:.1f}",
        f"Min : {s['min']}   Max: {s['max']}",
        f"Mode: {s['mode']}",
        f"Entropy: {e:.2f} bits",
    ]
    y = 34
    for line in lines:
        cv2.putText(image_rgb, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)           # black outline
        cv2.putText(image_rgb, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2, cv2.LINE_AA)        # yellow text on top
        y += 30
    return image_rgb


# ---------------------------------------------------------------------------
#  SPECIAL TASK: Face landmarks (neural network) + fun overlay
# ---------------------------------------------------------------------------

# The Face Mesh model is loaded once (lazily) and reused.
# Loading it is slow (~1 second), so we never load it more than once.
_face_mesh = None


def _get_face_mesh():
    """
    Load MediaPipe Face Mesh neural network model once and reuse it.
    MediaPipe Face Mesh detects 468 specific landmarks on a human face.
    Pre-trained weights are used (allowed by the project requirements).
    """
    global _face_mesh
    if _face_mesh is None:
        try:
            # Newer MediaPipe versions need explicit submodule import
            from mediapipe.python.solutions import face_mesh as mp_face_mesh
        except ImportError:
            # Older MediaPipe versions use the classic attribute path
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
        _face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=2,              # detect up to 2 faces at once
            refine_landmarks=True,        # more precise eye and lip landmarks
            min_detection_confidence=0.5, # minimum confidence to report a face
            min_tracking_confidence=0.5,  # minimum confidence to keep tracking
        )
    return _face_mesh


def _midpoint(a, b):
    """Return the pixel midpoint between two (x, y) points."""
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def _dist(a, b):
    """Return the Euclidean distance between two (x, y) points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _draw_sunglasses(frame, left_eye, right_eye):
    """
    Draw two filled black circles (lenses) with a connecting bridge line.
    Radius scales with the distance between eyes so it always fits the face,
    regardless of how far the person is from the camera.
    """
    eye_dist = _dist(left_eye, right_eye)
    r = max(6, int(eye_dist * 0.42))                        # lens radius
    cv2.circle(frame, left_eye,  r, (12, 12, 12), -1)      # left lens
    cv2.circle(frame, right_eye, r, (12, 12, 12), -1)      # right lens
    cv2.line(frame, left_eye, right_eye, (12, 12, 12), max(3, r // 3))  # bridge


def _draw_mustache(frame, left_eye, right_eye, nose_bottom):
    """
    Draw a filled dark ellipse below the nose as a mustache.
    The angle is computed from the eye positions so the mustache rotates
    correctly when the person tilts their head.
    """
    eye_dist = _dist(left_eye, right_eye)
    # angle matches the tilt of the head using the eye line
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                  right_eye[0] - left_eye[0]))
    center = (nose_bottom[0], nose_bottom[1] + int(eye_dist * 0.12))
    axes = (max(8, int(eye_dist * 0.45)), max(4, int(eye_dist * 0.18)))
    cv2.ellipse(frame, center, axes, angle, 0, 360, (35, 25, 20), -1)


def apply_face_overlay(image_rgb):
    """
    SPECIAL TASK - Neural network face overlay.

    Runs the MediaPipe Face Mesh model on the current frame to detect facial
    landmarks (468 points per face). Uses specific landmark indices to find:
      - Left eye center  (midpoint of landmarks 33 and 133)
      - Right eye center (midpoint of landmarks 362 and 263)
      - Nose bottom      (landmark 2)

    Then draws sunglasses over the eyes and a mustache below the nose.
    The overlay follows the face as it moves, scales, and tilts in real time.
    Returns the frame unchanged if no face is detected.
    """
    face_mesh = _get_face_mesh()
    h, w = image_rgb.shape[:2]

    # MediaPipe expects RGB input - our pipeline is already RGB, no conversion needed
    results = face_mesh.process(image_rgb)

    # If no face found in this frame, return the original image unchanged
    if not results.multi_face_landmarks:
        return image_rgb

    for landmarks in results.multi_face_landmarks:
        def px(idx):
            # MediaPipe gives normalized coordinates (0.0 to 1.0)
            # Multiply by image dimensions to get actual pixel positions
            lm = landmarks.landmark[idx]
            return (int(lm.x * w), int(lm.y * h))

        # Compute eye centers and nose position from landmark indices
        left_eye   = _midpoint(px(33),  px(133))   # left eye inner/outer corners
        right_eye  = _midpoint(px(362), px(263))   # right eye inner/outer corners
        nose_bottom = px(2)                         # tip of the nose

        _draw_sunglasses(image_rgb, left_eye, right_eye)
        _draw_mustache(image_rgb, left_eye, right_eye, nose_bottom)

    return image_rgb


# ===========================================================================
#  LIVE PIPELINE  -  webcam -> processing -> OBS virtual camera
# ===========================================================================

_CONTROLS = """
[live] Controls (focus the 'Preview' window, then press a key):
   1 stats overlay        2 RGB histogram
   3 equalization         4 Sobel edge filter
   5 face overlay (ON)    6 linear transform
   0 turn everything off  q quit
"""


def run_local_live(width=640, height=480, fps=30, device_id=0):
    """
    Full real-time pipeline:
      1. Capture frames from the webcam (OpenCV)
      2. Apply the currently enabled effects (your image processing code)
      3. Show a local Preview window (with keyboard controls)
      4. Send processed frames to the OBS Virtual Camera (pyvirtualcam)

    The virtual camera appears as a selectable webcam in Zoom, Discord, Teams, etc.
    """
    _ensure([("pyvirtualcam", "pyvirtualcam")])
    import pyvirtualcam

    # Open the webcam using Windows Media Foundation backend
    # CAP_MSMF shares the webcam with other apps (like OBS), unlike CAP_DSHOW
    cap = cv2.VideoCapture(device_id, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {device_id}. "
            "Check that the webcam is connected and not exclusively used by another app."
        )

    # Toggle state for each effect - face overlay is ON by default for the demo
    state = {
        "stats":    False,
        "hist":     False,
        "equalize": False,
        "sobel":    False,
        "face":     True,    # special task on by default
        "linear":   False,
    }
    print(_CONTROLS)

    try:
        # Connect to OBS Virtual Camera driver
        # backend='obs' forces pyvirtualcam to use the OBS driver specifically
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, backend='obs') as cam:
            print(f"[live] Virtual camera running: {cam.device}")

            while True:
                ret, bgr = cap.read()       # read one frame from the webcam (BGR format)
                if not ret:
                    break                   # camera disconnected or stream ended

                # OpenCV reads frames as BGR, our pipeline works in RGB -> convert once
                source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                out = source.copy()         # work on a copy, keep source clean for stats

                # --- Apply pixel-value effects ---
                if state["equalize"]: out = equalize(out)
                if state["linear"]:   out = linear_transform(out)
                if state["sobel"]:    out = sobel_filter(out)

                # --- Apply special task (neural network face overlay) ---
                if state["face"]:     out = apply_face_overlay(out)

                # --- Draw informational overlays on top ---
                if state["hist"]:     out = draw_histogram_overlay(out)
                if state["stats"]:    out = draw_stats_overlay(out, source)

                # Resize if the frame dimensions don't match the virtual camera settings
                if out.shape[1] != width or out.shape[0] != height:
                    out = cv2.resize(out, (width, height))

                # Send processed frame to the virtual camera (expects RGB uint8)
                cam.send(np.ascontiguousarray(out, dtype=np.uint8))
                cam.sleep_until_next_frame()    # maintain the target FPS (30fps)

                # Show local preview window (convert back to BGR for OpenCV display)
                cv2.imshow("Preview (q to quit)", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

                # Check for keyboard input (wait 1ms per frame)
                key = cv2.waitKey(1) & 0xFF
                if   key == ord("q"): break                               # quit
                elif key == ord("1"): state["stats"]    = not state["stats"]
                elif key == ord("2"): state["hist"]     = not state["hist"]
                elif key == ord("3"): state["equalize"] = not state["equalize"]
                elif key == ord("4"): state["sobel"]    = not state["sobel"]
                elif key == ord("5"): state["face"]     = not state["face"]
                elif key == ord("6"): state["linear"]   = not state["linear"]
                elif key == ord("0"): state = {k: False for k in state}   # all off

    finally:
        cap.release()           # always release the webcam
        cv2.destroyAllWindows() # always close the preview window


# ===========================================================================
#  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_local_live()
