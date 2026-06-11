"""
============================================================================
 Computer Vision Project - SINGLE FILE
============================================================================
 Works in TWO environments automatically:

   * Google Colab  -> demo / debug mode (image, webcam selfie, or video).
                      Results are shown inline. (No virtual camera in the cloud!)
   * Local PC      -> full LIVE virtual-camera mode for your actual demo
                      (needs a webcam + OBS Studio + pyvirtualcam).

 In Colab, you can paste this whole thing into ONE cell and run it.
 If you prefer the classic Colab style, put this as the FIRST line of the cell:
     !pip install opencv-python mediapipe matplotlib numpy pillow
 (The block below also auto-installs anything missing, so it is optional.)
============================================================================
"""

# ---------------------------------------------------------------------------
# 0) Make sure required packages are present (works in Colab AND locally)
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
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
])

import cv2
import numpy as np
import matplotlib.pyplot as plt


def in_colab():
    """True if we are running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ===========================================================================
#  IMAGE PROCESSING FUNCTIONS  (used by BOTH Colab and local modes)
#  All functions assume RGB uint8 images, shape (H, W, 3).
# ===========================================================================

# ----- BASICS --------------------------------------------------------------

def compute_stats(image_rgb):
    """mean, std, min, max, mode on the grayscale version of the image."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    flat = gray.flatten()
    return {
        "mean": float(np.mean(flat)),
        "std":  float(np.std(flat)),
        "min":  int(np.min(flat)),
        "max":  int(np.max(flat)),
        # mode = most frequent pixel value (0..255)
        "mode": int(np.argmax(np.bincount(flat, minlength=256))),
    }


def compute_entropy(image_rgb):
    """Shannon entropy (bits) of the grayscale intensity distribution."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    prob = prob[prob > 0]            # drop empty bins (log(0) is undefined)
    return float(-np.sum(prob * np.log2(prob)))


def linear_transform(image_rgb, alpha=1.4, beta=15):
    """Linear point op: output = alpha*input + beta (contrast + brightness)."""
    return cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)


def equalize(image_rgb):
    """Histogram equalization on the luminance channel (keeps colors sane)."""
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def sobel_filter(image_rgb):
    """Sobel edge detection -> returned as a 3-channel image (not identity)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)


def draw_histogram_overlay(image_rgb, panel_w=320, panel_h=180, margin=12):
    """Live RGB histogram (3 lines, one panel) blended into the top-right."""
    h, w = image_rgb.shape[:2]
    panel = np.full((panel_h, panel_w, 3), 25, dtype=np.uint8)
    colors = [(255, 70, 70), (70, 255, 70), (90, 150, 255)]   # R, G, B
    for channel, color in enumerate(colors):
        hist = cv2.calcHist([image_rgb], [channel], None, [256], [0, 256]).flatten()
        cv2.normalize(hist, hist, 0, panel_h - 12, cv2.NORM_MINMAX)
        pts = np.array(
            [[int(x * panel_w / 256), panel_h - 1 - int(hist[x])] for x in range(256)],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(panel, [pts], False, color, 1)
    cv2.putText(panel, "RGB Histogram", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    x0, y0 = w - panel_w - margin, margin
    roi = image_rgb[y0:y0 + panel_h, x0:x0 + panel_w]
    image_rgb[y0:y0 + panel_h, x0:x0 + panel_w] = cv2.addWeighted(roi, 0.25, panel, 0.85, 0)
    return image_rgb


def draw_stats_overlay(image_rgb, source_rgb):
    """Write statistics (from the unprocessed source) in the top-left corner."""
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
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_rgb, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2, cv2.LINE_AA)
        y += 30
    return image_rgb


# ----- SPECIAL TASK: face landmarks (neural network) + overlay -------------

_face_mesh = None


def _get_face_mesh():
    """
    Load MediaPipe Face Mesh once and reuse it.

    On newer MediaPipe versions (the ones Colab installs by default),
    `mediapipe.solutions` is not exposed until you explicitly import the
    submodule first. We try both styles so the code works everywhere.
    """
    global _face_mesh
    if _face_mesh is None:
        try:
            # Newer MediaPipe: need to import the submodule explicitly
            from mediapipe.python.solutions import face_mesh as mp_face_mesh
        except ImportError:
            # Older MediaPipe: the classic attribute path works
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
        _face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _face_mesh


def _midpoint(a, b):
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _draw_sunglasses(frame, left_eye, right_eye):
    eye_dist = _dist(left_eye, right_eye)
    r = max(6, int(eye_dist * 0.42))
    cv2.circle(frame, left_eye, r, (12, 12, 12), -1)
    cv2.circle(frame, right_eye, r, (12, 12, 12), -1)
    cv2.line(frame, left_eye, right_eye, (12, 12, 12), max(3, r // 3))


def _draw_mustache(frame, left_eye, right_eye, nose_bottom):
    eye_dist = _dist(left_eye, right_eye)
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                  right_eye[0] - left_eye[0]))
    center = (nose_bottom[0], nose_bottom[1] + int(eye_dist * 0.12))
    axes = (max(8, int(eye_dist * 0.45)), max(4, int(eye_dist * 0.18)))
    cv2.ellipse(frame, center, axes, angle, 0, 360, (35, 25, 20), -1)


def apply_face_overlay(image_rgb):
    """
    SPECIAL TASK: run Face Mesh, draw sunglasses + mustache that follow the
    face position, scale and tilt. Returns the frame unchanged if no face found.
    """
    face_mesh = _get_face_mesh()
    h, w = image_rgb.shape[:2]
    results = face_mesh.process(image_rgb)      # MediaPipe wants RGB (we are RGB)
    if not results.multi_face_landmarks:
        return image_rgb
    for landmarks in results.multi_face_landmarks:
        def px(idx):
            lm = landmarks.landmark[idx]
            return (int(lm.x * w), int(lm.y * h))
        left_eye = _midpoint(px(33), px(133))
        right_eye = _midpoint(px(362), px(263))
        nose_bottom = px(2)
        _draw_sunglasses(image_rgb, left_eye, right_eye)
        _draw_mustache(image_rgb, left_eye, right_eye, nose_bottom)
    return image_rgb


# A single dispatcher so both modes can apply the same named effect ----------

def apply_effect(image_rgb, effect):
    """Apply one named effect to an RGB frame and return the result."""
    if effect == "face":
        return apply_face_overlay(image_rgb)
    if effect == "sobel":
        return sobel_filter(image_rgb)
    if effect == "equalize":
        return equalize(image_rgb)
    if effect == "linear":
        return linear_transform(image_rgb)
    if effect == "hist":
        return draw_histogram_overlay(image_rgb)
    if effect == "stats":
        return draw_stats_overlay(image_rgb, image_rgb.copy())
    return image_rgb   # "none"


# ===========================================================================
#  COLAB MODE  -  develop / debug everything without a virtual camera
# ===========================================================================

def _show_rgb(image_rgb, title=""):
    plt.figure(figsize=(7, 5))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def print_statistics(image_rgb):
    """Print per-channel + overall stats and entropy."""
    print("\n================ BASIC STATISTICS ================")
    for i, name in enumerate(("Red", "Green", "Blue")):
        ch = image_rgb[:, :, i].flatten()
        mode = int(np.argmax(np.bincount(ch, minlength=256)))
        print(f"{name:>5}: mean={ch.mean():6.2f}  std={ch.std():6.2f}  "
              f"min={ch.min():3d}  max={ch.max():3d}  mode={mode:3d}")
    s = compute_stats(image_rgb)
    e = compute_entropy(image_rgb)
    print("-" * 50)
    print(f"Overall (gray): mean={s['mean']:.2f}  std={s['std']:.2f}  "
          f"min={s['min']}  max={s['max']}  mode={s['mode']}")
    print(f"Shannon entropy: {e:.3f} bits")
    print("=" * 50)


def show_basics(image_rgb):
    """One matplotlib figure with all the visual basics (great for the report)."""
    equalized = equalize(image_rgb)
    edges = sobel_filter(image_rgb)
    transformed = linear_transform(image_rgb, alpha=1.4, beta=15)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Computer Vision - Basics", fontsize=15)

    ax[0, 0].imshow(image_rgb); ax[0, 0].set_title("Original"); ax[0, 0].axis("off")

    for i, c in enumerate(("red", "green", "blue")):   # 3 lines, one plot
        hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        ax[0, 1].plot(hist, color=c, label=c, linewidth=1)
    ax[0, 1].set_title("RGB Histogram"); ax[0, 1].set_xlim([0, 256]); ax[0, 1].legend()

    ax[0, 2].imshow(transformed); ax[0, 2].set_title("Linear transform"); ax[0, 2].axis("off")
    ax[1, 0].imshow(equalized);   ax[1, 0].set_title("Equalized");        ax[1, 0].axis("off")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.cvtColor(equalized, cv2.COLOR_RGB2GRAY)
    ax[1, 1].hist(gray.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.6, label="before")
    ax[1, 1].hist(gray_eq.ravel(), bins=256, range=(0, 256), color="orange", alpha=0.6, label="after")
    ax[1, 1].set_title("Equalization effect"); ax[1, 1].set_xlim([0, 256]); ax[1, 1].legend()

    ax[1, 2].imshow(edges); ax[1, 2].set_title("Sobel filter"); ax[1, 2].axis("off")
    plt.tight_layout(); plt.show()


def analyze_image(path):
    """Colab helper: full basics analysis on an image file (+ face overlay)."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print_statistics(rgb)
    show_basics(rgb)
    _show_rgb(apply_face_overlay(rgb.copy()), "Face overlay (special task)")


def webcam_face_demo():
    """Colab helper: snap one photo with your webcam (JS) and overlay the face."""
    if not in_colab():
        print("webcam_face_demo() is for Colab. Locally, just run main() for live mode.")
        return
    from IPython.display import display, Javascript
    from google.colab.output import eval_js
    from base64 import b64decode

    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const btn = document.createElement('button');
            btn.textContent = 'Capture';
            div.appendChild(btn);
            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
            await new Promise((resolve) => btn.onclick = resolve);
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
        }
        ''')
    display(js)
    data = eval_js('takePhoto(0.9)')
    binary = b64decode(data.split(',')[1])
    with open("selfie.jpg", "wb") as f:
        f.write(binary)

    bgr = cv2.imread("selfie.jpg")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _show_rgb(rgb, "Your photo")
    _show_rgb(apply_face_overlay(rgb.copy()), "With face overlay")


def process_video(in_path, out_path="output.mp4", effect="face", show_samples=True):
    """
    Colab helper: apply an effect to every frame of a video file and save the
    result. Effects: 'face', 'sobel', 'equalize', 'linear', 'hist', 'stats'.
    Saves out_path (download it to play) and shows a few sample frames inline.
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise FileNotFoundError(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    samples, frame_count = [], 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = apply_effect(rgb, effect)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))   # writer expects BGR
        if show_samples and frame_count % 30 == 0 and len(samples) < 3:
            samples.append(rgb.copy())
        frame_count += 1

    cap.release()
    writer.release()
    print(f"[video] processed {frame_count} frames -> saved '{out_path}' (download to play)")
    for i, s in enumerate(samples):
        _show_rgb(s, f"Sample frame {i + 1} (effect='{effect}')")


def run_colab_demo():
    """Runs automatically in Colab: instant basics demo + usage instructions."""
    print("Running in Google Colab -> demo/debug mode (no virtual camera here).\n")

    # Instant output so 'just run it' does something: basics on a sample image.
    yy, xx = np.mgrid[0:360, 0:640]
    sample = np.zeros((360, 640, 3), dtype=np.uint8)
    sample[..., 0] = (xx % 256)                       # R gradient
    sample[..., 1] = (yy % 256)                       # G gradient
    sample[..., 2] = ((xx + yy) % 256)                # B gradient
    print("Demo: basics on a generated sample image.")
    print_statistics(sample)
    show_basics(sample)

    print("""
Next steps (run these in a new cell):
  webcam_face_demo()                         # snap a selfie -> test the face overlay
  analyze_image('/content/your_photo.jpg')   # upload a photo, then analyse it
  process_video('/content/in.mp4', effect='face')   # effects: face/sobel/equalize/linear/hist/stats

To upload files in Colab:
  from google.colab import files; files.upload()
""")


# ===========================================================================
#  LOCAL LIVE MODE  -  webcam -> processing -> virtual camera (the real demo)
# ===========================================================================

_CONTROLS = """
[live] Controls (focus the 'Preview' window, then press a key):
   1 stats   2 histogram   3 equalize   4 sobel
   5 face overlay (SPECIAL, on by default)   6 linear   0 off   q quit
"""


def run_local_live(width=640, height=480, fps=30, device_id=0):
    """Full pipeline: webcam input -> effects -> OBS virtual camera output."""
    _ensure([("pyvirtualcam", "pyvirtualcam")])
    import pyvirtualcam

    cap = cv2.VideoCapture(device_id, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {device_id}.")

    state = {"stats": False, "hist": False, "equalize": False,
             "sobel": False, "face": True, "linear": False}
    print(_CONTROLS)

    try:
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, backend='obs') as cam:
            print(f"[live] Virtual camera running: {cam.device}")
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                out = source.copy()

                if state["equalize"]: out = equalize(out)
                if state["linear"]:   out = linear_transform(out)
                if state["sobel"]:    out = sobel_filter(out)
                if state["face"]:     out = apply_face_overlay(out)
                if state["hist"]:     out = draw_histogram_overlay(out)
                if state["stats"]:    out = draw_stats_overlay(out, source)

                if out.shape[1] != width or out.shape[0] != height:
                    out = cv2.resize(out, (width, height))

                cam.send(np.ascontiguousarray(out, dtype=np.uint8))
                cam.sleep_until_next_frame()

                cv2.imshow("Preview (q to quit)", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"): break
                elif key == ord("1"): state["stats"] = not state["stats"]
                elif key == ord("2"): state["hist"] = not state["hist"]
                elif key == ord("3"): state["equalize"] = not state["equalize"]
                elif key == ord("4"): state["sobel"] = not state["sobel"]
                elif key == ord("5"): state["face"] = not state["face"]
                elif key == ord("6"): state["linear"] = not state["linear"]
                elif key == ord("0"): state = {k: False for k in state}
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ===========================================================================
#  DISPATCH: pick the right mode for the current environment
# ===========================================================================

def main():
    if in_colab():
        run_colab_demo()
    else:
        run_local_live()


if __name__ == "__main__":
    main()
