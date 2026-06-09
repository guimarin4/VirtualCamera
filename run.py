# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
import argparse
import urllib.request
from pathlib import Path

import keyboard # pip install keyboard
import numpy as np
import cv2

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import histogram_figure_numba

MODEL_DIR = Path(__file__).resolve().parent / 'models'
FACE_PROTO = MODEL_DIR / 'deploy.prototxt'
FACE_MODEL = MODEL_DIR / 'res10_300x300_ssd_iter_140000.caffemodel'
FACE_PROTO_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
FACE_MODEL_URL = 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'


def download_file(url, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading model file to {target_path} ...')
    urllib.request.urlretrieve(url, str(target_path))
    print('Download complete.')


def load_face_detector():
    try:
        if not FACE_PROTO.exists():
            download_file(FACE_PROTO_URL, FACE_PROTO)
        if not FACE_MODEL.exists():
            download_file(FACE_MODEL_URL, FACE_MODEL)

        net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_MODEL))
        return net
    except Exception as exc:
        print('Warning: face detector could not be loaded:', exc)
        return None


def compute_channel_mode(channel):
    values, counts = np.unique(channel.ravel(), return_counts=True)
    if counts.size == 0:
        return 0
    return int(values[np.argmax(counts)])


def compute_image_statistics(image):
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    min_val = np.min(image, axis=(0, 1))
    max_val = np.max(image, axis=(0, 1))
    mode = np.array([compute_channel_mode(image[:, :, i]) for i in range(3)], dtype=np.int32)
    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'mode': mode,
    }


def compute_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob))) if prob.size > 0 else 0.0


def apply_linear_transformation(image, alpha=1.1, beta=20):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def equalize_color_image(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_advanced_filter(image):
    blurred = cv2.GaussianBlur(image, (7, 7), 1.5)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(blurred, 0.7, edges_color, 0.3, 0)


def create_replacement_patch(width, height):
    patch = np.full((height, width, 3), (30, 30, 80), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = max(10, min(width, height) // 3)
    cv2.circle(patch, center, radius, (255, 230, 0), -1)
    eye_y = center[1] - radius // 3
    eye_x = center[0] - radius // 2
    eye_radius = max(3, radius // 10)
    cv2.circle(patch, (eye_x, eye_y), eye_radius, (60, 60, 60), -1)
    cv2.circle(patch, (eye_x + radius, eye_y), eye_radius, (60, 60, 60), -1)
    smile_center = (center[0], center[1] + radius // 6)
    cv2.ellipse(patch, smile_center, (radius // 2, radius // 4), 0, 15, 165, (60, 60, 60), 4)
    return patch


def replace_faces_with_patch(image, net, confidence_threshold=0.6):
    if net is None:
        return image

    h, w = image.shape[:2]
    bgr_image = image[..., ::-1]
    blob = cv2.dnn.blobFromImage(bgr_image, 1.0, (300, 300), [104.0, 117.0, 123.0], False, False)
    net.setInput(blob)
    detections = net.forward()

    output = image.copy()
    for i in range(detections.shape[2]):
        score = float(detections[0, 0, i, 2])
        if score < confidence_threshold:
            continue

        left = int(detections[0, 0, i, 3] * w)
        top = int(detections[0, 0, i, 4] * h)
        right = int(detections[0, 0, i, 5] * w)
        bottom = int(detections[0, 0, i, 6] * h)

        left = max(0, left)
        top = max(0, top)
        right = min(w - 1, right)
        bottom = min(h - 1, bottom)

        if right - left < 20 or bottom - top < 20:
            continue

        patch = create_replacement_patch(right - left, bottom - top)
        roi = output[top:bottom, left:right]
        blended = cv2.addWeighted(patch, 0.8, roi, 0.2, 0)
        output[top:bottom, left:right] = blended

    return output


def custom_processing(img_source_generator):
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    face_detector = None
    face_detector_available = True

    toggle_state = { 'e': False, 'f': False, 'b': False }
    active = { 'equalize': False, 'special': True, 'filter': True }

    for sequence in img_source_generator:
        statistics = compute_image_statistics(sequence)
        entropy_value = compute_entropy(sequence)

        for key in toggle_state:
            pressed = keyboard.is_pressed(key)
            if pressed and not toggle_state[key]:
                if key == 'e':
                    active['equalize'] = not active['equalize']
                elif key == 'f':
                    active['special'] = not active['special']
                elif key == 'b':
                    active['filter'] = not active['filter']
            toggle_state[key] = pressed

        processed = apply_linear_transformation(sequence)

        if active['equalize']:
            processed = equalize_color_image(processed)

        if active['filter']:
            processed = apply_advanced_filter(processed)

        if active['special']:
            if face_detector is None and face_detector_available:
                face_detector = load_face_detector()
                if face_detector is None:
                    face_detector_available = False

            if face_detector is not None:
                processed = replace_faces_with_patch(processed, face_detector)

        r_bars, g_bars, b_bars = histogram_figure_numba(processed)
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        processed = plot_overlay_to_image(processed, fig)

        display_text_arr = [
            f'Mean RGB: {statistics["mean"][0]:.1f}, {statistics["mean"][1]:.1f}, {statistics["mean"][2]:.1f}',
            f'Mode RGB: {statistics["mode"][0]}, {statistics["mode"][1]}, {statistics["mode"][2]}',
            f'Std RGB: {statistics["std"][0]:.1f}, {statistics["std"][1]:.1f}, {statistics["std"][2]:.1f}',
            f'Entropy: {entropy_value:.3f}',
            f'[E]qualize: {"ON" if active["equalize"] else "OFF"}  [B]lur/Edge: {"ON" if active["filter"] else "OFF"}',
            f'[F]ace replace: {"ON" if active["special"] else "OFF"}',
        ]
        processed = plot_strings_to_image(processed, display_text_arr, text_color=(255, 255, 255), right_space=460)

        yield processed


def main():
    width = 1280
    height = 720
    fps = 30

    parser = argparse.ArgumentParser(description='VirtualCamera project runner')
    parser.add_argument('--source', choices=['camera', 'screen'], default='camera', help='Use camera or screen capture')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device index')
    args = parser.parse_args()

    vc = VirtualCamera(fps, width, height)

    if args.source == 'screen':
        source_generator = vc.capture_screen()
    else:
        source_generator = vc.capture_cv_video(args.camera_id, bgr_to_rgb=True)

    try:
        vc.virtual_cam_interaction(custom_processing(source_generator))
    except RuntimeError as exc:
        print('Camera capture failed:', exc)
        if args.source == 'camera':
            print('Falling back to screen capture. Use --source screen if you want screen input explicitly.')
            vc.virtual_cam_interaction(custom_processing(vc.capture_screen()))
        else:
            raise

if __name__ == '__main__':
    main()
