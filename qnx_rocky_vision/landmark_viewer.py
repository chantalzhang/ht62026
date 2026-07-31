import argparse
import mmap
import os
import struct
import time

import cv2
import mediapipe as mp
import numpy as np

from qnx_rocky_vision.gesture_classifier import UNKNOWN, classify_rps
from qnx_rocky_vision.rps import counter_move
from qnx_rocky_vision.stability import GestureStabilizer
from qnx_rocky_vision.ui import render_game


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class OpenCVCamera:
    """Generic USB/webcam camera source via OpenCV (dev/fallback use only)."""

    name = "opencv"

    def __init__(self, camera_index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

    def read_rgb(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


# --- QNX camera1 access via a shared-memory frame buffer -------------------
# QNX's Camera API (libcamapi.so, <camera/camera_api.h>) is a native C library
# with no Python bindings, and invoking a ctypes CFUNCTYPE callback from a
# thread libcamapi creates internally (not one Python spawned) reliably
# segfaults this interpreter -- a systemic ctypes/foreign-thread limitation
# (this repo used to work around it with a `camera_bridge` C subprocess
# piping frames over stdout; that approach is now retired in favor of this
# simpler one).
#
# Instead, a separate native process owns the entire camera_open/
# camera_start_viewfinder callback dance and writes each finished frame
# straight into a POSIX shared-memory segment. Python never touches libcamapi
# at all here -- it just mmaps that file and reads whichever frame is
# newest, so there's no ctypes/foreign-thread callback involved on the
# Python side whatsoever.
#
# Wire format written by the native producer, refreshed once per frame:
#   uint32_t frame_id      (bumped every frame; lets readers detect new ones)
#   uint32_t width
#   uint32_t height
#   uint32_t frametype
#   uint32_t data_size
#   uint8_t  pixel_data[data_size]   (RGBA/BGRA, YUYV, or NV12)
SHM_FILE = "/dev/shmem/qnx_camera_shm"
_SHM_HEADER_FIELDS = "IIIII"  # frame_id, width, height, frametype, data_size
_SHM_HEADER_SIZE = struct.calcsize(_SHM_HEADER_FIELDS)
# Matches the max NV12 buffer size the native producer allocates for
# (IMX708's full sensor resolution).
_SHM_MAX_FRAME_BYTES = 2304 * 1296 * 2
SHM_SIZE = _SHM_HEADER_SIZE + _SHM_MAX_FRAME_BYTES


def _decode_bgr(pixels, width, height, data_size):
    """Decode a shared-memory pixel payload to BGR, branching on data_size
    the same way the native producer's supported formats distinguish
    themselves (4 bytes/pixel, 2 bytes/pixel, or NV12's 1.5 bytes/pixel)."""
    if data_size == width * height * 4:
        return pixels.reshape((height, width, 4))[:, :, :3]
    if data_size == width * height * 2:
        return cv2.cvtColor(pixels.reshape((height, width, 2)), cv2.COLOR_YUV2BGR_YUYV)
    if data_size == int(width * height * 1.5):
        return cv2.cvtColor(pixels.reshape((int(height * 1.5), width)), cv2.COLOR_YUV2BGR_NV12)
    return None


class QnxCameraSource:
    """Raspberry Pi Camera Module 3 (camera1 / CAMERA_UNIT_1) on QNX, read
    from the shared-memory frame buffer written by the native camera
    producer process (see the module comment above for why)."""

    name = "qnx"

    def __init__(self, width: int = None, height: int = None):
        del width, height  # negotiated by the camera/BSP, not requested here.

        if not os.path.exists(SHM_FILE):
            raise RuntimeError(
                f"{SHM_FILE} not found. Is the native camera process running?"
            )

        self._file = open(SHM_FILE, "r+b")
        self._shm = mmap.mmap(self._file.fileno(), SHM_SIZE)
        self._last_frame_id = 0

    def read_rgb(self):
        # New frames land asynchronously from the native producer; briefly
        # wait for one instead of returning None right away.
        shm = self._shm
        for _ in range(500):
            frame_id = struct.unpack("I", shm[:4])[0]
            if frame_id != self._last_frame_id:
                self._last_frame_id = frame_id
                _, width, height, _frametype, data_size = struct.unpack(
                    _SHM_HEADER_FIELDS, shm[:_SHM_HEADER_SIZE]
                )
                if width > 0 and height > 0 and data_size > 0:
                    raw = shm[_SHM_HEADER_SIZE : _SHM_HEADER_SIZE + data_size]
                    pixels = np.frombuffer(raw, dtype=np.uint8)
                    bgr = _decode_bgr(pixels, width, height, data_size)
                    if bgr is not None:
                        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            time.sleep(0.001)
        return None

    def close(self):
        self._shm.close()
        self._file.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open the QNX camera1 feed and draw MediaPipe hand landmarks."
    )
    parser.add_argument(
        "--source",
        choices=("qnx", "opencv"),
        default="qnx",
        help="Camera source. qnx reads camera1 frames from shared memory; opencv is a dev/fallback path.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-detection-confidence", type=float, default=0.6)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--stable-frames", type=int, default=3)
    return parser.parse_args()


def open_camera(args):
    if args.source == "opencv":
        return OpenCVCamera(args.camera_index, args.width, args.height)

    return QnxCameraSource(args.width, args.height)


def main():
    args = parse_args()
    camera = open_camera(args)
    print(f"Using camera source: {camera.name} at {args.width}x{args.height}")

    prev_time = time.perf_counter()
    fps = 0.0
    last_locked = None
    armed = False
    rocky_score = 0
    locked_frame = None
    taunts = [
        "Rocky wins again.",
        "You can't beat Rocky.",
        "Too slow.",
        "Rocky remains undefeated.",
    ]
    taunt = taunts[0]
    stabilizer = GestureStabilizer(args.stable_frames)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,  # faster/lighter; good for Pi-class hardware
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        while True:
            rgb = camera.read_rgb()
            if rgb is None:
                print("Failed to read camera frame")
                break

            # Mirror for a more natural demo view.
            rgb = cv2.flip(rgb, 1)

            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # OpenCV display/drawing uses BGR.
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            gesture = UNKNOWN
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                gesture = classify_rps(hand_landmarks)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            locked = stabilizer.locked
            if armed and not locked:
                locked = stabilizer.update(gesture)
                if locked and locked != last_locked:
                    rocky_score += 1
                    locked_frame = frame.copy()
                    taunt = taunts[(rocky_score - 1) % len(taunts)]
                    print(f"Locked: {locked} -> Rocky: {counter_move(locked)}")
                    last_locked = locked

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)
            prev_time = now

            state = "LOCKED" if locked else "ARMED" if armed else "WAITING"
            stable_text = "-" if not armed else "LOCKED" if locked else f"{stabilizer.count}/{args.stable_frames}"
            display_frame = locked_frame if locked_frame is not None else frame
            ui = render_game(
                display_frame,
                state=state,
                gesture=gesture,
                stable=stable_text,
                locked=locked,
                rocky_move=counter_move(locked),
                score=rocky_score,
                fps=fps,
                camera_name=camera.name,
                taunt=taunt,
            )

            cv2.imshow("Rocky RPS", ui)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                armed = True
                stabilizer.reset()
                last_locked = None
                locked_frame = None
                print("Armed")
            elif key == ord("r"):
                armed = False
                stabilizer.reset()
                last_locked = None
                locked_frame = None
                print("Reset")
            elif key == ord("c"):
                rocky_score = 0
                print("Score cleared")
            elif key == ord("q"):
                break

    camera.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
