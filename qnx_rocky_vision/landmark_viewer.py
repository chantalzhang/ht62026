import argparse
import struct
import subprocess
import threading
import time
from pathlib import Path

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


# --- QNX camera1 access via the camera_bridge native helper subprocess ---
# QNX's Camera API (libcamapi.so, <camera/camera_api.h>) is a native C library
# with no Python bindings. Talking to it directly via ctypes doesn't work on
# this QNX Python build: camera_start_viewfinder() invokes our callback from a
# thread libcamapi creates internally (not one Python spawned), and invoking
# any ctypes CFUNCTYPE callback from such a "foreign" thread reliably
# segfaults this interpreter -- a systemic ctypes/threading limitation, not a
# bug in any particular binding (confirmed independently of the camera
# entirely: even a plain libc pthread_create() thread calling a trivial
# ctypes callback crashes the same way).
#
# So instead, QnxCameraSource launches camera_bridge (built from
# camera_bridge.c, see that file for the wire protocol) as a subprocess.
# camera_bridge does the whole camera_open/camera_start_viewfinder dance and
# all pixel-format conversion in plain C -- which has no such limitation --
# and streams ready-to-use RGB24 frames to its own stdout. Python only ever
# reads bytes from an ordinary pipe, so no native callback ever runs inside
# the Python interpreter.
_BRIDGE_PATH = Path(__file__).resolve().parent / "camera_bridge"
_BRIDGE_MAGIC = b"QCF1"
_BRIDGE_HEADER = struct.Struct("<4sII")  # magic, height, width


def _read_exact(fileobj, size):
    """Read exactly `size` bytes from a blocking pipe, or None on EOF."""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = fileobj.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class QnxCameraSource:
    """Raspberry Pi Camera Module 3 (camera1 / CAMERA_UNIT_1) via QNX's Sensor
    Framework, accessed through the camera_bridge native helper subprocess
    (see camera_bridge.c for why this indirection is necessary).
    """

    name = "qnx"

    def __init__(self, width: int = None, height: int = None):
        del width, height  # negotiated by the camera/BSP, not requested here.

        if not _BRIDGE_PATH.exists():
            raise RuntimeError(
                f"{_BRIDGE_PATH} not found. Build it on the QNX target first:\n"
                f"  gcc -o {_BRIDGE_PATH} {_BRIDGE_PATH}.c -lcamapi"
            )

        self._proc = subprocess.Popen(
            [str(_BRIDGE_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )

        self._lock = threading.Lock()
        self._latest_rgb = None
        self._stopped = False

        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        stdout = self._proc.stdout
        while not self._stopped:
            header = _read_exact(stdout, _BRIDGE_HEADER.size)
            if header is None:
                break
            magic, height, width = _BRIDGE_HEADER.unpack(header)
            if magic != _BRIDGE_MAGIC:
                break
            payload = _read_exact(stdout, height * width * 3)
            if payload is None:
                break
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
            with self._lock:
                self._latest_rgb = rgb

    def read_rgb(self):
        # Frames arrive asynchronously on the reader thread; briefly wait for
        # one instead of returning None right away.
        for _ in range(50):
            with self._lock:
                rgb = self._latest_rgb
            if rgb is not None:
                return rgb
            if self._proc.poll() is not None:
                return None  # camera_bridge exited; nothing more to wait for.
            time.sleep(0.01)
        return None

    def close(self):
        self._stopped = True
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._reader_thread.join(timeout=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open the QNX camera1 feed and draw MediaPipe hand landmarks."
    )
    parser.add_argument(
        "--source",
        choices=("qnx", "opencv"),
        default="qnx",
        help="Camera source. qnx uses camera1 via the camera_bridge helper; opencv is a dev/fallback path.",
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
