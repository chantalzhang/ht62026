import argparse
import ctypes
import struct
import threading
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


# --- QNX Sensor Framework (libcamapi) bindings for camera1 / CAMERA_UNIT_1 ---
# QNX's Camera API (libcamapi.so, <camera/camera_api.h>) is a native C library
# with no Python bindings, so QnxCameraSource talks to it directly via ctypes
# in callback mode (camera_start_viewfinder), which needs no Screen/window
# setup. Constants below were confirmed against the headers shipped by the
# qnx-sensor-framework-dev apk package (QNX 8.0.4, Raspberry Pi 5 image).
CAMERA_UNIT_1 = 1
CAMERA_HANDLE_INVALID = -1
CAMERA_EOK = 0
CAMERA_MODE_PREAD = 1 << 0
CAMERA_MODE_PWRITE = 1 << 1
CAMERA_MODE_DREAD = 1 << 2
CAMERA_MODE_RO = CAMERA_MODE_PREAD | CAMERA_MODE_DREAD
# camera_set_vf_mode() changes camera configuration, which needs config-write
# access (PWRITE) even though we only ever read image data (DREAD, no DWRITE).
CAMERA_MODE_VIEWFINDER = CAMERA_MODE_PREAD | CAMERA_MODE_PWRITE | CAMERA_MODE_DREAD
CAMERA_MAX_FRAMEDESC_SIZE = 256

CAMERA_FRAMETYPE_RGB8888 = 2
CAMERA_FRAMETYPE_CBYCRY = 8
CAMERA_FRAMETYPE_YCBYCR = 14
CAMERA_FRAMETYPE_BGR8888 = 31

# camera_vfmode_t: a viewfinder mode must be selected with camera_set_vf_mode()
# before camera_start_viewfinder() is called, or the datapath is left
# unconfigured (this reliably segfaults inside camera_start_viewfinder).
CAMERA_VFMODE_DEFAULT = 0
CAMERA_VFMODE_VIDEO = 1

_camapi = None


def _load_camapi():
    """Lazily load libcamapi.so and bind the handful of functions we need."""
    global _camapi
    if _camapi is not None:
        return _camapi

    lib = ctypes.CDLL("libcamapi.so")

    class camera_buffer_t(ctypes.Structure):
        _fields_ = [
            ("frametype", ctypes.c_int32),
            ("framesize", ctypes.c_uint64),
            ("framebuf", ctypes.POINTER(ctypes.c_uint8)),
            ("framemetasize", ctypes.c_uint64),
            ("framemeta", ctypes.c_void_p),
            ("frametimestamp", ctypes.c_int64),
            ("framedesc", ctypes.c_uint8 * CAMERA_MAX_FRAMEDESC_SIZE),
        ]

    viewfinder_cb_t = ctypes.CFUNCTYPE(
        None, ctypes.c_int32, ctypes.POINTER(camera_buffer_t), ctypes.c_void_p
    )
    status_cb_t = ctypes.CFUNCTYPE(
        None, ctypes.c_int32, ctypes.c_int32, ctypes.c_uint16, ctypes.c_void_p
    )

    lib.camera_open.argtypes = [ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
    lib.camera_open.restype = ctypes.c_int32
    lib.camera_close.argtypes = [ctypes.c_int32]
    lib.camera_close.restype = ctypes.c_int32
    lib.camera_set_vf_mode.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.camera_set_vf_mode.restype = ctypes.c_int32
    lib.camera_get_supported_vf_modes.argtypes = [
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.camera_get_supported_vf_modes.restype = ctypes.c_int32
    lib.camera_start_viewfinder.argtypes = [ctypes.c_int32, viewfinder_cb_t, status_cb_t, ctypes.c_void_p]
    lib.camera_start_viewfinder.restype = ctypes.c_int32
    lib.camera_stop_viewfinder.argtypes = [ctypes.c_int32]
    lib.camera_stop_viewfinder.restype = ctypes.c_int32

    _camapi = {
        "lib": lib,
        "camera_buffer_t": camera_buffer_t,
        "viewfinder_cb_t": viewfinder_cb_t,
        "status_cb_t": status_cb_t,
    }
    return _camapi


def get_supported_vf_modes(handle: int):
    """Debug helper: return the list of camera_vfmode_t values this camera
    handle actually supports, per camera_get_supported_vf_modes()."""
    camapi = _load_camapi()
    lib = camapi["lib"]

    num_supported = ctypes.c_uint32(0)
    err = lib.camera_get_supported_vf_modes(handle, 0, ctypes.byref(num_supported), None)
    if err != CAMERA_EOK:
        raise RuntimeError(f"camera_get_supported_vf_modes(presize) failed: err={err}")

    count = num_supported.value
    modes = (ctypes.c_int32 * count)()
    err = lib.camera_get_supported_vf_modes(handle, count, ctypes.byref(num_supported), modes)
    if err != CAMERA_EOK:
        raise RuntimeError(f"camera_get_supported_vf_modes failed: err={err}")

    return list(modes[: num_supported.value])


def _yuv422_to_rgb(y0, cb, y1, cr):
    """Convert BT.601-style 4:2:2 luma/chroma planes (as pixel pairs) to RGB."""

    def _to_rgb(y, cb, cr):
        y = y.astype(np.float32)
        cb = cb.astype(np.float32) - 128.0
        cr = cr.astype(np.float32) - 128.0
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        return np.stack([r, g, b], axis=-1)

    rgb_even = _to_rgb(y0, cb, cr)
    rgb_odd = _to_rgb(y1, cb, cr)
    height, half_width, _ = rgb_even.shape
    out = np.empty((height, half_width * 2, 3), dtype=np.float32)
    out[:, 0::2, :] = rgb_even
    out[:, 1::2, :] = rgb_odd
    return np.clip(out, 0, 255).astype(np.uint8)


def _qnx_frame_to_rgb(buf):
    """Convert a camera_buffer_t (RGB8888/BGR8888/YCbYCr/CbYCrY) to an RGB ndarray."""
    frametype = buf.frametype
    height, width, stride = struct.unpack_from("<III", bytes(buf.framedesc))
    if not height or not width or not stride:
        return None

    if frametype in (CAMERA_FRAMETYPE_RGB8888, CAMERA_FRAMETYPE_BGR8888):
        data = ctypes.string_at(buf.framebuf, height * stride)
        rows = np.frombuffer(data, dtype=np.uint8).reshape(height, stride)
        pixels = rows[:, : width * 4].reshape(height, width, 4)
        if frametype == CAMERA_FRAMETYPE_RGB8888:
            return pixels[:, :, :3].copy()
        return pixels[:, :, 2::-1].copy()  # BGR8888 stores B,G,R,X -> reverse to R,G,B

    if frametype in (CAMERA_FRAMETYPE_YCBYCR, CAMERA_FRAMETYPE_CBYCRY):
        data = ctypes.string_at(buf.framebuf, height * stride)
        rows = np.frombuffer(data, dtype=np.uint8).reshape(height, stride)
        macropixels = rows[:, : width * 2].reshape(height, width // 2, 4)
        if frametype == CAMERA_FRAMETYPE_YCBYCR:
            y0, cb, y1, cr = (macropixels[..., i] for i in range(4))
        else:
            cb, y0, cr, y1 = (macropixels[..., i] for i in range(4))
        return _yuv422_to_rgb(y0, cb, y1, cr)

    return None


class QnxCameraSource:
    """Raspberry Pi Camera Module 3 (camera1 / CAMERA_UNIT_1) via QNX's Sensor
    Framework, accessed directly through libcamapi with ctypes (callback mode).
    """

    name = "qnx"

    def __init__(self, width: int = None, height: int = None):
        camapi = _load_camapi()
        self._lib = camapi["lib"]

        handle = ctypes.c_int32(CAMERA_HANDLE_INVALID)
        err = self._lib.camera_open(CAMERA_UNIT_1, CAMERA_MODE_VIEWFINDER, ctypes.byref(handle))
        if err != CAMERA_EOK or handle.value == CAMERA_HANDLE_INVALID:
            raise RuntimeError(f"camera_open(CAMERA_UNIT_1) failed: err={err}")
        self._handle = handle.value

        # Must select a viewfinder mode before starting the viewfinder, otherwise
        # the datapath is left unconfigured and camera_start_viewfinder segfaults.
        # This camera/BSP doesn't necessarily support the standard camera_vfmode_t
        # values (e.g. CAMERA_VFMODE_VIDEO) -- some Raspberry Pi camera BSPs add
        # their own vendor-specific mode values -- so pick whichever non-default
        # mode camera_get_supported_vf_modes() actually reports instead of
        # hardcoding one.
        supported_modes = get_supported_vf_modes(self._handle)
        vf_mode = next((m for m in supported_modes if m != CAMERA_VFMODE_DEFAULT), None)
        if vf_mode is None:
            self._lib.camera_close(self._handle)
            raise RuntimeError(
                f"Camera reports no non-default viewfinder mode (supported={supported_modes})"
            )

        err = self._lib.camera_set_vf_mode(self._handle, vf_mode)
        if err != CAMERA_EOK:
            self._lib.camera_close(self._handle)
            raise RuntimeError(f"camera_set_vf_mode({vf_mode}) failed: err={err}")

        self._lock = threading.Lock()
        self._latest_rgb = None

        def _on_frame(_handle, buf_ptr, _arg):
            try:
                rgb = _qnx_frame_to_rgb(buf_ptr.contents)
            except Exception:
                rgb = None
            if rgb is not None:
                with self._lock:
                    self._latest_rgb = rgb

        def _on_status(_handle, _status, _ext_status, _arg):
            pass

        # Keep references on self so the CFUNCTYPE closures outlive camera_start_viewfinder.
        # ctypes on this build won't implicitly convert a bare None into a NULL function
        # pointer for a CFUNCTYPE argtype, so we pass real no-op callbacks instead of None.
        self._viewfinder_cb = camapi["viewfinder_cb_t"](_on_frame)
        self._status_cb = camapi["status_cb_t"](_on_status)

        err = self._lib.camera_start_viewfinder(self._handle, self._viewfinder_cb, self._status_cb, None)
        if err != CAMERA_EOK:
            self._lib.camera_close(self._handle)
            raise RuntimeError(f"camera_start_viewfinder(CAMERA_UNIT_1) failed: err={err}")

    def read_rgb(self):
        # Frames arrive asynchronously on the library's own callback thread;
        # briefly wait for the first one instead of returning None right away.
        for _ in range(50):
            with self._lock:
                rgb = self._latest_rgb
            if rgb is not None:
                return rgb
            time.sleep(0.01)
        return None

    def close(self):
        self._lib.camera_stop_viewfinder(self._handle)
        self._lib.camera_close(self._handle)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open the QNX camera1 feed and draw MediaPipe hand landmarks."
    )
    parser.add_argument(
        "--source",
        choices=("qnx", "opencv"),
        default="qnx",
        help="Camera source. qnx uses camera1 via libcamapi; opencv is a dev/fallback path.",
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
