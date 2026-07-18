import argparse
import time

import cv2
import mediapipe as mp

from zero_vision.gesture_classifier import UNKNOWN, classify_rps
from zero_vision.rps import counter_move
from zero_vision.stability import GestureStabilizer


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class OpenCVCamera:
    """Generic USB/webcam camera source via OpenCV."""

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


class PiCamera2Source:
    """Raspberry Pi Camera Module source via Picamera2/libcamera."""

    name = "picamera2"

    def __init__(self, width: int, height: int):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is not installed. On Raspberry Pi OS, run: "
                "sudo apt install -y python3-picamera2"
            ) from exc

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=4,
        )
        self.picam2.configure(config)
        self.picam2.start()
        # Give auto-exposure/autofocus a brief moment to settle.
        time.sleep(0.5)

    def read_rgb(self):
        return self.picam2.capture_array()

    def close(self):
        self.picam2.stop()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open a camera feed and draw MediaPipe hand landmarks."
    )
    parser.add_argument(
        "--source",
        choices=("auto", "picamera2", "opencv"),
        default="auto",
        help="Camera source. Use picamera2 for Raspberry Pi Camera Module 3.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-detection-confidence", type=float, default=0.6)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--stable-frames", type=int, default=3)
    return parser.parse_args()


def open_camera(args):
    if args.source == "picamera2":
        return PiCamera2Source(args.width, args.height)

    if args.source == "opencv":
        return OpenCVCamera(args.camera_index, args.width, args.height)

    # Auto mode prefers the actual Raspberry Pi camera path, then falls back to webcam.
    try:
        return PiCamera2Source(args.width, args.height)
    except RuntimeError as pi_error:
        print(f"Picamera2 unavailable, falling back to OpenCV: {pi_error}")
        return OpenCVCamera(args.camera_index, args.width, args.height)


def main():
    args = parse_args()
    camera = open_camera(args)
    print(f"Using camera source: {camera.name} at {args.width}x{args.height}")

    prev_time = time.perf_counter()
    fps = 0.0
    last_locked = None
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

            locked = stabilizer.update(gesture)
            if locked and locked != last_locked:
                print(f"Locked: {locked} -> Zero: {counter_move(locked)}")
                last_locked = locked

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)
            prev_time = now

            cv2.putText(
                frame,
                f"{camera.name} | MediaPipe Hands | FPS: {fps:.1f}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            stable_text = "LOCKED" if locked else f"{stabilizer.count}/{args.stable_frames}"
            cv2.putText(
                frame,
                f"Gesture: {gesture.upper()}  Stable: {stable_text}",
                (16, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Locked: {(locked or '-').upper()}  Zero: {(counter_move(locked) or '-').upper()}",
                (16, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 128, 255) if locked else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Press r to reset, q to quit",
                (16, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Zero hand landmark viewer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                stabilizer.reset()
                last_locked = None
                print("Reset")
            elif key == ord("q"):
                break

    camera.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
