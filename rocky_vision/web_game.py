import argparse
import json
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import mediapipe as mp

from rocky_vision.gesture_classifier import UNKNOWN, classify_rps
from rocky_vision.landmark_viewer import OpenCVCamera, PiCamera2Source
from rocky_vision.rps import counter_move, result_text
from rocky_vision.stability import GestureStabilizer


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class Game:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.stabilizer = GestureStabilizer(args.stable_frames)
        self.armed = False
        self.countdown_start = None
        self.score = 0
        self.gesture = UNKNOWN
        self.frame_jpg = None
        self.locked_frame_jpg = None
        self.lock_id = 0
        self.running = True

    def state(self):
        with self.lock:
            locked = self.stabilizer.locked
            rocky = counter_move(locked)
            countdown_word = self.countdown_word()
            stable = "LOCKED" if locked else f"{self.stabilizer.count}/{self.args.stable_frames}"
            return {
                "state": "LOCKED" if locked else "COUNTDOWN" if countdown_word else "ARMED" if self.armed else "WAITING",
                "countdown_word": countdown_word,
                "gesture": self.gesture,
                "stable": stable,
                "locked": locked,
                "rocky_move": rocky,
                "result": result_text(locked, rocky),
                "score": self.score,
                "lock_id": self.lock_id,
            }

    def countdown_word(self):
        if self.countdown_start is None:
            return None
        elapsed = time.monotonic() - self.countdown_start
        words = ["ROCK", "PAPER", "SCISSORS"]
        index = int(elapsed / self.args.countdown_step)
        return words[index] if index < len(words) else None

    def update_countdown(self):
        if self.countdown_start is None:
            return
        if time.monotonic() - self.countdown_start >= self.args.countdown_step * 3:
            self.countdown_start = None
            self.armed = True
            self.stabilizer.reset()

    def control(self, key):
        with self.lock:
            if key == "space":
                self.armed = False
                self.countdown_start = time.monotonic()
                self.locked_frame_jpg = None
                self.stabilizer.reset()
            elif key == "r":
                self.armed = False
                self.countdown_start = None
                self.locked_frame_jpg = None
                self.stabilizer.reset()
            elif key == "c":
                self.score = 0
            elif key == "q":
                self.running = False

    def run_vision(self):
        camera = PiCamera2Source(self.args.width, self.args.height) if self.args.source == "picamera2" else OpenCVCamera(self.args.camera_index, self.args.width, self.args.height)
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=self.args.min_detection_confidence,
            min_tracking_confidence=self.args.min_tracking_confidence,
        ) as hands:
            while self.running:
                rgb = camera.read_rgb()
                if rgb is None:
                    continue
                rgb = cv2.flip(rgb, 1)
                results = hands.process(rgb)
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                gesture = UNKNOWN
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    gesture = classify_rps(hand)
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                ok, jpg = cv2.imencode(".jpg", frame)
                frame_bytes = jpg.tobytes() if ok else None

                with self.lock:
                    self.gesture = gesture
                    self.update_countdown()
                    was_locked = self.stabilizer.locked
                    if self.armed and not was_locked:
                        locked = self.stabilizer.update(gesture)
                        if locked:
                            self.score += 1
                            self.lock_id += 1
                            self.locked_frame_jpg = frame_bytes
                    if frame_bytes:
                        self.frame_jpg = frame_bytes

        camera.close()


def make_handler(game):
    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            path = urlparse(path).path
            if path == "/":
                path = "/game.html"
            return str(FRONTEND / path.lstrip("/"))

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/state":
                data = json.dumps(game.state()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if parsed.path == "/control":
                key = parse_qs(parsed.query).get("key", [""])[0]
                game.control(key)
                self.send_response(204)
                self.end_headers()
                return

            if parsed.path == "/locked_frame":
                with game.lock:
                    frame = game.locked_frame_jpg
                if not frame:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
                return

            if parsed.path == "/video":
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    while game.running:
                        with game.lock:
                            frame = game.frame_jpg
                        if frame:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                        time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            return super().do_GET()

        def log_message(self, format, *args):
            pass

    return Handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("opencv", "picamera2"), default="opencv")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--countdown-step", type=float, default=0.74)
    parser.add_argument("--min-detection-confidence", type=float, default=0.6)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    game = Game(args)
    threading.Thread(target=game.run_vision, daemon=True).start()
    server = ThreadingHTTPServer(("localhost", args.port), make_handler(game))
    url = f"http://localhost:{args.port}"
    print(f"Open {url}")
    webbrowser.open(url)
    while game.running:
        server.handle_request()


if __name__ == "__main__":
    main()
