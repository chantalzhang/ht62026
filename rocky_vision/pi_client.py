"""HTTP client for the servo-control server running on the Rocky Pi.

The Pi (hostname "qnxpi-rocky") runs an HTTP server with 6 endpoints for
driving the rock/paper/scissors servos: GET /rock, /paper, /scissors move
and hold a gesture, and GET /rock/reset, /paper/reset, /scissors/reset
return the servos to their starting position. Requests are fired off in a
background thread so a slow/unreachable Pi never blocks the game loop.
"""

import threading
import urllib.request

from rocky_vision.gesture_classifier import PAPER, ROCK, SCISSORS

PI_HOST = "qnxpi-rocky.local"
PI_PORT = 8080
BASE_URL = f"http://{PI_HOST}:{PI_PORT}"

REQUEST_TIMEOUT_SECONDS = 3

_GESTURE_PATHS = {ROCK: "rock", PAPER: "paper", SCISSORS: "scissors"}


def _get(path):
    url = f"{BASE_URL}/{path}"
    try:
        urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS)
    except Exception as exc:  # best-effort hardware call, never crash the game
        print(f"pi_client: request to {url} failed: {exc}")


def _get_async(path):
    threading.Thread(target=_get, args=(path,), daemon=True).start()


def send_move(move):
    """Tell the Pi to move the servo(s) for Rocky's gesture and hold it."""
    path = _GESTURE_PATHS.get(move)
    if path:
        _get_async(path)


def reset_move(move):
    """Tell the Pi to return the servo(s) for the given gesture to start."""
    path = _GESTURE_PATHS.get(move)
    if path:
        _get_async(f"{path}/reset")
