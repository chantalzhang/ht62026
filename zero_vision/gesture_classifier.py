from math import sqrt


ROCK = "rock"
PAPER = "paper"
SCISSORS = "scissors"
UNKNOWN = "unknown"


FINGERS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


def _distance(a, b):
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _is_extended(landmarks, tip_idx, pip_idx):
    wrist = landmarks[0]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    return _distance(wrist, tip) > _distance(wrist, pip) * 1.05


def classify_rps(hand_landmarks):
    """Classify MediaPipe hand landmarks as rock/paper/scissors/unknown."""
    if hand_landmarks is None:
        return UNKNOWN

    landmarks = hand_landmarks.landmark
    extended = {
        finger: _is_extended(landmarks, tip_idx, pip_idx)
        for finger, (tip_idx, pip_idx) in FINGERS.items()
    }

    index = extended["index"]
    middle = extended["middle"]
    ring = extended["ring"]
    pinky = extended["pinky"]

    if not any(extended.values()):
        return ROCK
    if all(extended.values()):
        return PAPER
    if index and middle and not ring and not pinky:
        return SCISSORS
    return UNKNOWN
