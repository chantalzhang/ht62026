from math import acos, degrees, sqrt


ROCK = "rock"
PAPER = "paper"
SCISSORS = "scissors"
UNKNOWN = "unknown"


FINGERS = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


OPEN_ANGLE = 130
OPEN_RATIO = 1.05
STARTING_ANGLE = 75
STARTING_RATIO = 0.96
SCISSORS_RELATIVE_GAP = 8
FIST_THUMB_MAX_RATIO = 0.85
FIST_THUMB_WRIST_MAX_RATIO = 1.6


def _distance(a, b):
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _joint_angle(a, b, c):
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    denom = _distance(a, b) * _distance(c, b)
    if denom == 0:
        return 0

    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / denom
    cosine = max(-1, min(1, cosine))
    return degrees(acos(cosine))


def _finger_features(landmarks):
    wrist = landmarks[0]
    features = {}

    for finger, (mcp_idx, pip_idx, tip_idx) in FINGERS.items():
        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]
        tip = landmarks[tip_idx]
        pip_distance = _distance(wrist, pip)
        ratio = _distance(wrist, tip) / pip_distance if pip_distance else 0
        angle = _joint_angle(mcp, pip, tip)

        features[finger] = {
            "angle": angle,
            "ratio": ratio,
            "open": angle >= OPEN_ANGLE or ratio >= OPEN_RATIO,
            "starting": angle >= STARTING_ANGLE and ratio >= STARTING_RATIO,
        }

    return features


def _thumb_is_fist_like(landmarks):
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    middle_mcp = landmarks[9]

    palm_size = max(_distance(wrist, middle_mcp), 1e-6)
    thumb_to_fist = min(
        _distance(thumb_tip, landmarks[5]),
        _distance(thumb_tip, landmarks[9]),
        _distance(thumb_tip, landmarks[8]),
        _distance(thumb_tip, landmarks[12]),
        _distance(thumb_tip, landmarks[16]),
        _distance(thumb_tip, landmarks[20]),
    )

    return (
        thumb_to_fist <= palm_size * FIST_THUMB_MAX_RATIO
        and _distance(wrist, thumb_tip) <= palm_size * FIST_THUMB_WRIST_MAX_RATIO
    )


def classify_rps(hand_landmarks):
    """Classify MediaPipe hand landmarks as rock/paper/scissors/unknown."""
    if hand_landmarks is None:
        return UNKNOWN

    landmarks = hand_landmarks.landmark
    features = _finger_features(landmarks)

    index = features["index"]
    middle = features["middle"]
    ring = features["ring"]
    pinky = features["pinky"]

    open_count = sum(feature["open"] for feature in features.values())
    starting_count = sum(feature["starting"] for feature in features.values())

    if starting_count == 4 and open_count >= 3:
        return PAPER

    first_two_angle = (index["angle"] + middle["angle"]) / 2
    last_two_angle = (ring["angle"] + pinky["angle"]) / 2

    if (
        index["starting"]
        and middle["starting"]
        and not ring["starting"]
        and not pinky["starting"]
        and first_two_angle >= last_two_angle + SCISSORS_RELATIVE_GAP
    ):
        return SCISSORS

    if starting_count == 0 and _thumb_is_fist_like(landmarks):
        return ROCK

    return UNKNOWN
