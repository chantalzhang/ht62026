import cv2
import numpy as np


BG = (20, 20, 20)
PANEL = (45, 45, 45)
WHITE = (240, 240, 240)
GRAY = (150, 150, 150)


def _text(img, text, xy, scale=0.8, color=WHITE, thickness=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_game(frame, *, state, gesture, stable, locked, rocky_move, score, fps, camera_name, taunt):
    canvas = np.full((720, 1280, 3), BG, dtype=np.uint8)

    _text(canvas, "ROCKY vs HUMAN", (40, 55), 1.2)
    _text(canvas, f"Score: Rocky {score} - 0 Human", (850, 55), 0.8)
    _text(canvas, f"{camera_name} | {fps:.1f} FPS", (40, 90), 0.55, GRAY, 1)

    cv2.rectangle(canvas, (40, 120), (690, 610), PANEL, -1)
    cv2.rectangle(canvas, (730, 120), (1240, 610), PANEL, -1)
    _text(canvas, "Human", (60, 155), 0.8)
    _text(canvas, "Rocky", (760, 155), 0.8)

    camera_view = cv2.resize(frame, (610, 410))
    canvas[180:590, 60:670] = camera_view

    rocky_text = (rocky_move or "-").upper()
    _text(canvas, rocky_text, (820, 330), 1.8)
    if locked:
        _text(canvas, "Rocky wins", (820, 395), 1.0)
        _text(canvas, taunt, (820, 440), 0.65, GRAY, 1)
    else:
        _text(canvas, "-", (820, 395), 1.0, GRAY)

    _text(canvas, f"State: {state}", (40, 655), 0.75)
    _text(canvas, f"Gesture: {gesture.upper()}   Stable: {stable}", (270, 655), 0.75)
    _text(canvas, "space=start  r=reset  c=clear score  q=quit", (40, 695), 0.65, GRAY, 1)

    return canvas
