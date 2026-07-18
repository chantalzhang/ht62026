from rocky_vision.gesture_classifier import UNKNOWN


class GestureStabilizer:
    def __init__(self, required_frames=3):
        self.required_frames = required_frames
        self.reset()

    def reset(self):
        self.candidate = UNKNOWN
        self.count = 0
        self.locked = None

    def update(self, gesture):
        if self.locked:
            return self.locked

        if gesture == UNKNOWN:
            self.candidate = UNKNOWN
            self.count = 0
            return None

        if gesture == self.candidate:
            self.count += 1
        else:
            self.candidate = gesture
            self.count = 1

        if self.count >= self.required_frames:
            self.locked = gesture
            return self.locked

        return None
