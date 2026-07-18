from zero_vision.gesture_classifier import PAPER, ROCK, SCISSORS


COUNTERS = {
    ROCK: PAPER,
    PAPER: SCISSORS,
    SCISSORS: ROCK,
}


def counter_move(move):
    return COUNTERS.get(move)
