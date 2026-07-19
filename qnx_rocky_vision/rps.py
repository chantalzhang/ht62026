from qnx_rocky_vision.gesture_classifier import PAPER, ROCK, SCISSORS


COUNTERS = {
    ROCK: PAPER,
    PAPER: SCISSORS,
    SCISSORS: ROCK,
}

RESULTS = {
    (ROCK, PAPER): "Paper covers rock",
    (PAPER, SCISSORS): "Scissors cut paper",
    (SCISSORS, ROCK): "Rock crushes scissors",
}


def counter_move(move):
    return COUNTERS.get(move)


def result_text(human_move, rocky_move):
    return RESULTS.get((human_move, rocky_move))
