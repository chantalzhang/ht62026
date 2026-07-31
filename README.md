# Rocky 🪨✂️📄

**First things first: Rocky is a physical robot.**

Yes, most of the code in this repo is for the supporting UI — the game screens, the countdown, the sounds. But the star of the show is a real robotic hand, built on a Raspberry Pi 5 running QNX, that plays rock-paper-scissors against you with actual servo-driven fingers.

## How it works

1. A camera watches your hand and figures out if you threw rock, paper, or scissors.
2. Rocky picks its move and the servos physically shape the robot hand into it.
3. The web UI shows the countdown, the result, and lets Rocky talk a little trash.

## What's in here

| Folder | What it is |
|---|---|
| `qnxrp5Script/` | The servo server that runs on the QNX Raspberry Pi 5 — this is what moves the actual hand |
| `rocky_vision/` | Hand tracking + gesture recognition (rock/paper/scissors detection) |
| `qnx_rocky_vision/` | The QNX-side version of the vision code |
| `frontend/` | The game UI — countdown, game screen, and the white flag for when you give up |
| `rocky_hardware/` | Hardware helpers |
| `scripts/` | Generates Rocky's voicelines and moves |

## Running it

The servo server runs on the Pi:

```
python3 qnxrp5Script/servorNetwork.py
```

Then the vision/game side talks to it over HTTP (`/rock`, `/paper`, `/scissors`) to make the hand move.

---

Built for Hack the 6ix 2026.
