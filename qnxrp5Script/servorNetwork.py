#!/usr/bin/env python3
"""
rps_servo_server.py

HTTP server controlling 2 servos (Motor A, Motor B) for a rock/paper/
scissors style gesture setup, on a QNX Raspberry Pi 5 via rpi_gpio.

Endpoints (6 total):
    GET /rock            Motor A moves up 70 degrees, holds there
    GET /rock/reset       Motor A returns to starting position

    GET /paper            Motor A and Motor B both move down 80 degrees, hold there
    GET /paper/reset       Motor A and Motor B both return to starting position

    GET /scissors         Motor B moves up 70 degrees, holds there
    GET /scissors/reset    Motor B returns to starting position

    GET /status           (bonus) shows current angle of both servos

Each gesture endpoint moves and holds -- it does NOT auto-return. Only
calling that gesture's /reset endpoint brings it back to the starting
position.

Wiring: same as servo_control.py
    Motor A signal -> GPIO 12
    Motor B signal -> GPIO 13
    Power (both servos) -> external 5V supply
    Ground -> common ground (Pi + external supply)

Run:
    python3 rps_servo_server.py

Then, from another machine/app:
    curl http://<pi-ip>:8080/rock
    curl http://<pi-ip>:8080/rock/reset
    curl http://<pi-ip>:8080/paper
    curl http://<pi-ip>:8080/paper/reset
    curl http://<pi-ip>:8080/scissors
    curl http://<pi-ip>:8080/scissors/reset
    curl http://<pi-ip>:8080/status
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import rpi_gpio as GPIO

MOTOR_A_PIN = 12
MOTOR_B_PIN = 13

PWM_FREQ_HZ = 50

MIN_PULSE_MS = 1.0
MAX_PULSE_MS = 2.0
PERIOD_MS = 1000 / PWM_FREQ_HZ

START_ANGLE = 90     # resting position for both servos
ROCK_DELTA = 70       # Motor A moves up this many degrees for "rock"
PAPER_DELTA = -80     # Both motors move down this many degrees for "paper"
SCISSORS_DELTA = -70   # Motor B moves down this many degrees for "scissors"

# Set these based on which direction each servo actually needs to spin.
# Flip a flag to True if that motor moves the opposite way from what
# you expect for a given angle.
MOTOR_A_REVERSED = True
MOTOR_B_REVERSED = True

PORT = 8080

state = {"motor_a": START_ANGLE, "motor_b": START_ANGLE}


def angle_to_duty_cycle(angle):
    angle = max(0, min(180, angle))
    pulse_ms = MIN_PULSE_MS + (MAX_PULSE_MS - MIN_PULSE_MS) * (angle / 180)
    return (pulse_ms / PERIOD_MS) * 100


def servo_setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MOTOR_A_PIN, GPIO.OUT)
    GPIO.setup(MOTOR_B_PIN, GPIO.OUT)

    motor_a = GPIO.PWM(MOTOR_A_PIN, PWM_FREQ_HZ)
    motor_b = GPIO.PWM(MOTOR_B_PIN, PWM_FREQ_HZ)
    start_a_physical = 180 - START_ANGLE if MOTOR_A_REVERSED else START_ANGLE
    start_b_physical = 180 - START_ANGLE if MOTOR_B_REVERSED else START_ANGLE
    motor_a.start(angle_to_duty_cycle(start_a_physical))
    motor_b.start(angle_to_duty_cycle(start_b_physical))

    return motor_a, motor_b


def set_motor_a(angle):
    angle = max(0, min(180, angle))
    state["motor_a"] = angle  # store the logical (visual) angle

    physical_angle = 180 - angle if MOTOR_A_REVERSED else angle
    motor_a.ChangeDutyCycle(angle_to_duty_cycle(physical_angle))


def set_motor_b(angle):
    angle = max(0, min(180, angle))
    state["motor_b"] = angle  # store the logical (visual) angle

    physical_angle = 180 - angle if MOTOR_B_REVERSED else angle
    motor_b.ChangeDutyCycle(angle_to_duty_cycle(physical_angle))


class RPSHandler(BaseHTTPRequestHandler):
    def _respond(self, code, message):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write((message + "\n").encode())

    def do_GET(self):
        path = self.path.rstrip("/")

        if path == "/status":
            self._respond(200, str(state))
            return

        if path == "/rock":
            set_motor_a(START_ANGLE + ROCK_DELTA)
            self._respond(200, f"rock: motor A -> {state['motor_a']} degrees")
            return

        if path == "/rock/reset":
            set_motor_a(START_ANGLE)
            self._respond(200, f"rock reset: motor A -> {state['motor_a']} degrees")
            return

        if path == "/paper":
            set_motor_a(START_ANGLE + PAPER_DELTA)
            set_motor_b(START_ANGLE - PAPER_DELTA)  # inverted relative to Motor A
            self._respond(
                200,
                f"paper: motor A -> {state['motor_a']} degrees, "
                f"motor B -> {state['motor_b']} degrees",
            )
            return

        if path == "/paper/reset":
            set_motor_a(START_ANGLE)
            set_motor_b(START_ANGLE)
            self._respond(
                200,
                f"paper reset: motor A -> {state['motor_a']} degrees, "
                f"motor B -> {state['motor_b']} degrees",
            )
            return

        if path == "/scissors":
            set_motor_b(START_ANGLE + SCISSORS_DELTA)
            self._respond(200, f"scissors: motor B -> {state['motor_b']} degrees")
            return

        if path == "/scissors/reset":
            set_motor_b(START_ANGLE)
            self._respond(200, f"scissors reset: motor B -> {state['motor_b']} degrees")
            return

        self._respond(404, "unknown endpoint")

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def main():
    global motor_a, motor_b
    print("Setting up servo GPIO pins...")
    motor_a, motor_b = servo_setup()

    server = HTTPServer(("0.0.0.0", PORT), RPSHandler)
    print(f"Listening on port {PORT}. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping, returning servos to starting position.")
    finally:
        set_motor_a(START_ANGLE)
        set_motor_b(START_ANGLE)
        motor_a.stop()
        motor_b.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()