from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend" / "assets" / "sounds"


def load_env_file():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


load_env_file()

API_KEY = os.environ.get("ELEVENLABS_API_KEY")
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")
MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
OUTPUT_FORMAT = "mp3_44100_128"
STABILITY = float(os.environ.get("ELEVENLABS_STABILITY", "0.18"))
SIMILARITY = float(os.environ.get("ELEVENLABS_SIMILARITY", "0.8"))
STYLE = float(os.environ.get("ELEVENLABS_STYLE", "0.95"))

LINES = {
    "yell_rock.mp3": os.environ.get("ELEVENLABS_ROCK_TEXT", "Rock!"),
    "yell_paper.mp3": os.environ.get("ELEVENLABS_PAPER_TEXT", "Paper!"),
    "yell_scissors.mp3": os.environ.get("ELEVENLABS_SCISSORS_TEXT", "Scissors!"),
}


def available_voice_ids():
    if VOICE_ID:
        return [("selected", VOICE_ID)]
    request = urllib.request.Request(
        "https://api.elevenlabs.io/v1/voices",
        headers={"xi-api-key": API_KEY},
    )
    try:
        with urllib.request.urlopen(request) as response:
            voices = json.loads(response.read()).get("voices", [])
    except HTTPError as exc:
        details = exc.read().decode(errors="replace")
        raise SystemExit(f"Could not list ElevenLabs voices: {exc.code} {details}") from exc
    return [(voice.get("name", "voice"), voice["voice_id"]) for voice in voices]


def generate_clip(voice_id: str, text: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format={OUTPUT_FORMAT}"
    body = json.dumps({
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": STABILITY,
            "similarity_boost": SIMILARITY,
            "style": STYLE,
            "use_speaker_boost": True,
        },
    }).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "xi-api-key": API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return response.read()


def main():
    if not API_KEY:
        raise SystemExit("Paste ELEVENLABS_API_KEY into .env first")

    OUT.mkdir(parents=True, exist_ok=True)
    last_error = None
    for voice_name, voice_id in available_voice_ids():
        try:
            print(f"Trying voice: {voice_name}")
            for filename, text in LINES.items():
                path = OUT / filename
                path.write_bytes(generate_clip(voice_id, text))
                print(path)
            print(f"Used ElevenLabs voice: {voice_name} ({voice_id})")
            return
        except HTTPError as exc:
            last_error = exc.read().decode(errors="replace")
            print(f"Skipped voice {voice_name}: ElevenLabs error {exc.code}")
            if VOICE_ID:
                raise SystemExit(f"Selected voice failed: {last_error}") from exc

    raise SystemExit(f"No usable ElevenLabs voice found. Last error: {last_error}")


if __name__ == "__main__":
    main()
