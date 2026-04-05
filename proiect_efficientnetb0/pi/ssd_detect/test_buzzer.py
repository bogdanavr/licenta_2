from gpiozero import TonalBuzzer
from gpiozero.tones import Tone
from time import sleep

BUZZER_PIN = 17
buzzer = TonalBuzzer(BUZZER_PIN)


def play_tone(note, duration, pause=0.05):
    buzzer.play(Tone(note))
    sleep(duration)
    buzzer.stop()
    sleep(pause)


def play_sequence(sequence, title=""):
    """
    sequence = list of tuples: (note, duration, pause)
    Exemple note: "A3", "C4", "E4", "A4", "C5", "E5", "A5"
    """
    if title:
        print(f"\n>>> {title}")

    for note, duration, pause in sequence:
        play_tone(note, duration, pause)


def buzz_for_emotion(emotion):
    if not emotion:
        return

    emotion = emotion.upper()

    if emotion == "HAPPY":
        play_sequence([
            ("A4", 0.08, 0.03),
            ("C5", 0.08, 0.03),
            ("E5", 0.10, 0.04),
            ("A5", 0.12, 0.06),
        ], "HAPPY :)")

    elif emotion == "NEUTRAL":
        play_sequence([
            ("A4", 0.18, 0.08),
            ("A4", 0.18, 0.08),
        ], "NEUTRAL :|")

    elif emotion == "SAD":
        play_sequence([
            ("E4", 0.20, 0.05),
            ("C4", 0.25, 0.06),
            ("A3", 0.40, 0.10),
        ], "SAD :(")

    elif emotion == "SURPRISE":
        play_sequence([
            ("A4", 0.04, 0.02),
            ("C5", 0.04, 0.02),
            ("E5", 0.05, 0.03),
            ("A5", 0.10, 0.06),
            ("E5", 0.05, 0.03),
        ], "SURPRISE :O")

    else:
        play_sequence([
            ("A4", 0.08, 0.05),
        ], "UNKNOWN")


if __name__ == "__main__":
    emotions = ["HAPPY", "NEUTRAL", "SAD", "SURPRISE"]

    print("=== Buzzer emotion demo ===")
    for e in emotions:
        print(f"Testing: {e}")
        buzz_for_emotion(e)
        sleep(1)
