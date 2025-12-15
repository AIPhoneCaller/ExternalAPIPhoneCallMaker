# recorder_vad.py
import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 16000
FRAME_DURATION = 0.02
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

CALIBRATION_SECONDS = 0.3


def _frame_energy(frame: np.ndarray) -> float:
    mono = frame.reshape(-1).astype(np.float32)
    return float(np.sqrt(np.mean(mono ** 2)) + 1e-9)


def record_until_silence(
    max_wait_seconds: float = 30.0,
    end_silence_seconds: float = 0.25,
    debug: bool = True,
    debug_every_frames: int = 25,
    threshold_factor: float = 3.0,      # how much above noise floor counts as speech
    min_threshold: float = 0.0020,      # absolute minimum RMS threshold (prevents 0)
):
    audio_frames = []
    silence_counter = 0
    had_speech = False
    start_time = time.time()

    end_silence_frames = int(end_silence_seconds / FRAME_DURATION)

    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="float32",
    )

    with stream:
        calib_frames = int(CALIBRATION_SECONDS / FRAME_DURATION)
        energies = []

        if debug:
            print(f"[VAD] Calibrating for {CALIBRATION_SECONDS:.2f}s ({calib_frames} frames)...")

        for _ in range(calib_frames):
            frame, overflow = stream.read(FRAME_SIZE)
            if overflow and debug:
                print("[VAD] WARN: input overflow during calibration")
            energies.append(_frame_energy(frame))

        # Robust noise floor: use median (or a percentile) instead of mean
        noise_floor = float(np.median(energies)) if energies else 1e-6

        # Protect against zeros / near-zeros
        threshold = max(noise_floor * threshold_factor, min_threshold)

        if debug:
            print(f"[VAD] noise_floor(median)={noise_floor:.8f} threshold={threshold:.8f}")
            print("[VAD] Listening...")

        frame_idx = 0
        last_state = "silence"

        while True:
            if time.time() - start_time > max_wait_seconds:
                if debug:
                    print("[VAD] Timeout reached -> stopping")
                break

            frame, overflow = stream.read(FRAME_SIZE)
            if overflow and debug:
                print("[VAD] WARN: input overflow")

            energy = _frame_energy(frame)
            audio_frames.append(frame)

            is_speech = energy > threshold

            if is_speech:
                if not had_speech:
                    had_speech = True
                    if debug:
                        print(f"[VAD] Speech started (energy={energy:.8f} > {threshold:.8f})")
                silence_counter = 0
                if last_state != "speech":
                    last_state = "speech"
                    if debug:
                        print(f"[VAD] -> SPEECH (energy={energy:.8f})")
            else:
                if had_speech:
                    silence_counter += 1
                if last_state != "silence":
                    last_state = "silence"
                    if debug:
                        print(f"[VAD] -> SILENCE (energy={energy:.8f}), silence_frames={silence_counter}/{end_silence_frames}")

            if debug and (frame_idx % debug_every_frames == 0):
                state = "SPEECH" if is_speech else "silence"
                print(f"[VAD] frame={frame_idx} energy={energy:.8f} state={state} silence_frames={silence_counter}")

            if had_speech and silence_counter >= end_silence_frames:
                if debug:
                    print(f"[VAD] End-of-utterance: {end_silence_seconds:.2f}s silence -> stopping")
                break

            frame_idx += 1

    if not had_speech:
        if debug:
            dur = time.time() - start_time
            print(f"[VAD] No speech detected (listened {dur:.2f}s) -> returning None")
        return None, SAMPLE_RATE

    buffer_np = np.concatenate(audio_frames, axis=0).reshape(-1).astype("float32")

    if debug:
        dur = len(buffer_np) / SAMPLE_RATE
        print(f"[VAD] Captured audio: {dur:.2f}s, frames={len(audio_frames)}")

    return buffer_np, SAMPLE_RATE



def wait_for_speech(
    threshold: float,
    timeout: float = 2.0,
    min_duration_ms: int = 300,
) -> bool:
    """
    Bar-ge-in חכם:
    מחזיר True רק אם יש דיבור רציף אמיתי (לא רעש).
    """
    min_frames = int((min_duration_ms / 1000) / FRAME_DURATION)
    speech_frames = 0

    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="float32",
    )

    start = time.time()
    with stream:
        while time.time() - start < timeout:
            frame, _ = stream.read(FRAME_SIZE)
            if _frame_energy(frame) > threshold:
                speech_frames += 1
                if speech_frames >= min_frames:
                    return True
            else:
                speech_frames = 0

    return False
