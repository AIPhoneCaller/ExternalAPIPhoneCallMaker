# recorder_vad.py
import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 16000
FRAME_DURATION = 0.02
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

CALIBRATION_SECONDS = 0.3

# -------- Human conversation tuning --------
MIN_START_SPEECH_SECONDS = 0.18   # must speak this long before we accept speech
MIN_SPEECH_DURATION = 0.50        # lock speech only after real phrase
END_SILENCE_SECONDS = 0.70        # silence to end utterance (IMPORTANT)

GLOBAL_NOISE_FLOOR = None


def _frame_energy(frame: np.ndarray) -> float:
    mono = frame.reshape(-1).astype(np.float32)
    return float(np.sqrt(np.mean(mono ** 2)) + 1e-9)


def _calibrate_noise_floor(stream, debug: bool) -> float:
    frames = int(CALIBRATION_SECONDS / FRAME_DURATION)
    energies = []

    if debug:
        print(f"[VAD] Calibrating ONCE for {CALIBRATION_SECONDS:.2f}s...")

    for _ in range(frames):
        frame, _ = stream.read(FRAME_SIZE)
        energies.append(_frame_energy(frame))

    noise = float(np.median(energies)) if energies else 1e-6

    if debug:
        print(f"[VAD] Calibration done → noise_floor={noise:.8f}")

    return noise


def record_until_silence(max_wait_seconds: float = 30.0, debug: bool = True):
    global GLOBAL_NOISE_FLOOR

    start_time = time.time()
    audio_frames = []

    # counters
    speech_frames = 0
    start_gate_frames = 0
    silence_frames = 0

    had_speech = False
    speech_locked = False

    start_gate_needed = int(MIN_START_SPEECH_SECONDS / FRAME_DURATION)
    lock_needed = int(MIN_SPEECH_DURATION / FRAME_DURATION)
    end_silence_needed = int(END_SILENCE_SECONDS / FRAME_DURATION)

    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="float32",
    )

    with stream:
        if GLOBAL_NOISE_FLOOR is None:
            GLOBAL_NOISE_FLOOR = _calibrate_noise_floor(stream, debug)

        noise = GLOBAL_NOISE_FLOOR
        start_th = max(noise * 1.8, 0.003)
        end_th = max(noise * 2.5, 0.005)

        if debug:
            print(
                f"[VAD] Using noise_floor={noise:.8f} "
                f"start_th={start_th:.6f} end_th={end_th:.6f}"
            )
            print("[VAD] Listening...")

        while True:
            if time.time() - start_time > max_wait_seconds:
                if debug:
                    print("[VAD] Timeout → stopping")
                break

            frame, _ = stream.read(FRAME_SIZE)
            energy = _frame_energy(frame)
            audio_frames.append(frame)

            # -------- BEFORE speech --------
            if not had_speech:
                if energy > start_th:
                    start_gate_frames += 1
                    if start_gate_frames >= start_gate_needed:
                        had_speech = True
                        speech_frames = start_gate_frames
                        if debug:
                            print("[VAD] Speech CONFIRMED (gate passed)")
                else:
                    start_gate_frames = 0
                continue

            # -------- AFTER speech --------
            if energy > end_th:
                speech_frames += 1
                silence_frames = 0

                if not speech_locked and speech_frames >= lock_needed:
                    speech_locked = True
                    if debug:
                        print("[VAD] Speech LOCKED")
            else:
                silence_frames += 1

            if speech_locked and silence_frames >= end_silence_needed:
                if debug:
                    print("[VAD] End-of-utterance → stopping")
                break

    if not had_speech:
        if debug:
            print("[VAD] No speech detected")
        return None, SAMPLE_RATE

    buffer = np.concatenate(audio_frames, axis=0).reshape(-1).astype("float32")

    if debug:
        print(f"[VAD] Captured audio: {len(buffer)/SAMPLE_RATE:.2f}s")

    return buffer, SAMPLE_RATE
