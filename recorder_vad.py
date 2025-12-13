# recorder_vad.py
import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 16000
FRAME_DURATION = 0.03  # 30 ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

SILENCE_MS = 500  # how long of silence to treat as "end of utterance"
SILENCE_FRAMES = int(SILENCE_MS / (FRAME_DURATION * 1000))

CALIBRATION_SECONDS = 0.5  # listen to background to set threshold


def _frame_energy(frame: np.ndarray) -> float:
    """RMS energy of a mono frame."""
    # frame shape is (N, 1) because channels=1
    mono = frame.astype(np.float32).reshape(-1)
    return float(np.sqrt(np.mean(mono ** 2)) + 1e-9)


def record_until_silence(
    max_wait_seconds: float = 60.0,
    return_buffer: bool = True,
    return_threshold: bool = True,
):
    """
    ChatGPT-style end-of-speech detector using simple energy-based VAD.

    - Calibrates to ambient noise for CALIBRATION_SECONDS
    - Detects speech when energy > (noise_floor * factor)
    - Stops when we had speech and then ~0.5s of silence

    Returns:
        (audio_buffer: np.ndarray | None, samplerate: int, energy_threshold: float)
        If no speech detected at all → (None, SAMPLE_RATE, energy_threshold)
    """
    print("🎤 Listening... (energy-based VAD)")

    audio_frames = []
    silence_counter = 0
    start_time = time.time()
    had_speech = False

    # Open audio stream
    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="float32",
    )

    with stream:
        # --- 1️⃣ Calibration phase: measure ambient noise ---
        print(f"🎚 Calibrating noise floor for {CALIBRATION_SECONDS:.1f}s...")
        calib_frames = int(CALIBRATION_SECONDS / FRAME_DURATION)
        calib_energies = []

        for _ in range(calib_frames):
            frame, overflow = stream.read(FRAME_SIZE)
            if overflow:
                # not fatal, but nice to log
                print("[WARN] Input overflow during calibration")
            e = _frame_energy(frame)
            calib_energies.append(e)

        noise_floor = float(np.mean(calib_energies)) if calib_energies else 1e-6
        # how aggressive to be above noise
        energy_threshold = noise_floor * 2.5
        print(f"📏 Noise floor: {noise_floor:.8f}, threshold: {energy_threshold:.8f}")

        # --- 2️⃣ Main recording loop ---
        while True:
            if time.time() - start_time > max_wait_seconds:
                print("⏳ Timeout reached → stopping.")
                break

            frame, overflow = stream.read(FRAME_SIZE)
            if overflow:
                print("[WARN] Input overflow")

            energy = _frame_energy(frame)
            audio_frames.append(frame.copy())

            if energy > energy_threshold:
                # We are in speech
                had_speech = True
                silence_counter = 0
            else:
                # We are in (relative) silence
                if had_speech:  # only count silence *after* speech started
                    silence_counter += 1

            # End of utterance condition
            if had_speech and silence_counter >= SILENCE_FRAMES:
                print("⏹ Detected ~0.5s of silence after speech → stopping.")
                break

    if not audio_frames or not had_speech:
        print("⚠️ No clear speech detected, returning None buffer.")
        return (None, SAMPLE_RATE, energy_threshold)

    # Concatenate frames into one buffer
    buffer_np = np.concatenate(audio_frames, axis=0).reshape(-1).astype("float32")

    if return_buffer:
        if return_threshold:
            return buffer_np, SAMPLE_RATE, energy_threshold
        else:
            return buffer_np, SAMPLE_RATE, None

    # If for some reason not returning buffer, stay compatible
    return (None, SAMPLE_RATE, energy_threshold)
