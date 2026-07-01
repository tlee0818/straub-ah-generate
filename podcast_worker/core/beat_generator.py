"""Beat Generator — procedurally generates electronic beats at a target BPM.

Used by: straub-ah-tts (TTS + Audio worktree)
"""

import numpy as np
from . import config

SAMPLE_RATE = config.SAMPLE_RATE


def _bpm_to_beat_interval(bpm: float) -> float:
    return 60.0 / bpm


def _get_energy_profile(bpm: int) -> dict:
    if bpm >= 160:
        return {"kick_gain": 1.0, "snare_gain": 0.7, "hihat_gain": 0.8, "sub_gain": 0.5,
                "noise_gain": 0.15, "pattern_complexity": "high", "hihat_pattern": "16th", "sidechain_speed": "fast"}
    elif bpm >= 120:
        return {"kick_gain": 0.9, "snare_gain": 0.6, "hihat_gain": 0.6, "sub_gain": 0.4,
                "noise_gain": 0.1, "pattern_complexity": "medium", "hihat_pattern": "8th", "sidechain_speed": "medium"}
    elif bpm >= 90:
        return {"kick_gain": 0.8, "snare_gain": 0.5, "hihat_gain": 0.4, "sub_gain": 0.3,
                "noise_gain": 0.05, "pattern_complexity": "low", "hihat_pattern": "4th", "sidechain_speed": "slow"}
    else:
        return {"kick_gain": 0.7, "snare_gain": 0.4, "hihat_gain": 0.3, "sub_gain": 0.3,
                "noise_gain": 0.03, "pattern_complexity": "minimal", "hihat_pattern": "downbeat", "sidechain_speed": "slow"}


def _generate_kick(duration_samples: int) -> np.ndarray:
    t = np.arange(duration_samples) / SAMPLE_RATE
    click = np.exp(-t * 2000) * np.random.normal(0, 1, len(t)) * 0.3
    body = np.sin(2 * np.pi * 60 * np.exp(-t * 30)) * np.exp(-t * 35)
    return click + body * 0.7


def _generate_snare(duration_samples: int) -> np.ndarray:
    t = np.arange(duration_samples) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 40)
    noise = np.random.normal(0, 1, len(t)) * np.exp(-t * 25)
    return tone * 0.4 + noise * 0.6


def _generate_hihat(duration_samples: int, closed: bool = True) -> np.ndarray:
    t = np.arange(duration_samples) / SAMPLE_RATE
    noise = np.random.normal(0, 1, len(t))
    env = np.exp(-t * 150) if closed else np.exp(-t * 20)
    return noise * env * 0.3


def _generate_sub(bpm: int, total_samples: int) -> np.ndarray:
    interval = _bpm_to_beat_interval(bpm)
    t = np.arange(total_samples) / SAMPLE_RATE
    beat_phase = (t % interval) / interval
    pulse = np.sin(beat_phase * np.pi) ** 2
    sub = np.sin(2 * np.pi * 55.0 * t) * pulse * 0.3
    return sub


def _generate_noise_swell(bpm: int, total_samples: int) -> np.ndarray:
    t = np.arange(total_samples) / SAMPLE_RATE
    interval = _bpm_to_beat_interval(bpm)
    beat_phase = (t % interval) / interval
    noise = np.random.normal(0, 1, len(t))
    window = 100
    kernel = np.ones(window) / window
    smooth_noise = np.convolve(noise, kernel, mode="same")
    amplitude = (1 - beat_phase) * 0.05
    return smooth_noise * amplitude


def _apply_sidechain(audio: np.ndarray, bpm: int) -> np.ndarray:
    interval = _bpm_to_beat_interval(bpm)
    t = np.arange(len(audio)) / SAMPLE_RATE
    beat_phase = (t % interval) / interval
    sidechain = 1.0 - 0.4 * np.exp(-beat_phase * 8)
    return audio * sidechain


def generate_beat(bpm: int, duration_seconds: float) -> np.ndarray:
    total_samples = int(SAMPLE_RATE * duration_seconds)
    if total_samples == 0:
        return np.array([])
    interval_samples = int(SAMPLE_RATE * _bpm_to_beat_interval(bpm))
    energy = _get_energy_profile(bpm)
    mix = np.zeros(total_samples)

    # Kick
    kick_samples = int(0.15 * SAMPLE_RATE)
    for i in range(0, total_samples, interval_samples):
        end = min(i + kick_samples, total_samples)
        seg_len = end - i
        mix[i:end] += _generate_kick(seg_len) * energy["kick_gain"]

    # Snare (2 & 4)
    if bpm >= 90:
        snare_samples = int(0.12 * SAMPLE_RATE)
        half_interval = interval_samples // 2
        for i in range(half_interval, total_samples, interval_samples):
            end = min(i + snare_samples, total_samples)
            seg_len = end - i
            mix[i:end] += _generate_snare(seg_len) * energy["snare_gain"]

    # Hi-hats
    step = {"16th": interval_samples // 4, "8th": interval_samples // 2,
            "4th": interval_samples}.get(energy["hihat_pattern"], interval_samples * 2)
    hihat_samples = int(0.05 * SAMPLE_RATE)
    for i in range(0, total_samples, step):
        if i > 0 and energy["hihat_pattern"] in ("16th", "8th") and (i // step) % 4 == 3 and energy["hihat_pattern"] == "16th":
            hihat_samples_open = int(0.15 * SAMPLE_RATE)
            end = min(i + hihat_samples_open, total_samples)
            seg_len = end - i
            mix[i:end] += _generate_hihat(seg_len, closed=False) * energy["hihat_gain"] * 0.5
            continue
        end = min(i + hihat_samples, total_samples)
        seg_len = end - i
        mix[i:end] += _generate_hihat(seg_len) * energy["hihat_gain"]

    # Sub bass
    if energy["sub_gain"] > 0:
        mix += _generate_sub(bpm, total_samples) * energy["sub_gain"]

    # Atmosphere
    if energy["noise_gain"] > 0:
        mix += _generate_noise_swell(bpm, total_samples) * energy["noise_gain"]

    # Sidechain
    mix = _apply_sidechain(mix, bpm)

    max_peak = np.max(np.abs(mix))
    if max_peak > 0.95:
        mix = mix / max_peak * 0.95
    return mix


def save_beat_to_wav(bpm: int, duration_seconds: float, filepath: str) -> str:
    import wave
    samples = generate_beat(bpm, duration_seconds)
    samples_int16 = (samples * 32767).astype(np.int16)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples_int16.tobytes())
    print(f"  Beat saved: {filepath}")
    return filepath
