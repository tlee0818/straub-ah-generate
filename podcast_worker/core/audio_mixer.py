"""Audio Mixer — combines speech and beat with ducking.

Used by: straub-ah-tts (TTS + Audio worktree)

Input:  speech WAV + beat WAV
Output: mixed podcast WAV with intro/outro
"""

import numpy as np
from . import config

SAMPLE_RATE = config.SAMPLE_RATE


def _load_wav_to_numpy(filepath: str) -> np.ndarray:
    import wave
    with wave.open(filepath, "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
            raise ValueError("unsupported_pcm_format")
        channels = wf.getnchannels()
        if channels not in {1, 2}:
            raise ValueError("unsupported_pcm_channels")
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32767.0
    return samples.reshape(-1, channels).mean(axis=1) if channels == 2 else samples


def _save_numpy_to_wav(samples: np.ndarray, filepath: str):
    import wave
    if not len(samples):
        raise ValueError("empty_audio")
    peak = np.max(np.abs(samples))
    if peak > 0.98:
        samples = samples / peak * 0.98
    samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    stereo = np.column_stack((samples_int16, samples_int16)).ravel()
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(stereo.tobytes())


def detect_speech_regions(speech_audio: np.ndarray, silence_thresh: float = 0.02, min_silence_ms: int = 200) -> list:
    frame_size = int(SAMPLE_RATE * 0.02)
    energy = np.array([
        np.sqrt(np.mean(speech_audio[i:i+frame_size]**2))
        for i in range(0, len(speech_audio), frame_size)
    ])
    is_speech = energy > silence_thresh
    regions = []
    in_speech = False
    start = 0
    min_silence_frames = int(min_silence_ms / 20)
    for i, speaking in enumerate(is_speech):
        if speaking and not in_speech:
            start = i * frame_size
            in_speech = True
        elif not speaking and in_speech:
            silence_len = 0
            for j in range(i, min(len(is_speech), i + min_silence_frames)):
                silence_len += 1 if not is_speech[j] else 0
                if silence_len >= min_silence_frames:
                    break
            if silence_len >= min_silence_frames or i == len(is_speech) - 1:
                regions.append((start, i * frame_size))
                in_speech = False
    if in_speech:
        regions.append((start, len(speech_audio)))
    return regions


def mix_with_ducking(speech_audio: np.ndarray, beat_audio: np.ndarray,
                     ducking_db: float = -18.0, base_db: float = -6.0,
                     attack_ms: float = 10.0, release_ms: float = 50.0) -> np.ndarray:
    speech_regions = detect_speech_regions(speech_audio)
    ducking_gain = 10 ** (ducking_db / 20.0)
    base_gain = 10 ** (base_db / 20.0)

    min_len = max(len(speech_audio), len(beat_audio))
    if len(beat_audio) == 0:
        # No beat to mix — return speech only (padded to min_len)
        speech_padded = np.zeros(min_len)
        speech_padded[:len(speech_audio)] = speech_audio
        return speech_padded
    if len(beat_audio) < min_len:
        repeats = int(np.ceil(min_len / len(beat_audio)))
        beat_audio = np.tile(beat_audio, repeats)[:min_len]

    beat = beat_audio[:min_len].copy()
    speech = np.zeros(min_len)
    speech[:len(speech_audio)] = speech_audio

    gain_envelope = np.ones(min_len) * base_gain
    attack_samples = int(SAMPLE_RATE * attack_ms / 1000)
    release_samples = int(SAMPLE_RATE * release_ms / 1000)

    for start, end in speech_regions:
        fade_start = max(0, start - attack_samples)
        for i in range(fade_start, start):
            t = (i - fade_start) / max(1, start - fade_start)
            gain_envelope[i] = base_gain + (ducking_gain - base_gain) * t
        gain_envelope[start:end] = ducking_gain
        fade_end = min(min_len, end + release_samples)
        for i in range(end, fade_end):
            t = (i - end) / max(1, fade_end - end)
            gain_envelope[i] = ducking_gain + (base_gain - ducking_gain) * t

    mixed = speech + beat * gain_envelope
    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed / max_val * 0.95
    return mixed


def build_podcast_audio(speech_wav_path: str, beat_wav_path: str, output_path: str, bpm: int,
                        intro_seconds: float = 4.0, outro_seconds: float = 6.0,
                        duration_minutes: int | None = None):
    from .beat_generator import generate_beat

    speech = _load_wav_to_numpy(speech_wav_path)
    beat = _load_wav_to_numpy(beat_wav_path)

    # Intro
    intro = generate_beat(bpm, intro_seconds)
    fade_len = int(SAMPLE_RATE * 2.0)
    intro[:min(fade_len, len(intro))] *= np.linspace(0, 1, min(fade_len, len(intro)))

    # Main
    mixed = mix_with_ducking(speech, beat)

    # Outro
    outro = generate_beat(bpm, outro_seconds)
    fade_len = int(SAMPLE_RATE * 3.0)
    outro[-min(fade_len, len(outro)):] *= np.linspace(1, 0, min(fade_len, len(outro)))

    full = np.concatenate([intro, mixed, outro])

    # Safety fades
    fade_len = int(SAMPLE_RATE * 0.05)
    full[:fade_len] *= np.linspace(0, 1, fade_len)
    full[-fade_len:] *= np.linspace(1, 0, fade_len)

    _save_numpy_to_wav(full, output_path)
    return output_path


def convert_to_mp3(wav_path: str, mp3_path: str = None) -> str:
    import subprocess
    mp3_path = mp3_path or wav_path.replace(".wav", ".mp3")
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "192k", mp3_path],
                   capture_output=True, check=True)
    print(f"  Converted: {mp3_path}")
    return mp3_path
