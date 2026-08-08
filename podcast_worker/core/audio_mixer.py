"""Audio Mixer — combines speech and beat with ducking.

Used by: straub-ah-tts (TTS + Audio worktree)

Input:  speech WAV + beat WAV
Output: mixed podcast WAV with intro/outro
"""

import numpy as np
from . import config

SAMPLE_RATE = config.SAMPLE_RATE


def _as_stereo(samples: np.ndarray) -> np.ndarray:
    """Return float samples as an ``(frames, 2)`` array without dropping channels."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return np.column_stack((audio, audio))
    if audio.ndim != 2 or audio.shape[1] not in {1, 2}:
        raise ValueError("unsupported_pcm_channels")
    return np.repeat(audio, 2, axis=1) if audio.shape[1] == 1 else audio


def _as_mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _load_wav_to_numpy(filepath: str) -> np.ndarray:
    import wave
    with wave.open(filepath, "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
            raise ValueError("unsupported_pcm_format")
        channels = wf.getnchannels()
        if channels not in {1, 2}:
            raise ValueError("unsupported_pcm_channels")
        samples = np.frombuffer(
            wf.readframes(wf.getnframes()), dtype=np.int16
        ).astype(np.float32) / 32768.0
    return samples.reshape(-1, channels) if channels == 2 else samples


def _save_numpy_to_wav(samples: np.ndarray, filepath: str):
    import wave
    stereo = _as_stereo(samples)
    if not len(stereo):
        raise ValueError("empty_audio")
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.98:
        stereo = stereo / peak * 0.98
    samples_int16 = np.rint(np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples_int16.tobytes())


def detect_speech_regions(speech_audio: np.ndarray, silence_thresh: float = 0.02, min_silence_ms: int = 200) -> list:
    speech_audio = _as_mono(speech_audio)
    frame_size = int(SAMPLE_RATE * 0.02)
    if not len(speech_audio):
        return []
    energy = np.array([
        np.sqrt(np.mean(speech_audio[i:i + frame_size] ** 2))
        for i in range(0, len(speech_audio), frame_size)
    ])
    is_speech = energy > silence_thresh
    regions = []
    in_speech = False
    start = 0
    min_silence_frames = max(1, int(min_silence_ms / 20))
    for i, speaking in enumerate(is_speech):
        if speaking and not in_speech:
            start = i * frame_size
            in_speech = True
        elif not speaking and in_speech:
            remaining = is_speech[i:min(len(is_speech), i + min_silence_frames)]
            if len(remaining) == min_silence_frames and not remaining.any():
                regions.append((start, i * frame_size))
                in_speech = False
    if in_speech:
        regions.append((start, len(speech_audio)))
    return regions


def mix_with_ducking(speech_audio: np.ndarray, beat_audio: np.ndarray,
                     ducking_db: float = -18.0, base_db: float = -6.0,
                     attack_ms: float = 10.0, release_ms: float = 50.0) -> np.ndarray:
    speech = _as_stereo(speech_audio)
    beat = _as_stereo(beat_audio)
    speech_regions = detect_speech_regions(speech)
    ducking_gain = 10 ** (ducking_db / 20.0)
    base_gain = 10 ** (base_db / 20.0)

    min_len = max(len(speech), len(beat))
    if not len(beat):
        padded = np.zeros((min_len, 2), dtype=np.float32)
        padded[:len(speech)] = speech
        return padded
    if len(beat) < min_len:
        beat = np.tile(beat, (int(np.ceil(min_len / len(beat))), 1))[:min_len]
    else:
        beat = beat[:min_len].copy()

    speech_padded = np.zeros((min_len, 2), dtype=np.float32)
    speech_padded[:len(speech)] = speech
    gain_envelope = np.full(min_len, base_gain, dtype=np.float32)
    attack_samples = int(SAMPLE_RATE * attack_ms / 1000)
    release_samples = int(SAMPLE_RATE * release_ms / 1000)

    for start, end in speech_regions:
        fade_start = max(0, start - attack_samples)
        if start > fade_start:
            gain_envelope[fade_start:start] = np.linspace(
                base_gain, ducking_gain, start - fade_start, endpoint=False
            )
        gain_envelope[start:end] = ducking_gain
        fade_end = min(min_len, end + release_samples)
        if fade_end > end:
            gain_envelope[end:fade_end] = np.linspace(
                ducking_gain, base_gain, fade_end - end, endpoint=False
            )

    mixed = speech_padded + beat * gain_envelope[:, None]
    max_val = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if max_val > 0.95:
        mixed = mixed / max_val * 0.95
    return mixed


def _fit_beat_to_duration(beat_audio: np.ndarray, frames: int) -> np.ndarray:
    """Loop or trim the BPM bed to the decoded speech duration."""
    beat = _as_stereo(beat_audio)
    if frames <= 0:
        return np.empty((0, 2), dtype=np.float32)
    if not len(beat):
        return np.zeros((frames, 2), dtype=np.float32)
    return np.tile(beat, (int(np.ceil(frames / len(beat))), 1))[:frames]


def build_podcast_audio(speech_wav_path: str, beat_wav_path: str, output_path: str, bpm: int,
                        intro_seconds: float = 4.0, outro_seconds: float = 6.0,
                        duration_minutes: int | None = None):
    from .beat_generator import generate_beat

    speech = _load_wav_to_numpy(speech_wav_path)
    beat = _load_wav_to_numpy(beat_wav_path)
    speech_frames = len(speech)
    if not speech_frames:
        raise ValueError("empty_speech_audio")

    intro = _as_stereo(generate_beat(bpm, intro_seconds))
    intro[:min(int(SAMPLE_RATE * 2.0), len(intro))] *= np.linspace(
        0, 1, min(int(SAMPLE_RATE * 2.0), len(intro))
    )[:, None]

    # The requested project duration is not media duration.  The bed must follow
    # the decoded speech frames for this segment.
    mixed = mix_with_ducking(speech, _fit_beat_to_duration(beat, speech_frames))

    outro = _as_stereo(generate_beat(bpm, outro_seconds))
    outro[-min(int(SAMPLE_RATE * 3.0), len(outro)):] *= np.linspace(
        1, 0, min(int(SAMPLE_RATE * 3.0), len(outro))
    )[:, None]

    full = np.concatenate([intro, mixed, outro])
    fade_len = min(int(SAMPLE_RATE * 0.05), len(full))
    full[:fade_len] *= np.linspace(0, 1, fade_len)[:, None]
    full[-fade_len:] *= np.linspace(1, 0, fade_len)[:, None]
    _save_numpy_to_wav(full, output_path)
    return output_path


def convert_to_mp3(wav_path: str, mp3_path: str = None) -> str:
    import subprocess
    mp3_path = mp3_path or wav_path.replace(".wav", ".mp3")
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "192k", mp3_path],
                   capture_output=True, check=True)
    print(f"  Converted: {mp3_path}")
    return mp3_path
