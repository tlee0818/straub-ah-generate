"""
Unit tests for beat_generator.py — procedural beat synthesis.

Tests validate that beats are generated at the correct BPM,
within amplitude limits, and with the expected structure.
"""

import os
import sys
import tempfile
import wave

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from podcast_worker.core.beat_generator import generate_beat, save_beat_to_wav, _bpm_to_beat_interval, _get_energy_profile

SAMPLE_RATE = 44100


class TestBPMConversions:
    def test_bpm_to_beat_interval_120(self):
        """120 BPM = 0.5s per beat."""
        assert abs(_bpm_to_beat_interval(120) - 0.5) < 0.001

    def test_bpm_to_beat_interval_60(self):
        """60 BPM = 1.0s per beat."""
        assert abs(_bpm_to_beat_interval(60) - 1.0) < 0.001

    def test_bpm_to_beat_interval_180(self):
        """180 BPM = 0.333s per beat."""
        assert abs(_bpm_to_beat_interval(180) - 1.0 / 3.0) < 0.001


class TestEnergyProfile:
    def test_high_energy_160_plus(self):
        profile = _get_energy_profile(180)
        assert profile["pattern_complexity"] == "high"
        assert profile["kick_gain"] == 1.0

    def test_medium_energy_120(self):
        profile = _get_energy_profile(120)
        assert profile["pattern_complexity"] == "medium"

    def test_low_energy_60(self):
        profile = _get_energy_profile(60)
        assert profile["pattern_complexity"] == "minimal"
        assert profile["kick_gain"] < 0.8

    def test_chill_energy_90(self):
        profile = _get_energy_profile(90)
        assert profile["pattern_complexity"] == "low"


class TestGenerateBeat:
    def test_generate_beat_returns_array(self):
        """Beat generation should return a numpy array."""
        beat = generate_beat(120, 1.0)
        assert isinstance(beat, np.ndarray)
        assert len(beat) > 0

    def test_generate_beat_correct_length(self):
        """1 second at 44100 Hz = 44100 samples."""
        beat = generate_beat(120, 1.0)
        assert len(beat) == SAMPLE_RATE

    def test_generate_beat_duration(self):
        """2.5 seconds at 44100 Hz = 110250 samples."""
        beat = generate_beat(100, 2.5)
        assert len(beat) == int(SAMPLE_RATE * 2.5)

    def test_generate_beat_within_amplitude(self):
        """Audio should not clip (max amplitude <= 0.95)."""
        beat = generate_beat(140, 2.0)
        max_amp = np.max(np.abs(beat))
        assert max_amp <= 0.95, f"Peak amplitude {max_amp} exceeds 0.95"

    def test_generate_beat_not_silent(self):
        """Beat should have non-zero energy."""
        beat = generate_beat(120, 1.0)
        rms = np.sqrt(np.mean(beat**2))
        assert rms > 0.001, f"RMS {rms} is too low — beat is silent"

    def test_generate_beat_at_various_bpms(self):
        """Should generate valid beats across the BPM range."""
        for bpm in [60, 90, 120, 160, 200, 220]:
            beat = generate_beat(bpm, 0.5)
            assert len(beat) > 0
            rms = np.sqrt(np.mean(beat**2))
            assert rms > 0.001, f"Beat at {bpm} BPM is silent"

    def test_generate_beat_kick_present(self):
        """Kick drum should create transient spikes."""
        beat = generate_beat(120, 0.5)
        # Look for sharp attack transients (kicks)
        diffs = np.abs(np.diff(beat))
        max_transient = np.max(diffs)
        assert max_transient > 0.01, "No kick transients detected"

    def test_generate_beat_sidechain(self):
        """Sidechain compression should create a rhythmic envelope."""
        beat = generate_beat(120, 1.0)
        # The envelope should vary over time (sidechain pumping)
        envelope = np.abs(beat)
        # Check that the envelope isn't flat
        std_env = np.std(envelope)
        assert std_env > 0.001, "Sidechain envelope is flat — no pumping effect"

    def test_generate_beat_empty_duration(self):
        """0 duration should produce empty array."""
        beat = generate_beat(120, 0)
        assert len(beat) == 0


class TestSaveBeatToWav:
    def test_save_creates_wav_file(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            result = save_beat_to_wav(120, 0.5, path)
            assert result == path
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0

            # Verify WAV header
            with wave.open(path, "r") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == SAMPLE_RATE
                frames = wf.readframes(wf.getnframes())
                assert len(frames) > 0
        finally:
            if os.path.isfile(path):
                os.remove(path)

    def test_save_wav_reasonable_size(self):
        """0.5 seconds mono 16-bit = 44100 bytes."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            save_beat_to_wav(120, 0.5, path)
            expected_min = 44  # WAV header
            expected_max = SAMPLE_RATE * 2  # 16-bit * 1 channel
            size = os.path.getsize(path)
            assert expected_min < size < expected_max + 1000
        finally:
            if os.path.isfile(path):
                os.remove(path)


class TestEdgeCases:
    def test_extreme_bpm_low(self):
        """BPM at minimum should still produce valid output."""
        beat = generate_beat(60, 1.0)
        assert len(beat) == SAMPLE_RATE
        assert np.max(np.abs(beat)) <= 0.95

    def test_extreme_bpm_high(self):
        """BPM at maximum should still produce valid output."""
        beat = generate_beat(220, 1.0)
        assert len(beat) == SAMPLE_RATE
        assert np.max(np.abs(beat)) <= 0.95

    def test_very_short_duration(self):
        """Very short durations should not crash."""
        for duration in [0.01, 0.05, 0.1]:
            beat = generate_beat(120, duration)
            expected = int(SAMPLE_RATE * duration)
            assert len(beat) == expected, f"Expected {expected} samples for {duration}s, got {len(beat)}"