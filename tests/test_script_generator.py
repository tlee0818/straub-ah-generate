"""
Tests for script_generator.py — follow-up questions and summary generation.

These tests validate prompt construction and function dispatch without
calling real LLM APIs.
"""

import json
import os
import sys

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from podcast_worker.core.script_generator import (
    _get_follow_up_prompt,
    _get_summary_prompt,
    generate_follow_up_questions,
    generate_script_summary,
    flatten_script,
)

SAMPLE_SCRIPT = {
    "title": "Test Episode",
    "segments": [
        {"segment_type": "intro", "text": "Welcome to the show.", "approx_duration_seconds": 10},
        {"segment_type": "content", "text": "Today we talk about AI.", "approx_duration_seconds": 20},
        {"segment_type": "content", "text": "It is changing the world.", "approx_duration_seconds": 20},
        {"segment_type": "outro", "text": "Thanks for listening.", "approx_duration_seconds": 10},
    ],
}


class TestFlattenScript:
    def test_flatten_basic(self):
        text = flatten_script(SAMPLE_SCRIPT)
        assert "Welcome to the show." in text
        assert "Thanks for listening." in text
        assert "Today we talk about AI." in text

    def test_flatten_empty_segments(self):
        assert flatten_script({"title": "x", "segments": []}) == ""

    def test_flatten_no_segments_key(self):
        assert flatten_script({"title": "x"}) == ""


class TestFollowUpPrompt:
    def test_prompt_contains_script_content(self):
        prompt = _get_follow_up_prompt("AI", SAMPLE_SCRIPT)
        assert "Test Episode" in prompt
        assert "AI" in prompt
        assert "follow_up_questions" in prompt
        assert "beginner" in prompt
        assert "advanced" in prompt

    def test_prompt_requests_json(self):
        prompt = _get_follow_up_prompt("Tech", SAMPLE_SCRIPT)
        assert "JSON" in prompt or "json" in prompt
        assert "question" in prompt

    def test_prompt_with_default_title(self):
        script_no_title = {"segments": [{"segment_type": "content", "text": "Hello"}]}
        prompt = _get_follow_up_prompt("Test", script_no_title)
        assert "Untitled" in prompt


class TestSummaryPrompt:
    def test_prompt_contains_script_content(self):
        prompt = _get_summary_prompt(SAMPLE_SCRIPT)
        assert "Test Episode" in prompt
        assert "summary" in prompt
        assert "key_points" in prompt
        assert "key_takeaway" in prompt

    def test_prompt_requests_structured_output(self):
        prompt = _get_summary_prompt(SAMPLE_SCRIPT)
        assert "JSON" in prompt or "json" in prompt

    def test_prompt_with_default_title(self):
        script_no_title = {"segments": [{"segment_type": "content", "text": "Hello"}]}
        prompt = _get_summary_prompt(script_no_title)
        assert "Untitled" in prompt


class TestGenerateFollowUpDispatch:
    def test_raises_on_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            generate_follow_up_questions("AI", SAMPLE_SCRIPT, provider="nonexistent")

    def test_raises_openai_without_key(self):
        """OpenAI dispatch should fail because there's no real key configured."""
        with pytest.raises((ValueError, ImportError)):
            generate_follow_up_questions("AI", SAMPLE_SCRIPT, provider="openai", api_key="")


class TestGenerateSummaryDispatch:
    def test_raises_on_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            generate_script_summary(SAMPLE_SCRIPT, provider="nonexistent")

    def test_raises_openai_without_key(self):
        with pytest.raises((ValueError, ImportError)):
            generate_script_summary(SAMPLE_SCRIPT, provider="openai", api_key="")