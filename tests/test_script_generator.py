"""
Tests for script_generator.py — follow-up questions and summary generation.

These tests validate prompt construction and function dispatch without
calling real LLM APIs.
"""


import pytest

from podcast_worker.core import script_generator
from podcast_worker.core.script_generator import (
    _get_follow_up_prompt,
    _get_outline_prompt,
    _get_section_prompt,
    _get_summary_prompt,
    generate_follow_up_questions,
    generate_script,
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


class TestScriptGenerationFlow:
    def test_outline_prompt_requests_sections_not_script_text(self):
        prompt = _get_outline_prompt("AI safety", bpm=128, duration_minutes=4)

        assert "sections" in prompt
        assert "topic" in prompt
        assert '"segments"' not in prompt
        assert '"text"' not in prompt

    def test_section_prompt_includes_outline_neighbors_and_previous_text(self):
        outline = {
            "title": "AI Safety",
            "sections": [
                {"segment_type": "intro", "topic": "Hook", "approx_duration_seconds": 30},
                {"segment_type": "content", "topic": "Risk framing", "approx_duration_seconds": 45},
                {"segment_type": "outro", "topic": "Takeaway", "approx_duration_seconds": 30},
            ],
        }

        prompt = _get_section_prompt(
            "AI safety",
            128,
            outline,
            outline["sections"][1],
            outline["sections"][0],
            outline["sections"][2],
            "Welcome text from the prior section.",
            4,
        )

        assert "FULL OUTLINE" in prompt
        assert "1. intro (30s): Hook" in prompt
        assert "CURRENT SECTION TOPIC: Risk framing" in prompt
        assert "PREVIOUS SECTION TOPIC: Hook" in prompt
        assert "NEXT SECTION TOPIC: Takeaway" in prompt
        assert "Welcome text from the prior section." in prompt

    def test_generate_script_calls_outline_then_each_section_with_context(self, monkeypatch):
        calls = []

        outline = {
            "title": "AI Safety",
            "sections": [
                {"segment_type": "intro", "topic": "Hook", "approx_duration_seconds": 30},
                {"segment_type": "content", "topic": "Risk framing", "approx_duration_seconds": 45},
                {"segment_type": "outro", "topic": "Takeaway", "approx_duration_seconds": 30},
            ],
        }
        section_responses = [
            {"segment_type": "intro", "text": "Intro text.", "approx_duration_seconds": 30},
            {"segment_type": "content", "text": "Risk text.", "approx_duration_seconds": 45},
            {"segment_type": "outro", "text": "Outro text.", "approx_duration_seconds": 30},
        ]

        def fake_call_provider(system_prompt, user_prompt, provider=None, **kwargs):
            calls.append((system_prompt, user_prompt, provider, kwargs))
            if len(calls) == 1:
                return outline
            return section_responses[len(calls) - 2]

        monkeypatch.setattr(script_generator, "_call_provider", fake_call_provider)

        script = generate_script(
            "AI safety",
            128,
            duration_minutes=4,
            provider="test-provider",
            model="test-model",
        )

        assert script == {
            "title": "AI Safety",
            "segments": [
                {
                    "segment_type": "intro",
                    "subtopic": "Hook",
                    "title": "Hook",
                    "text": "Intro text.",
                    "approx_duration_seconds": 30,
                },
                {
                    "segment_type": "content",
                    "subtopic": "Risk framing",
                    "title": "Risk framing",
                    "text": "Risk text.",
                    "approx_duration_seconds": 45,
                },
                {
                    "segment_type": "outro",
                    "subtopic": "Takeaway",
                    "title": "Takeaway",
                    "text": "Outro text.",
                    "approx_duration_seconds": 30,
                },
            ],
        }
        assert len(calls) == 4
        assert "outline planner" in calls[0][0]
        assert "sections" in calls[0][1]
        assert all(call[2] == "test-provider" for call in calls)
        assert all(call[3] == {"model": "test-model"} for call in calls)

        second_section_prompt = calls[2][1]
        assert "FULL OUTLINE" in second_section_prompt
        assert "PREVIOUS SECTION TOPIC: Hook" in second_section_prompt
        assert "CURRENT SECTION TOPIC: Risk framing" in second_section_prompt
        assert "NEXT SECTION TOPIC: Takeaway" in second_section_prompt
        assert "IMMEDIATELY PREVIOUS SECTION TEXT:\nIntro text." in second_section_prompt

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


class TestPipelineSegmentCreation:
    """Verify outline-first script output matches what the pipeline expects."""

    def test_generate_script_output_has_segments_with_required_fields(self, monkeypatch):
        """Pipeline consumes script['segments'] — each must have text, segment_type, approx_duration_seconds."""
        outline = {
            "title": "Test",
            "sections": [
                {"segment_type": "intro", "topic": "Start", "approx_duration_seconds": 15},
                {"segment_type": "content", "topic": "Middle", "approx_duration_seconds": 30},
            ],
        }
        responses = [
            {"segment_type": "intro", "text": "Hello.", "approx_duration_seconds": 15},
            {"segment_type": "content", "text": "World.", "approx_duration_seconds": 30},
        ]
        calls = []

        def fake_call(system_prompt, user_prompt, provider=None, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return outline
            return responses[len(calls) - 2]

        monkeypatch.setattr(script_generator, "_call_provider", fake_call)

        script = generate_script("test", 120)
        segments = script["segments"]
        assert len(segments) == 2
        for seg in segments:
            assert "text" in seg, "Pipeline needs text per segment"
            assert "segment_type" in seg, "Pipeline needs segment_type per segment"
            assert "approx_duration_seconds" in seg, "Pipeline needs approx_duration_seconds per segment"
            assert seg["text"], "Generated text must be non-empty"

    def test_segment_subtopic_and_title_present(self, monkeypatch):
        """Pipeline uses subtopic for DB segment metadata and title for episode context."""
        outline = {
            "title": "Full Episode Title",
            "sections": [
                {"segment_type": "intro", "topic": "The Hook", "approx_duration_seconds": 10},
            ],
        }
        monkeypatch.setattr(script_generator, "_call_provider", lambda *a, **kw: (
            outline if len(getattr(monkeypatch, "call_count", [0])) == 0
            else {"segment_type": "intro", "text": "Hook text.", "approx_duration_seconds": 10}
        ))
        calls = []

        def fake_call(*a, **kw):
            calls.append(1)
            if len(calls) == 1:
                return outline
            return {"segment_type": "intro", "text": "Hook text.", "approx_duration_seconds": 10}

        monkeypatch.setattr(script_generator, "_call_provider", fake_call)

        script = generate_script("test", 120)
        seg = script["segments"][0]
        assert seg["subtopic"] == "The Hook"
        assert seg["title"] == "The Hook"

    def test_generate_script_passes_section_context_to_prompts(self, monkeypatch):
        """Each section prompt must include: full outline, prev/current/next topics, and previous section text."""
        outline = {
            "title": "Context Test",
            "sections": [
                {"segment_type": "intro", "topic": "A", "approx_duration_seconds": 10},
                {"segment_type": "content", "topic": "B", "approx_duration_seconds": 20},
                {"segment_type": "outro", "topic": "C", "approx_duration_seconds": 10},
            ],
        }
        section_texts = [
            {"segment_type": "intro", "text": "First.", "approx_duration_seconds": 10},
            {"segment_type": "content", "text": "Second.", "approx_duration_seconds": 20},
            {"segment_type": "outro", "text": "Third.", "approx_duration_seconds": 10},
        ]
        prompts = []

        def fake_call(system_prompt, user_prompt, provider=None, **kwargs):
            prompts.append(user_prompt)
            if len(prompts) == 1:
                return outline
            return section_texts[len(prompts) - 2]

        monkeypatch.setattr(script_generator, "_call_provider", fake_call)

        generate_script("context-topic", 100, duration_minutes=2)

        # 1 outline call + 3 section calls = 4
        assert len(prompts) == 4

        # Section 2 (index 1): middle section
        section_prompt_2 = prompts[2]
        assert "FULL OUTLINE" in section_prompt_2, "Must include full outline"
        assert "PREVIOUS SECTION TOPIC:" in section_prompt_2
        assert "CURRENT SECTION TOPIC:" in section_prompt_2
        assert "NEXT SECTION TOPIC:" in section_prompt_2
        assert "IMMEDIATELY PREVIOUS SECTION TEXT:" in section_prompt_2
        assert "First." in section_prompt_2, "Must include previous section text"

        # Section 1 (index 0): first section — no previous text
        section_prompt_1 = prompts[1]
        assert "None - this is the first section" in section_prompt_1

        # Section 3 (index 2): last section — no next section
        section_prompt_3 = prompts[3]
        assert "None - this is the final section" in section_prompt_3
        assert "Second." in section_prompt_3, "Must include previous section text"


class TestPipelineSegmentDataFlow:
    """Validate that generated script data maps correctly to pipeline DB segments."""

    def test_segment_shape_matches_db_schema(self, monkeypatch):
        """Script segment fields map to pipeline DB columns: text, segment_type, approx_duration_seconds."""
        outline = {
            "title": "DB Shape Test",
            "sections": [
                {"segment_type": "content", "topic": "Topic A", "approx_duration_seconds": 42},
            ],
        }
        monkeypatch.setattr(script_generator, "_call_provider", lambda *a, **kw: (
            outline if not hasattr(monkeypatch, "_db_called")
            else {"segment_type": "content", "text": "Body text here.", "approx_duration_seconds": 42}
        ))
        calls = []

        def fake_call(*a, **kw):
            calls.append(1)
            if len(calls) == 1:
                return outline
            return {"segment_type": "content", "text": "Body text here.", "approx_duration_seconds": 42}

        monkeypatch.setattr(script_generator, "_call_provider", fake_call)

        script = generate_script("db-test", 120)
        seg = script["segments"][0]

        assert seg["text"] == "Body text here."
        assert seg["segment_type"] == "content"
        assert seg["approx_duration_seconds"] == 42
        assert script["title"] == "DB Shape Test"