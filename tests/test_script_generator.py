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
    _get_research_brief_prompt,
    _get_subtopic_research_prompt,
    _get_fact_check_prompt,
    _get_realism_context,
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


class TestProviderRouting:
    def test_openrouter_uses_openrouter_api_key(self, monkeypatch):
        calls = []

        monkeypatch.setattr(script_generator.config, "OPENROUTER_API_KEY", "openrouter-secret")
        monkeypatch.setattr(script_generator.config, "OPENAI_API_KEY", "")
        monkeypatch.setattr(
            script_generator,
            "_call_openai",
            lambda *args, **kwargs: calls.append((args, kwargs)) or {"ok": True},
        )

        result = script_generator._call_provider("system", "user", provider="openrouter")

        assert result == {"ok": True}
        assert calls[0][0][2] == "openrouter-secret"
        assert calls[0][0][5] == script_generator.config.OPENROUTER_BASE_URL

    def test_openai_uses_openai_api_key(self, monkeypatch):
        calls = []

        monkeypatch.setattr(script_generator.config, "OPENAI_API_KEY", "openai-secret")
        monkeypatch.setattr(
            script_generator,
            "_call_openai",
            lambda *args, **kwargs: calls.append((args, kwargs)) or {"ok": True},
        )

        script_generator._call_provider("system", "user", provider="openai")

        assert calls[0][0][2] == "openai-secret"


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
        assert "natural host/guest conversation" in prompt
        assert "do not write dialogue or stage directions" in prompt

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
            {"research_brief": "Lead context"},
            {"key_points": ["Important research point"]},
            duration_minutes=4,
        )

        assert "FULL OUTLINE" in prompt
        assert "1. intro (30s): Hook" in prompt
        assert "CURRENT SECTION TOPIC: Risk framing" in prompt
        assert "PREVIOUS SECTION TOPIC: Hook" in prompt
        assert "NEXT SECTION TOPIC: Takeaway" in prompt
        assert "Welcome text from the prior section." in prompt
        assert "RESEARCH CONTEXT" in prompt
        assert "Important research point" in prompt
        assert "REALISM AND PERFORMANCE" in prompt
        assert "motivated laughs or coughs" in prompt
        assert "Allowed nonverbal performance cues" in prompt

    def test_realism_context_can_disable_nonverbal_cues(self, monkeypatch):
        monkeypatch.setattr(script_generator.config.settings, "allow_nonverbal_cues", False)

        context = _get_realism_context()

        assert "Only [pause] is allowed" in context
        assert "imply laughs or breaths through wording" in context

    def test_research_prompts_seed_conversation_hooks_without_new_claims(self):
        outline = {
            "title": "AI Safety",
            "sections": [{"segment_type": "content", "topic": "Risk framing", "approx_duration_seconds": 45}],
        }

        brief_prompt = _get_research_brief_prompt("AI safety", outline)
        subtopic_prompt = _get_subtopic_research_prompt(
            "AI safety",
            outline,
            outline["sections"][0],
            {"research_brief": "Lead context"},
        )

        assert "recurring metaphor or callback seeds" in brief_prompt
        assert "Treat these as delivery guidance, not new factual claims" in brief_prompt
        assert "likely host follow-up or skeptical interruption" in subtopic_prompt
        assert "banter must not overstate" in subtopic_prompt

    def test_fact_check_prompt_preserves_safe_texture_but_removes_excess(self):
        outline = {
            "title": "AI Safety",
            "sections": [{"segment_type": "content", "topic": "Risk framing", "approx_duration_seconds": 45}],
        }

        prompt = _get_fact_check_prompt(
            "AI safety",
            outline,
            outline["sections"][0],
            {"research_brief": "Lead context"},
            {"key_points": ["Important research point"]},
            "Interviewer: Wait—so what matters here?\\nSME: The key point.",
        )

        assert "keep realistic performance cues" in prompt
        assert "remove only cues that are random, excessive" in prompt

    def test_generate_script_calls_outline_then_each_section_with_context(self, monkeypatch):
        calls = []

        outline = {
            "title": "AI Safety",
            "sections": [
                {"index": 0, "segment_type": "intro", "topic": "Hook", "title": "Hook", "approx_duration_seconds": 30},
                {"index": 1, "segment_type": "content", "topic": "Risk framing", "title": "Risk framing", "approx_duration_seconds": 45},
                {"index": 2, "segment_type": "outro", "topic": "Takeaway", "title": "Takeaway", "approx_duration_seconds": 30},
            ],
        }
        research_brief = {
            "research_brief": "Lead research context",
            "follow_up_questions": ["What angle should we go deeper on?"],
        }
        research_responses = [
            {"topic": "Hook", "key_points": ["Hook research"], "intriguing_angles": ["Surprise"]},
            {"topic": "Risk framing", "key_points": ["Risk research"], "intriguing_angles": ["Tension"]},
            {"topic": "Takeaway", "key_points": ["Takeaway research"], "intriguing_angles": ["Resolution"]},
        ]
        section_responses = [
            {"segment_type": "intro", "text": "Intro text.", "approx_duration_seconds": 30},
            {"segment_type": "content", "text": "Risk text.", "approx_duration_seconds": 45},
            {"segment_type": "outro", "text": "Outro text.", "approx_duration_seconds": 30},
        ]

        def fake_call_provider(system_prompt, user_prompt, provider=None, **kwargs):
            calls.append((system_prompt, user_prompt, provider, kwargs))
            if "outline planner" in system_prompt:
                return outline
            if "lead research agent" in system_prompt:
                return research_brief
            if "subtopic research agent" in system_prompt:
                return research_responses[len([call for call in calls if "subtopic research agent" in call[0]]) - 1]
            if "factfulness verification agent" in system_prompt:
                verified_index = len([call for call in calls if "factfulness verification agent" in call[0]]) - 1
                return {"outcome": "accepted", "issues": [], "verified_text": section_responses[verified_index]["text"]}
            return section_responses[len([call for call in calls if "script writer" in call[0]]) - 1]

        monkeypatch.setattr(script_generator, "_call_provider", fake_call_provider)

        script = generate_script(
            "AI safety",
            128,
            duration_minutes=5,
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
        assert len(calls) == 11
        assert "outline planner" in calls[0][0]
        assert "sections" in calls[0][1]
        assert "lead research agent" in calls[1][0]
        assert all("subtopic research agent" in call[0] for call in calls[2:5])
        assert all("script writer" in call[0] for call in (calls[5], calls[7], calls[9]))
        assert all("factfulness verification agent" in call[0] for call in (calls[6], calls[8], calls[10]))
        assert all(call[2] == "test-provider" for call in calls)
        assert calls[0][3]["model"] == "test-model"
        assert calls[0][3]["snapshot"] is None

        second_section_prompt = calls[7][1]
        assert "FULL OUTLINE" in second_section_prompt
        assert "PREVIOUS SECTION TOPIC: Hook" in second_section_prompt
        assert "CURRENT SECTION TOPIC: Risk framing" in second_section_prompt
        assert "NEXT SECTION TOPIC: Takeaway" in second_section_prompt
        assert "IMMEDIATELY PREVIOUS SECTION TEXT:\nIntro text." in second_section_prompt
        assert "RESEARCH CONTEXT" in second_section_prompt
        assert "Interviewer" in second_section_prompt
        assert "SME guest" in second_section_prompt
        assert "Risk research" in second_section_prompt

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
        with pytest.raises(ValueError, match="unknown_llm_provider"):
            generate_follow_up_questions("AI", SAMPLE_SCRIPT, provider="nonexistent")

    def test_raises_openai_without_key(self):
        """OpenAI dispatch should fail because there's no real key configured."""
        with pytest.raises((ValueError, ImportError)):
            generate_follow_up_questions("AI", SAMPLE_SCRIPT, provider="openai", api_key="")


class TestGenerateSummaryDispatch:
    def test_raises_on_unknown_provider(self):
        with pytest.raises(ValueError, match="unknown_llm_provider"):
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
                {"index": 0, "segment_type": "intro", "topic": "Start", "title": "Start", "approx_duration_seconds": 15},
                {"index": 1, "segment_type": "content", "topic": "Middle", "title": "Middle", "approx_duration_seconds": 30},
            ],
        }
        responses = [
            {"segment_type": "intro", "text": "Hello.", "approx_duration_seconds": 15},
            {"segment_type": "content", "text": "World.", "approx_duration_seconds": 30},
        ]
        calls = []

        def fake_call(system_prompt, user_prompt, provider=None, **kwargs):
            calls.append(system_prompt)
            if "outline planner" in system_prompt:
                return outline
            if "lead research agent" in system_prompt:
                return {"research_brief": "Brief"}
            if "subtopic research agent" in system_prompt:
                return {"key_points": ["Research point"]}
            if "factfulness verification agent" in system_prompt:
                index = len([call for call in calls if "factfulness verification agent" in call]) - 1
                return {
                    "is_factful": True,
                    "issues": [],
                    "verified_text": responses[index]["text"],
                }
            return responses[len([call for call in calls if "script writer" in call]) - 1]

        monkeypatch.setattr(script_generator, "_call_provider", fake_call)

        script = generate_script("test", 120, duration_minutes=3)
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
                {"index": 0, "segment_type": "intro", "topic": "The Hook", "title": "The Hook", "approx_duration_seconds": 10},
            ],
        }
        monkeypatch.setattr(script_generator, "_call_provider", lambda *a, **kw: (
            outline if len(getattr(monkeypatch, "call_count", [0])) == 0
            else {"segment_type": "intro", "text": "Hook text.", "approx_duration_seconds": 10}
        ))
        calls = []

        def fake_call(system_prompt, *args, **kwargs):
            calls.append(system_prompt)
            if "outline planner" in system_prompt:
                return outline
            if "lead research agent" in system_prompt:
                return {"research_brief": "Brief"}
            if "subtopic research agent" in system_prompt:
                return {"key_points": ["Research point"]}
            if "factfulness verification agent" in system_prompt:
                return {
                    "is_factful": True,
                    "issues": [],
                    "verified_text": "Hook text.",
                }
            return {"segment_type": "intro", "text": "Hook text.", "approx_duration_seconds": 10}

        monkeypatch.setattr(script_generator, "_call_provider", fake_call)

        script = generate_script("test", 120, duration_minutes=1)
        seg = script["segments"][0]
        assert seg["subtopic"] == "The Hook"
        assert seg["title"] == "The Hook"

    def test_generate_script_passes_section_context_to_prompts(self, monkeypatch):
        """Each section prompt must include: full outline, prev/current/next topics, and previous section text."""
        outline = {
            "title": "Context Test",
            "sections": [
                {"index": 0, "segment_type": "intro", "topic": "A", "title": "A", "approx_duration_seconds": 10},
                {"index": 1, "segment_type": "content", "topic": "B", "title": "B", "approx_duration_seconds": 20},
                {"index": 2, "segment_type": "outro", "topic": "C", "title": "C", "approx_duration_seconds": 10},
            ],
        }
        section_texts = [
            {"segment_type": "intro", "text": "First.", "approx_duration_seconds": 10},
            {"segment_type": "content", "text": "Second.", "approx_duration_seconds": 20},
            {"segment_type": "outro", "text": "Third.", "approx_duration_seconds": 10},
        ]
        prompts = []

        def fake_call(system_prompt, user_prompt, provider=None, **kwargs):
            if "outline planner" in system_prompt:
                return outline
            if "lead research agent" in system_prompt:
                return {"research_brief": "Brief"}
            if "subtopic research agent" in system_prompt:
                return {"key_points": ["Research point"]}
            if "factfulness verification agent" in system_prompt:
                verified_index = len([prompt for prompt in prompts if "DRAFT DIALOGUE:" not in prompt]) - 1
                return {"outcome": "accepted", "issues": [], "verified_text": section_texts[verified_index]["text"]}
            prompts.append(user_prompt)
            return section_texts[len(prompts) - 1]

        monkeypatch.setattr(script_generator, "_call_provider", fake_call)

        generate_script("context-topic", 100, duration_minutes=5)

        # 3 section writing calls after outline and research calls
        assert len(prompts) == 3

        # Section 2 (index 1): middle section
        section_prompt_2 = prompts[1]
        assert "FULL OUTLINE" in section_prompt_2, "Must include full outline"
        assert "PREVIOUS SECTION TOPIC:" in section_prompt_2
        assert "CURRENT SECTION TOPIC:" in section_prompt_2
        assert "NEXT SECTION TOPIC:" in section_prompt_2
        assert "IMMEDIATELY PREVIOUS SECTION TEXT:" in section_prompt_2
        assert "First." in section_prompt_2, "Must include previous section text"

        # Section 1 (index 0): first section — no previous text
        section_prompt_1 = prompts[0]
        assert "None - this is the first section" in section_prompt_1

        # Section 3 (index 2): last section — no next section
        section_prompt_3 = prompts[2]
        assert "None - this is the final section" in section_prompt_3
        assert "Second." in section_prompt_3, "Must include previous section text"


class TestPipelineSegmentDataFlow:
    """Validate that generated script data maps correctly to pipeline DB segments."""

    def test_segment_shape_matches_db_schema(self, monkeypatch):
        """Script segment fields map to pipeline DB columns: text, segment_type, approx_duration_seconds."""
        outline = {
            "title": "DB Shape Test",
            "sections": [
                {"index": 0, "segment_type": "content", "topic": "Topic A", "title": "Topic A", "approx_duration_seconds": 42},
            ],
        }
        monkeypatch.setattr(script_generator, "_call_provider", lambda *a, **kw: (
            outline if not hasattr(monkeypatch, "_db_called")
            else {"segment_type": "content", "text": "Body text here.", "approx_duration_seconds": 42}
        ))
        calls = []

        def fake_call(system_prompt, *args, **kwargs):
            calls.append(system_prompt)
            if "outline planner" in system_prompt:
                return outline
            if "lead research agent" in system_prompt:
                return {"research_brief": "Brief"}
            if "subtopic research agent" in system_prompt:
                return {"key_points": ["Research point"]}
            if "factfulness verification agent" in system_prompt:
                return {
                    "is_factful": True,
                    "issues": [],
                    "verified_text": "Body text here.",
                }
            return {"segment_type": "content", "text": "Body text here.", "approx_duration_seconds": 42}

        monkeypatch.setattr(script_generator, "_call_provider", fake_call)

        script = generate_script("db-test", 120, duration_minutes=1)
        seg = script["segments"][0]

        assert seg["text"] == "Body text here."
        assert seg["segment_type"] == "content"
        assert seg["approx_duration_seconds"] == 42
        assert script["title"] == "DB Shape Test"