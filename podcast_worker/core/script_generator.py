"""Script Generator - Uses an LLM to generate podcast scripts.

Supports OpenAI, Ollama, and OpenRouter via a unified dispatch.
Each operation type (script gen, follow-up, summary) has one entry
point that delegates to a shared provider client.
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

from . import config
from .config import ResolvedExecutionSnapshot, RoutingConfigurationError


_VALID_SEGMENT_TYPES = frozenset({"intro", "content", "outro"})


def required_segment_count(duration_minutes: int) -> int:
    """Return the server-owned, bounded segment count for a valid duration."""
    if isinstance(duration_minutes, bool) or not isinstance(duration_minutes, int):
        raise ValueError("duration_minutes must be an integer")
    if not 1 <= duration_minutes <= 30:
        raise ValueError("duration_minutes must be between 1 and 30")
    return min(max(math.ceil(duration_minutes / 2), 1), 12)


def validate_script_outline(outline: dict, duration_minutes: int) -> dict:
    """Validate provider outline shape without trimming, padding, or coercion."""
    expected_count = required_segment_count(duration_minutes)
    if not isinstance(outline, dict):
        raise ValueError("outline must be an object")
    if not isinstance(outline.get("title"), str) or not outline["title"].strip():
        raise ValueError("outline title must be nonblank")

    sections = outline.get("sections")
    if not isinstance(sections, list) or len(sections) != expected_count:
        raise ValueError(f"outline must contain exactly {expected_count} sections")

    for expected_index, section in enumerate(sections):
        if not isinstance(section, dict):
            raise ValueError(f"outline section {expected_index} must be an object")
        if section.get("index") != expected_index:
            raise ValueError("outline section indices must be contiguous from zero")
        if section.get("segment_type") not in _VALID_SEGMENT_TYPES:
            raise ValueError(f"outline section {expected_index} has invalid segment_type")
        for field in ("topic", "title"):
            value = section.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"outline section {expected_index} {field} must be nonblank")
        planned_duration = section.get("approx_duration_seconds")
        if (
            isinstance(planned_duration, bool)
            or not isinstance(planned_duration, (int, float))
            or planned_duration <= 0
        ):
            raise ValueError(
                f"outline section {expected_index} approx_duration_seconds must be positive"
            )
    return outline


# ---------------------------------------------------------------------------
# Shared LLM client wrappers
# ---------------------------------------------------------------------------


def _strict_json(text: str, dialect: str) -> dict[str, Any]:
    """Decode provider output without accepting arbitrary surrounding prose."""
    value = text.strip()
    if dialect == "openrouter_legacy_prompt_strict_parse" and value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RoutingConfigurationError("structured_output_failure") from exc
    if not isinstance(parsed, dict):
        raise RoutingConfigurationError("structured_output_failure")
    return parsed


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.8,
    base_url: Optional[str] = None,
    dialect: str = "openai_json_object",
) -> dict[str, Any]:
    """Call an OpenAI-compatible endpoint using the route's declared JSON dialect."""
    from openai import OpenAI

    key = api_key
    if not key:
        raise ValueError("API key not set for the selected OpenAI-compatible provider.")
    client_kwargs: dict[str, Any] = {"api_key": key}
    if base_url:
        client_kwargs["base_url"] = base_url
    options: dict[str, Any] = {"temperature": temperature}
    if dialect == "openai_json_object":
        options["response_format"] = {"type": "json_object"}
    elif dialect != "openrouter_legacy_prompt_strict_parse":
        raise RoutingConfigurationError("unsupported_llm_dialect")
    if base_url:
        options["extra_headers"] = {"HTTP-Referer": "https://github.com/straub-ah", "X-Title": "BPM Podcast Generator"}
    response = OpenAI(**client_kwargs).chat.completions.create(
        model=model or config.OPENAI_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        **options,
    )
    return _strict_json(response.choices[0].message.content or "", dialect)


def _call_ollama(
    system_prompt: str,
    user_prompt: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Call Ollama's explicitly declared JSON mode."""
    import requests

    response = requests.post(
        f"{base_url or config.OLLAMA_BASE_URL}/api/generate",
        json={"model": model or config.OLLAMA_MODEL, "prompt": f"{system_prompt}\n\n{user_prompt}", "stream": False, "format": "json"},
        timeout=120,
    )
    response.raise_for_status()
    return _strict_json(response.json().get("response", ""), "ollama_format_json")


def _call_provider(
    system_prompt: str,
    user_prompt: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.8,
    *,
    snapshot: ResolvedExecutionSnapshot | None = None,
    purpose: str | None = None,
) -> dict[str, Any]:
    """Dispatch a purpose through an immutable snapshot, or legacy scalar routing."""
    route = snapshot.route_for(purpose) if snapshot and purpose else None
    selected_provider = route.provider if route else (provider or config.LLM_PROVIDER)
    selected_model = route.model if route else model
    selected_temperature = route.temperature if route else temperature
    if selected_provider == "openai":
        return _call_openai(system_prompt, user_prompt, api_key or config.OPENAI_API_KEY, selected_model, selected_temperature, dialect=route.dialect if route else "openai_json_object")
    if selected_provider == "ollama":
        return _call_ollama(system_prompt, user_prompt, model=selected_model)
    if selected_provider == "openrouter":
        return _call_openai(system_prompt, user_prompt, api_key or config.OPENROUTER_API_KEY, selected_model or config.OPENROUTER_MODEL, selected_temperature, config.OPENROUTER_BASE_URL, route.dialect if route else "openrouter_legacy_prompt_strict_parse")
    raise RoutingConfigurationError("unknown_llm_provider")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _get_script_style_context(bpm: int, duration_minutes: int = 5) -> str:
    """Build shared style guidance for script prompts."""
    if bpm >= 160:
        energy = "HIGH ENERGY"
        pace = "fast-paced, punchy, exciting — like a workout hype session"
        sentence_length = "short to medium sentences, crisp delivery"
    elif bpm >= 120:
        energy = "MODERATE ENERGY"
        pace = "steady, engaging, upbeat but not rushing"
        sentence_length = "mix of short and medium sentences"
    elif bpm >= 90:
        energy = "CHILL ENERGY"
        pace = "relaxed, conversational, easygoing"
        sentence_length = "medium sentences, natural pauses"
    else:
        energy = "LOW ENERGY / DEEP FOCUS"
        pace = "calm, deliberate, thoughtful"
        sentence_length = "medium to longer sentences, reflective pauses"

    return f"""ENERGY LEVEL: {energy}
BPM: {bpm}

The podcast will be played over a beat at {bpm} BPM, so the {pace}.

Important rules:
- {sentence_length}
- Use natural conversational language — like a knowledgeable friend explaining something
- Write for the ear: contractions, short speakable lines, active voice, and clear first-listen explanations
- Avoid markdown, lists, or special formatting
- Include brief pauses marked with [pause]
- The entire script should take about {duration_minutes} minutes to read at natural pace
- Do NOT use production sound effect notations like [music], [applause], [rimshot], or [sfx]
- NO special characters or emojis"""


def _get_outline_prompt(topic: str, bpm: int, duration_minutes: int = 5) -> str:
    """Build the outline prompt for the first LLM step."""
    segment_count = required_segment_count(duration_minutes)
    style_context = _get_script_style_context(bpm, duration_minutes)

    return f"""You are planning a {duration_minutes}-minute educational dialogue podcast built around natural host/guest conversation.

TOPIC: {topic}
{style_context}

Create exactly {segment_count} ordered sections before any script text is written. The sections together must cover the full {duration_minutes}-minute target duration. Do not add, omit, merge, or pad sections.
Plan the conversation only; do not write dialogue or stage directions.

Return ONLY valid JSON with this exact structure:
{{
    "title": "Episode title",
    "sections": [
        {{
            "index": 0,
            "segment_type": "intro",
            "topic": "Specific section topic",
            "title": "Section title",
            "approx_duration_seconds": 60
        }}
    ]
}}

Requirements:
- sections contains exactly {segment_count} objects
- index values are exactly 0 through {segment_count - 1}, in order
- segment_type is exactly intro, content, or outro
- topic and title are nonblank
- approx_duration_seconds is positive
- total target duration is about {duration_minutes * 60} seconds
"""


def _speaker_profile(label: str, profile: Optional[dict]) -> str:
    """Format a compact dynamic persona instead of creating per-run files."""
    profile = profile or {}
    name = profile.get("name") or label
    voice = profile.get("voice") or "natural conversational voice"
    tone = profile.get("tone") or "curious, warm, and precise"
    humor = profile.get("humor") or "light and occasional"
    style = profile.get("style") or profile.get("expertise") or "clear and accessible"
    return f"{label}: name={name}; voice={voice}; tone={tone}; humor={humor}; style={style}"


def _format_outline_for_prompt(outline: dict) -> str:
    """Format the full outline for section-generation prompts."""
    lines = [f"TITLE: {outline.get('title', 'Untitled')}"]
    for index, section in enumerate(outline.get("sections", []), start=1):
        segment_type = section.get("segment_type", "content")
        topic = section.get("topic") or section.get("title") or f"Section {index}"
        duration = section.get("approx_duration_seconds", 45)
        lines.append(f"{index}. {segment_type} ({duration}s): {topic}")
    return "\n".join(lines)


def _get_research_brief_prompt(topic: str, outline: dict) -> str:
    """Build a prompt for a coordinator research agent."""
    return f"""You are the lead research agent for a podcast production team.

EPISODE TOPIC: {topic}

APPROVED OUTLINE:
{_format_outline_for_prompt(outline)}

Create a research brief that delegates one focused research lane to each outline section. Include clarifying follow-up questions the host could ask the listener if the topic needs sharper intent before script writing. Include presentation hooks the dialogue agents can use later: recurring metaphor or callback seeds, host skepticism angles, guest nuance boundaries, and places where uncertainty should be explicitly acknowledged. Treat these as delivery guidance, not new factual claims.

Return ONLY valid JSON with this exact structure:
{{
    "research_brief": "Short synthesis of the most important angle for the episode",
    "audience_assumptions": ["Assumption 1", "Assumption 2"],
    "follow_up_questions": ["Question that would clarify what the listener wants to learn"],
    "subtopics": [
        {{
            "section_index": 0,
            "topic": "Section topic",
            "research_questions": ["Question 1", "Question 2"],
            "source_keywords": ["keyword 1", "keyword 2"]
        }}
    ]
}}
"""


def _get_subtopic_research_prompt(topic: str, outline: dict, section: dict, research_brief: dict) -> str:
    """Build a prompt for one subtopic research agent."""
    section_topic = section.get("topic") or section.get("title") or topic

    return f"""You are a subtopic research agent. Research only the assigned section and return concise, script-ready notes.

EPISODE TOPIC: {topic}
ASSIGNED SECTION TOPIC: {section_topic}

FULL OUTLINE:
{_format_outline_for_prompt(outline)}

LEAD RESEARCH BRIEF:
{json.dumps(research_brief, ensure_ascii=False)}

Find the ideas, examples, definitions, tensions, and open questions that would make this section specific and useful. If the section needs more user intent, include follow_up_questions that can be shown to the user before final generation. Include section-level conversation opportunities: one concrete example, one likely host follow-up or skeptical interruption, any callback to earlier outline sections, and cautions that banter must not overstate.

Return ONLY valid JSON with this exact structure:
{{
    "topic": "{section_topic}",
    "key_points": ["Specific point 1", "Specific point 2"],
    "examples": ["Concrete example 1"],
    "intriguing_angles": ["Hook, tension, or surprising framing"],
    "follow_up_questions": ["Optional question for the user"],
    "cautions": ["Nuance or uncertainty to avoid overstating"]
}}
"""


def _format_research_for_prompt(research_brief: dict, section_research: dict) -> str:
    """Format research context for the script-writing prompt."""
    return "\n".join(
        [
            "LEAD RESEARCH BRIEF:",
            json.dumps(research_brief, ensure_ascii=False),
            "",
            "SECTION RESEARCH:",
            json.dumps(section_research, ensure_ascii=False),
        ]
    )


def _format_speaker_profiles(interviewer_profile: Optional[dict], sme_profile: Optional[dict]) -> str:
    """Format the two dialogue agents for prompt context."""
    return "\n".join(
        [
            _speaker_profile("Interviewer", interviewer_profile),
            _speaker_profile("SME guest", sme_profile),
        ]
    )


def _get_realism_context() -> str:
    """Build guidance for realistic, TTS-friendly podcast dialogue."""
    level = config.settings.realism_level
    allow_nonverbal = config.settings.allow_nonverbal_cues

    if level == "subtle":
        frequency = "Use one small human texture moment about every 8-12 dialogue turns."
        intensity = "Keep the show polished and mostly clean."
    elif level == "expressive":
        frequency = "Use one small human texture moment about every 4-6 dialogue turns when motivated."
        intensity = "Allow more warmth, surprise, and playful host chemistry, but never turn it into parody."
    else:
        frequency = "Use one small human texture moment about every 6-8 dialogue turns when motivated."
        intensity = "Aim for edited naturalism: clean enough to follow, imperfect enough to feel human."

    if allow_nonverbal:
        cue_rule = (
            "Allowed nonverbal performance cues are [pause], [laughs], [small laugh], "
            "[sighs], [clears throat], and [coughs]. Use [coughs] at most once in an entire episode "
            "and only when it serves a joke, nervous reset, or self-correction."
        )
    else:
        cue_rule = "Only [pause] is allowed as a bracketed cue; imply laughs or breaths through wording instead."

    return f"""REALISTIC PODCAST DIALOGUE GUIDANCE:
- {intensity}
- {frequency}
- Prefer conversational behavior over filler: correction, clarification, surprise, mild disagreement, callbacks, quick summaries, and listener-aware framing.
- Vary turn length. Mix short reactions, skeptical questions, concise explanations, and occasional one-line jokes; avoid perfect paragraph-by-paragraph alternation.
- Give the Interviewer and SME distinct rhythms. The Interviewer can interrupt gently, challenge, recap, or represent listener confusion. The SME can correct, qualify, and ground claims in examples.
- Use fillers such as "uh", "I mean", "you know", and "let me rephrase that" only when they reveal thought, uncertainty, or self-correction.
- Let speakers occasionally overlap in spirit with phrases like "Wait—", "Hold on", "Right, but", or "Exactly", but keep the transcript readable for TTS.
- {cue_rule}
- Do not add random laughs, coughs, or throat-clearing. Every cue must have a conversational reason.
- Do not sacrifice factual precision for banter. If the facts are complex, have a speaker pause and restate them clearly."""


def _get_section_prompt(
    topic: str,
    bpm: int,
    outline: dict,
    section: dict,
    previous_section: Optional[dict],
    next_section: Optional[dict],
    previous_section_text: str,
    research_brief: Optional[dict] = None,
    section_research: Optional[dict] = None,
    interviewer_profile: Optional[dict] = None,
    sme_profile: Optional[dict] = None,
    duration_minutes: int = 5,
) -> str:
    """Build a prompt for generating one outlined section."""
    style_context = _get_script_style_context(bpm, duration_minutes)
    current_topic = section.get("topic") or section.get("title") or topic
    previous_topic = (
        (previous_section or {}).get("topic")
        or (previous_section or {}).get("title")
        or "None - this is the first section"
    )
    next_topic = (
        (next_section or {}).get("topic")
        or (next_section or {}).get("title")
        or "None - this is the final section"
    )
    previous_text = previous_section_text or "None - this is the first section"
    segment_type = section.get("segment_type", "content")
    duration = section.get("approx_duration_seconds", 45)

    return f"""You are writing one section of a {duration_minutes}-minute solo educational podcast.

EPISODE TOPIC: {topic}
{style_context}

FULL OUTLINE:
{_format_outline_for_prompt(outline)}

CURRENT SECTION TOPIC: {current_topic}
CURRENT SEGMENT TYPE: {segment_type}
CURRENT TARGET DURATION SECONDS: {duration}
PREVIOUS SECTION TOPIC: {previous_topic}
NEXT SECTION TOPIC: {next_topic}
IMMEDIATELY PREVIOUS SECTION TEXT:
{previous_text}

RESEARCH CONTEXT:
{_format_research_for_prompt(research_brief or {}, section_research or {})}

SPEAKER AGENTS:
{_format_speaker_profiles(interviewer_profile, sme_profile)}

REALISM AND PERFORMANCE:
{_get_realism_context()}

Write ONLY the current section as a natural two-person podcast dialogue between the Interviewer and the SME guest. The interviewer should ask sharp, curious follow-ups and steer pacing; the SME should provide expertise, examples, and nuance. Use the previous section text for continuity, but do not repeat it. Use the research context for specificity, examples, nuance, and intriguing framing. Add realistic host/guest texture where it helps: brief reactions, self-corrections, callbacks, mild disagreement, motivated laughs or coughs, and listener-oriented recaps. Lead naturally toward the next topic when one exists.

Return ONLY valid JSON with this exact structure:
{{
    "segment_type": "{segment_type}",
    "text": "Interviewer: Question or setup.\\nSME: Expert answer.\\nInterviewer: Follow-up...",
    "approx_duration_seconds": {duration}
}}
"""


def _get_follow_up_prompt(topic: str, script: dict) -> str:
    """Build a prompt that asks the LLM to generate follow-up questions."""
    full_text = flatten_script(script)
    title = script.get("title", "Untitled")

    return f"""You are a podcast host who just finished an episode.

EPISODE TITLE: {title}
TOPIC: {topic}
SCRIPT: {full_text}

Generate 3 thoughtful follow-up questions that a listener might ask to go deeper into this topic.
The questions should be:
- Open-ended and thought-provoking (not yes/no)
- Related to the content but exploring angles not fully covered in the script
- Written in natural, conversational language
- Varied in complexity (one beginner-friendly, one intermediate, one advanced)

Return ONLY valid JSON with this exact structure:
{{
    "follow_up_questions": [
        {{
            "question": "The full question text?",
            "context": "Brief context on why this question is relevant (1 sentence)",
            "difficulty": "beginner" | "intermediate" | "advanced"
        }}
    ]
}}
"""


def _get_summary_prompt(script: dict) -> str:
    """Build a prompt that asks the LLM to summarize a podcast script."""
    full_text = flatten_script(script)
    title = script.get("title", "Untitled")

    return f"""You are given a podcast script.

EPISODE TITLE: {title}
SCRIPT: {full_text}

Generate a concise but informative summary of this podcast episode. The summary should:
- Be 2-3 paragraphs (about 150-250 words total)
- Capture the main thesis and key points
- Mention the tone/energy level implied by the content
- End with one "key takeaway" sentence
- Be written in a neutral, informative tone

Return ONLY valid JSON with this exact structure:
{{
    "title": "{title}",
    "summary": "The full summary text...",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "key_takeaway": "The single most important takeaway."
}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_script_outline(
    topic: str,
    bpm: int,
    duration_minutes: int = 5,
    provider: Optional[str] = None,
    snapshot: ResolvedExecutionSnapshot | None = None,
    **kwargs,
) -> dict:
    """Generate and strictly validate a podcast section outline."""
    outline_prompt = _get_outline_prompt(topic, bpm, duration_minutes)
    outline = _call_provider(
        "You are a podcast outline planner. You output raw JSON only.",
        outline_prompt,
        provider=provider,
        snapshot=snapshot,
        purpose="outline",
        **kwargs,
    )
    return validate_script_outline(outline, duration_minutes)


def generate_research_brief(
    topic: str,
    outline: dict,
    provider: Optional[str] = None,
    snapshot: ResolvedExecutionSnapshot | None = None,
    **kwargs,
) -> dict:
    """Run the lead research-agent step for an approved outline."""
    return _call_provider(
        "You are the lead research agent for a podcast production team. You output raw JSON only.",
        _get_research_brief_prompt(topic, outline),
        provider=provider,
        temperature=0.4,
        snapshot=snapshot,
        purpose="research_brief",
        **kwargs,
    )


def generate_subtopic_research(
    topic: str,
    outline: dict,
    section: dict,
    research_brief: dict,
    provider: Optional[str] = None,
    snapshot: ResolvedExecutionSnapshot | None = None,
    **kwargs,
) -> dict:
    """Run one subtopic research-agent step for an outline section."""
    return _call_provider(
        "You are a subtopic research agent for a podcast production team. You output raw JSON only.",
        _get_subtopic_research_prompt(topic, outline, section, research_brief),
        provider=provider,
        temperature=0.4,
        snapshot=snapshot,
        purpose="subtopic_research",
        **kwargs,
    )


def generate_section_draft(
    topic: str,
    bpm: int,
    duration_minutes: int,
    outline: dict,
    section: dict,
    research_brief: dict,
    section_research: dict,
    previous_section_text: str = "",
    interviewer_profile: Optional[dict] = None,
    sme_profile: Optional[dict] = None,
    provider: Optional[str] = None,
    **kwargs,
) -> dict:
    """Generate one internal section draft for durable staged execution."""
    sections = outline.get("sections", [])
    try:
        index = sections.index(section)
    except ValueError as exc:
        raise ValueError("section is not part of the accepted outline") from exc
    prompt = _get_section_prompt(
        topic,
        bpm,
        outline,
        section,
        sections[index - 1] if index > 0 else None,
        sections[index + 1] if index + 1 < len(sections) else None,
        previous_section_text,
        research_brief,
        section_research,
        interviewer_profile,
        sme_profile,
        duration_minutes,
    )
    result = _call_provider(
        "You are a podcast script writer. You output raw JSON only.",
        prompt,
        provider=provider,
        **kwargs,
    )
    segment = result.get("segment", result)
    text = segment.get("text") if isinstance(segment, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("section draft must contain nonblank text")
    return segment


def validated_fact_check_result(result: dict) -> dict:
    """Validate verifier output before any public text is persisted."""
    if not isinstance(result, dict) or not isinstance(result.get("is_factful"), bool):
        raise ValueError("fact check result is malformed")
    issues = result.get("issues")
    if not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues):
        raise ValueError("fact check issues must be a string array")
    verified_text = result.get("verified_text")
    if not isinstance(verified_text, str) or not verified_text.strip():
        raise ValueError("fact check result must contain nonblank verified_text")
    return result


def _get_fact_check_prompt(
    topic: str,
    outline: dict,
    section: dict,
    research_brief: dict,
    section_research: dict,
    generated_text: str,
) -> str:
    """Build a prompt for the factfulness verification agent."""
    section_topic = section.get("topic") or section.get("title") or topic
    return f"""You are the factfulness verification agent for a dialogue podcast script.

EPISODE TOPIC: {topic}
SECTION TOPIC: {section_topic}

FULL OUTLINE:
{_format_outline_for_prompt(outline)}

RESEARCH CONTEXT:
{_format_research_for_prompt(research_brief, section_research)}

DRAFT DIALOGUE:
{generated_text}

Verify that the dialogue is faithful to the research context, does not overstate uncertain claims, and keeps the interviewer/SME roles clear. If a correction is needed, return corrected dialogue in verified_text. Preserve the two-speaker dialogue format and keep realistic performance cues when they do not distort the facts; remove only cues that are random, excessive, or make a factual claim sound uncertain when it should be clear.

Return ONLY valid JSON with this exact structure:
{{
    "outcome": "accepted" | "corrected" | "blocked",
    "issues": ["Issue or empty list"],
    "verified_text": "Original/corrected dialogue, or null when blocked"
}}
"""


@dataclass(frozen=True)
class VerificationResult:
    outcome: Literal["accepted", "corrected", "blocked"]
    issues: tuple[str, ...]
    verified_text: str | None


def parse_verification_result(value: dict[str, Any], draft: str) -> VerificationResult:
    """Strict verifier gate: no unverified draft can reach TTS."""
    if "outcome" not in value and isinstance(value.get("is_factful"), bool):
        value = {
            "outcome": "accepted" if value["is_factful"] else "blocked",
            "issues": value.get("issues", []),
            "verified_text": value.get("verified_text") if value["is_factful"] else None,
        }
    outcome = value.get("outcome")
    issues = value.get("issues")
    verified_text = value.get("verified_text")
    if outcome not in {"accepted", "corrected", "blocked"} or not isinstance(issues, list) or any(not isinstance(issue, str) or not issue.strip() for issue in issues):
        raise RoutingConfigurationError("structured_output_failure")
    if outcome == "accepted" and verified_text == draft and not issues:
        return VerificationResult(outcome, tuple(), draft)
    if outcome == "corrected" and isinstance(verified_text, str) and verified_text.strip() and issues:
        return VerificationResult(outcome, tuple(issues), verified_text)
    if outcome == "blocked" and verified_text is None and issues:
        return VerificationResult(outcome, tuple(issues), None)
    raise RoutingConfigurationError("structured_output_failure")


def verify_section_factfulness(
    topic: str,
    outline: dict,
    section: dict,
    research_brief: dict,
    section_research: dict,
    generated_text: str,
    provider: Optional[str] = None,
    snapshot: ResolvedExecutionSnapshot | None = None,
    **kwargs,
) -> VerificationResult:
    """Run the typed factfulness verifier for one generated section."""
    value = _call_provider(
        "You are a factfulness verification agent for a podcast production team. You output raw JSON only.",
        _get_fact_check_prompt(topic, outline, section, research_brief, section_research, generated_text),
        provider=provider,
        temperature=0.2,
        snapshot=snapshot,
        purpose="fact_verification",
        **kwargs,
    )
    return parse_verification_result(value, generated_text)


def generate_verified_section(
    topic: str,
    outline: dict,
    section: dict,
    research_brief: dict,
    section_research: dict,
    generated_text: str,
    provider: Optional[str] = None,
    **kwargs,
) -> VerificationResult:
    """Verify one draft and reject malformed results before publication."""
    return verify_section_factfulness(
        topic=topic,
        outline=outline,
        section=section,
        research_brief=research_brief,
        section_research=section_research,
        generated_text=generated_text,
        provider=provider,
        **kwargs,
    )


def generate_script(
    topic: str,
    bpm: int,
    duration_minutes: int = 5,
    provider: Optional[str] = None,
    outline: Optional[dict] = None,
    interviewer_profile: Optional[dict] = None,
    sme_profile: Optional[dict] = None,
    snapshot: ResolvedExecutionSnapshot | None = None,
    **kwargs,
) -> dict:
    """Generate a podcast script using outline-first section generation."""
    if outline is None:
        outline = generate_script_outline(
            topic=topic,
            bpm=bpm,
            duration_minutes=duration_minutes,
            provider=provider,
            snapshot=snapshot,
            **kwargs,
        )

    sections = outline.get("sections", [])

    research_brief = generate_research_brief(
        topic=topic,
        outline=outline,
        provider=provider,
        snapshot=snapshot,
        **kwargs,
    )
    section_research_items = [
        generate_subtopic_research(
            topic=topic,
            outline=outline,
            section=section,
            research_brief=research_brief,
            provider=provider,
            snapshot=snapshot,
            **kwargs,
        )
        for section in sections
    ]

    generated_segments = []
    previous_section_text = ""

    for index, section in enumerate(sections):
        segment = generate_section_draft(
            topic=topic,
            bpm=bpm,
            duration_minutes=duration_minutes,
            outline=outline,
            section=section,
            research_brief=research_brief,
            section_research=section_research_items[index],
            previous_section_text=previous_section_text,
            interviewer_profile=interviewer_profile,
            sme_profile=sme_profile,
            provider=provider,
            snapshot=snapshot,
            purpose="dialogue_draft",
            **kwargs,
        )
        verification = generate_verified_section(
            topic=topic,
            outline=outline,
            section=section,
            research_brief=research_brief,
            section_research=section_research_items[index],
            generated_text=segment["text"],
            provider=provider,
            snapshot=snapshot,
            **kwargs,
        )
        if verification.outcome == "blocked" or verification.verified_text is None:
            raise RoutingConfigurationError("validation_failed")
        generated_segments.append(
            {
                "segment_type": segment.get(
                    "segment_type",
                    section.get("segment_type", "content"),
                ),
                "subtopic": section.get("topic") or section.get("title") or topic,
                "title": section.get("title") or section.get("topic"),
                "text": verification.verified_text,
                "approx_duration_seconds": segment.get(
                    "approx_duration_seconds",
                    section.get("approx_duration_seconds", 45),
                ),
            }
        )
        previous_section_text = generated_segments[-1]["text"]

    return {
        "title": outline.get("title", "Untitled"),
        "segments": generated_segments,
    }


def flatten_script(script: dict) -> str:
    """Flatten a script dict into a single text string for TTS."""
    segments = script.get("segments", [])
    return " ".join(seg["text"] for seg in segments)


def generate_follow_up_questions(
    topic: str,
    script: dict,
    provider: Optional[str] = None,
    **kwargs,
) -> dict:
    """Generate follow-up questions based on a podcast script.

    Args:
        topic: The original podcast topic.
        script: The generated script dict (with 'title' and 'segments').
        provider: LLM provider ('openai', 'ollama', or 'openrouter').
        **kwargs: Additional args passed through (api_key, model, etc.)

    Returns:
        dict with a 'follow_up_questions' list.
    """
    prompt = _get_follow_up_prompt(topic, script)
    return _call_provider(
        "You generate follow-up questions in JSON format only.",
        prompt,
        provider=provider,
        **kwargs,
    )


def generate_script_summary(
    script: dict,
    provider: Optional[str] = None,
    **kwargs,
) -> dict:
    """Generate a summary of a podcast script.

    Args:
        script: The generated script dict (with 'title' and 'segments').
        provider: LLM provider ('openai', 'ollama', or 'openrouter').
        **kwargs: Additional args passed through (api_key, model, etc.)

    Returns:
        dict with 'summary', 'key_points', and 'key_takeaway'.
    """
    prompt = _get_summary_prompt(script)
    return _call_provider(
        "You generate summaries in JSON format only.",
        prompt,
        provider=provider,
        temperature=0.5,
        **kwargs,
    )
