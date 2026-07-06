"""Script Generator - Uses an LLM to generate podcast scripts.

Supports OpenAI, Ollama, and OpenRouter via a unified dispatch.
Each operation type (script gen, follow-up, summary) has one entry
point that delegates to a shared provider client.
"""

import json
from typing import Any, Optional

from . import config


# ---------------------------------------------------------------------------
# Shared LLM client wrappers
# ---------------------------------------------------------------------------


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.8,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """Call OpenAI-compatible API and return parsed JSON response."""
    from openai import OpenAI

    key = api_key or config.OPENAI_API_KEY
    if not key and not base_url:
        raise ValueError(
            "OpenAI API key not set. Set PODCAST_OPENAI_API_KEY env var "
            "or pass api_key."
        )

    model_name = model or config.OPENAI_MODEL
    client_kwargs: dict[str, Any] = {"api_key": key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    extra_kwargs: dict[str, Any] = {"temperature": temperature}

    # OpenAI native responses use response_format; others use extra_headers
    if base_url:
        extra_kwargs["extra_headers"] = {
            "HTTP-Referer": "https://github.com/straub-ah",
            "X-Title": "BPM Podcast Generator",
        }
    else:
        extra_kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **extra_kwargs,
    )

    text = response.choices[0].message.content
    # Some routers (OpenRouter) wrap JSON in markdown fences
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


def _call_ollama(
    system_prompt: str,
    user_prompt: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Call Ollama API and return parsed JSON response."""
    import requests

    url = base_url or config.OLLAMA_BASE_URL
    model_name = model or config.OLLAMA_MODEL
    prompt = f"{system_prompt}\n\n{user_prompt}"

    resp = requests.post(
        f"{url}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=120,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["response"])


def _call_provider(
    system_prompt: str,
    user_prompt: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.8,
) -> dict[str, Any]:
    """Unified dispatch: route to the right provider and return parsed JSON."""
    provider = provider or config.LLM_PROVIDER

    if provider == "openai":
        return _call_openai(system_prompt, user_prompt, api_key, model, temperature)
    elif provider == "ollama":
        return _call_ollama(system_prompt, user_prompt, model=model)
    elif provider == "openrouter":
        return _call_openai(
            system_prompt,
            user_prompt,
            api_key=api_key,
            model=model or config.OPENROUTER_MODEL,
            temperature=temperature,
            base_url=config.OPENROUTER_BASE_URL,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


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
- Avoid markdown, lists, or special formatting
- Include brief pauses marked with [pause]
- The entire script should take about {duration_minutes} minutes to read at natural pace
- Do NOT use sound effect notations like [music], [applause] — only [pause] is allowed
- NO special characters or emojis"""


def _get_outline_prompt(topic: str, bpm: int, duration_minutes: int = 5) -> str:
    """Build the outline prompt for the first LLM step."""
    style_context = _get_script_style_context(bpm, duration_minutes)

    return f"""You are planning a {duration_minutes}-minute solo educational podcast.

TOPIC: {topic}
{style_context}

Create a section outline before any script text is written. The outline should include a hook/intro, several content sections, and an outro. Each section topic should be specific enough that it can be written independently while still fitting the full episode.

Return ONLY valid JSON with this exact structure:
{{
    "title": "Episode title",
    "sections": [
        {{
            "segment_type": "intro",
            "topic": "Section topic",
            "approx_duration_seconds": 30
        }},
        {{
            "segment_type": "content",
            "topic": "Section topic",
            "approx_duration_seconds": 45
        }}
    ]
}}

The total of approx_duration_seconds should be about {duration_minutes * 60}.
Each content section should be 30-60 seconds of natural speech.
"""


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

Create a research brief that delegates one focused research lane to each outline section. Include clarifying follow-up questions the host could ask the listener if the topic needs sharper intent before script writing.

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

Find the ideas, examples, definitions, tensions, and open questions that would make this section specific and useful. If the section needs more user intent, include follow_up_questions that can be shown to the user before final generation.

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

Write ONLY the current section. Use the previous section text for continuity, but do not repeat it. Use the research context for specificity, examples, nuance, and intriguing framing. Lead naturally toward the next topic when one exists.

Return ONLY valid JSON with this exact structure:
{{
    "segment_type": "{segment_type}",
    "text": "The current section script text...",
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
    **kwargs,
) -> dict:
    """Generate a podcast section outline without writing full script text."""
    outline_prompt = _get_outline_prompt(topic, bpm, duration_minutes)
    return _call_provider(
        "You are a podcast outline planner. You output raw JSON only.",
        outline_prompt,
        provider=provider,
        **kwargs,
    )


def generate_research_brief(
    topic: str,
    outline: dict,
    provider: Optional[str] = None,
    **kwargs,
) -> dict:
    """Run the lead research-agent step for an approved outline."""
    return _call_provider(
        "You are the lead research agent for a podcast production team. You output raw JSON only.",
        _get_research_brief_prompt(topic, outline),
        provider=provider,
        temperature=0.4,
        **kwargs,
    )


def generate_subtopic_research(
    topic: str,
    outline: dict,
    section: dict,
    research_brief: dict,
    provider: Optional[str] = None,
    **kwargs,
) -> dict:
    """Run one subtopic research-agent step for an outline section."""
    return _call_provider(
        "You are a subtopic research agent for a podcast production team. You output raw JSON only.",
        _get_subtopic_research_prompt(topic, outline, section, research_brief),
        provider=provider,
        temperature=0.4,
        **kwargs,
    )


def generate_script(
    topic: str,
    bpm: int,
    duration_minutes: int = 5,
    provider: Optional[str] = None,
    outline: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Generate a podcast script using outline-first section generation."""
    if outline is None:
        outline = generate_script_outline(
            topic=topic,
            bpm=bpm,
            duration_minutes=duration_minutes,
            provider=provider,
            **kwargs,
        )

    sections = outline.get("sections", [])

    research_brief = generate_research_brief(
        topic=topic,
        outline=outline,
        provider=provider,
        **kwargs,
    )
    section_research_items = [
        generate_subtopic_research(
            topic=topic,
            outline=outline,
            section=section,
            research_brief=research_brief,
            provider=provider,
            **kwargs,
        )
        for section in sections
    ]

    generated_segments = []
    previous_section_text = ""

    for index, section in enumerate(sections):
        previous_section = sections[index - 1] if index > 0 else None
        next_section = sections[index + 1] if index + 1 < len(sections) else None
        section_prompt = _get_section_prompt(
            topic,
            bpm,
            outline,
            section,
            previous_section,
            next_section,
            previous_section_text,
            research_brief,
            section_research_items[index],
            duration_minutes,
        )
        generated_section = _call_provider(
            "You are a podcast script writer. You output raw JSON only.",
            section_prompt,
            provider=provider,
            **kwargs,
        )
        segment = generated_section.get("segment", generated_section)
        generated_segments.append(
            {
                "segment_type": segment.get(
                    "segment_type",
                    section.get("segment_type", "content"),
                ),
                "subtopic": section.get("topic") or section.get("title") or topic,
                "title": section.get("title") or section.get("topic"),
                "text": segment.get("text", ""),
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
