"""
extraction.py — Phase 3: Per-scene extraction
Narraverse Encoder v2

Accepts: one Scene dict + global context (from segmentation.py)
Outputs: one SceneResult dict + mutated global context

Per scene, two LLM calls are made in this order:
  Call 1 — Style extraction (hard metrics via regex + soft metrics via LLM)
  Call 2 — Entities + Events (merged into one call for shared context benefit)

After both calls, update_context() mutates the global context in place
so the next scene's calls have access to everything learned so far.

SceneResult schema (contract with aggregation.py):
{
    "scene_id":           str,
    "chapter_index":      int,
    "scene_index":        int,
    "global_scene_index": int,
    "chapter_title":      str,
    "scene_summary":      str,
    "tension_level":      int,   — always int, never string
    "location":           str,
    "scene_type":         str,   — ACTION | DIALOGUE | INTROSPECTION | etc.
    "is_mutable":         bool,  — True if no structural pillar events in scene
    "style": {
        "avg_sentence_length":  float,
        "adverb_density":       float,   — percentage as float e.g. 3.2
        "dialogue_ratio":       float,   — percentage of text that is dialogue
        "total_words":          int,
        "pacing":               str,
        "pov":                  str,
        "tone_keywords":        list[str],
        "sensory_focus":        str,
    },
    "characters": [
        {
            "char_id":         str,
            "names_used":      list[str],
            "role_in_scene":   str,
            "emotional_state": str,
            "persistent_traits":  dict,   — only traits explicitly in text
            "evolving_traits":    dict,   — clothing, injuries, etc.
            "knowledge_gained":   list[str],
            "quote":           str | None
        }
    ],
    "character_states": [
        {
            "char_id":         str,
            "scene_id":        str,
            "chapter_index":   int,
            "scene_index":     int,
            "emotional_state": str,
            "goal":            str,
            "knowledge":       list[str],
            "evolving_traits": dict
        }
    ],
    "relationships": [
        {
            "from":    str,
            "to":      str,
            "type":    str,
            "dynamic": str
        }
    ],
    "events": [
        {
            "event_id":             str,
            "description":          str,
            "event_type":           str,
            "criticality_score":    int,   — 1-10
            "divergence_potential": int,   — 1-10
            "is_structural_pillar": bool,  — auto-set if criticality >= 9
            "is_mutable":           bool,  — inverse of is_structural_pillar
            "unresolved_consequence": str | None,
            "causes":    list[{"event_id": str, "relationship": str}],
            "caused_by": list[{"event_id": str, "relationship": str}],
            "characters_involved": list[{"char_id": str, "role": str}]
        }
    ]
}

Global context schema (lives in encoder.py, passed in and mutated here):
{
    "known_characters": {
        "char_jude": {
            "primary_name":     str,
            "all_names":        list[str],
            "persistent_traits": dict,
            "state_history":    list[StateSnapshot],
            "first_appearance": str   — scene_id
        }
    },
    "known_relationships": [
        {
            "from": str, "to": str, "type": str,
            "dynamic": str,
            "first_seen": str,    — scene_id
            "last_updated": str,  — scene_id
            "evolution": [{"scene_id": str, "dynamic": str}]
        }
    ],
    "known_locations": list[str],
    "known_event_ids": list[str],
    "scene_summaries": [{"scene_id": str, "summary": str}]
}
"""

import re
import json
import logging
from mistralai import Mistral

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [extraction] %(levelname)s — %(message)s"
)
log = logging.getLogger("extraction")

# ── Constants ─────────────────────────────────────────────────────────────────

# Model for all LLM extraction calls
EXTRACTION_MODEL = "mistral-large-latest"

# Maximum characters of scene text sent to the LLM.
# Covers ~900 words. Increase if your scenes are consistently longer.
SCENE_TEXT_LIMIT = 6000

# Maximum number of recent scene summaries injected into the context string.
# Keeping this bounded prevents the context string from growing unbounded.
MAX_CONTEXT_SUMMARIES = 10

# Criticality threshold above which an event is auto-flagged as a structural pillar
PILLAR_THRESHOLD = 9

# Scene types the LLM must choose from
SCENE_TYPES = [
    "ACTION", "DIALOGUE", "INTROSPECTION", "DESCRIPTION",
    "EXPOSITION", "TRANSITION", "ROMANCE", "CONFLICT",
    "REVELATION", "WORLD_BUILDING"
]

# Event types the LLM must choose from
EVENT_TYPES = [
    "ACTION", "DISCOVERY", "DECISION", "REVELATION",
    "CONFLICT", "TRANSFORMATION"
]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def extract_scene(scene: dict, context: dict, client: Mistral) -> dict:
    """
    Main entry point for extraction.
    Runs both extraction calls on one scene and updates the global context.

    Call order is strict:
      1. extract_style()        — no context needed, pure text analysis
      2. extract_entities_and_events() — needs context, returns chars + events
      3. update_context()       — must be LAST, mutates context for next scene

    Args:
        scene:    Scene dict from segmentation.assign_scene_meta()
        context:  Global context dict — READ by calls 1 and 2, WRITTEN by call 3
        client:   Mistral client instance

    Returns:
        dict — SceneResult matching the schema above
    """
    scene_id    = scene["scene_id"]
    scene_text  = scene["text"]
    chapter_idx = scene["chapter_index"]
    scene_idx   = scene["scene_index"]

    log.info(
        f"Extracting {scene_id} — "
        f"{scene['word_count']} words, "
        f"context: {len(context['known_characters'])} chars, "
        f"{len(context['known_relationships'])} rels, "
        f"{len(context['scene_summaries'])} summaries"
    )

    # ── Call 1: Style extraction ──────────────────────────────────────────────
    style = extract_style(scene_text, client)

    # ── Call 2: Entities + events (merged) ───────────────────────────────────
    entities_and_events = extract_entities_and_events(
        text=scene_text,
        scene_meta=scene,
        context=context,
        client=client
    )

    # ── Assemble SceneResult ──────────────────────────────────────────────────
    characters  = entities_and_events.get("entities", {}).get("characters", [])
    events      = entities_and_events.get("events", [])
    rels        = entities_and_events.get("entities", {}).get("relationships", [])
    entity_data = entities_and_events.get("entities", {})

    # Auto-set is_structural_pillar and is_mutable on every event
    events = _finalize_events(events, scene_id)

    # Build character_states — one snapshot per character in this scene
    character_states = _build_character_states(characters, scene)

    # Scene is mutable if it contains no structural pillar events
    scene_is_mutable = not any(e.get("is_structural_pillar", False) for e in events)

    scene_result = {
        "scene_id":           scene_id,
        "chapter_index":      chapter_idx,
        "scene_index":        scene_idx,
        "global_scene_index": scene["global_scene_index"],
        "chapter_title":      scene["chapter_title"],
        "scene_summary":      entity_data.get("scene_summary", ""),
        "tension_level":      _safe_int(entity_data.get("tension_level"), default=5),
        "location":           entity_data.get("location", "Unknown"),
        "scene_type":         style.get("scene_type", "UNKNOWN"),
        "is_mutable":         scene_is_mutable,
        "style":              style,
        "characters":         characters,
        "character_states":   character_states,
        "relationships":      rels,
        "events":             events,
    }

    # ── Call 3: Update context ────────────────────────────────────────────────
    # Must happen AFTER the result is assembled so we don't contaminate
    # this scene's extraction with its own output.
    update_context(context, scene_result)

    return scene_result


# ══════════════════════════════════════════════════════════════════════════════
# CALL 1 — STYLE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_style(text: str, client: Mistral) -> dict:
    """
    Extracts style metrics for one scene.

    Hard metrics: computed via regex — no LLM, no cost, always accurate.
    Soft metrics: one LLM call for subjective qualities (pacing, tone, etc.)

    On soft metric failure: logs a warning and fills in safe defaults.
    Never crashes the pipeline.

    Returns:
        dict — style object matching the style schema in SceneResult
    """
    # ── Hard metrics (regex) ──────────────────────────────────────────────────
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words     = re.findall(r'\b\w+\b', text.lower())

    avg_sentence_length = round(len(words) / len(sentences), 2) if sentences else 0

    adverbs     = [w for w in words if w.endswith("ly")]
    adverb_density = round((len(adverbs) / len(words)) * 100, 2) if words else 0

    # Dialogue ratio: percentage of text inside quotation marks
    dialogue_chars = sum(len(m) for m in re.findall(r'"[^"]*"', text))
    dialogue_ratio = round((dialogue_chars / len(text)) * 100, 2) if text else 0

    hard_metrics = {
        "avg_sentence_length": avg_sentence_length,
        "adverb_density":      adverb_density,
        "dialogue_ratio":      dialogue_ratio,
        "total_words":         len(words),
    }

    # ── Soft metrics (LLM) ────────────────────────────────────────────────────
    soft_defaults = {
        "pacing":         "Medium",
        "pov":            "Unknown",
        "tone_keywords":  [],
        "sensory_focus":  "Unknown",
        "scene_type":     "UNKNOWN",
    }

    prompt = f"""You are a literary analyst. Analyze the writing style of this scene.

Base your analysis ONLY on what is present in the provided text.

Return ONLY a valid JSON object with these exact keys:

{{
    "pacing": <EXACTLY ONE of: "Fast" | "Medium" | "Slow" | "Variable">,
    "pov": <EXACTLY ONE of: "1st Person" | "2nd Person" | "3rd Person Limited" | "3rd Person Omniscient" | "Multiple POV" | "Stream of Consciousness">,
    "tone_keywords": <list of 3-5 single-word tone descriptors e.g. ["ominous", "tense"]>,
    "sensory_focus": <EXACTLY ONE of: "Visual" | "Auditory" | "Tactile" | "Olfactory" | "Kinesthetic" | "Internal Monologue" | "Mixed">,
    "scene_type": <EXACTLY ONE of: {SCENE_TYPES}>
}}

No explanations. No markdown. Only the JSON object.

Scene text:
\"\"\"{text[:SCENE_TEXT_LIMIT]}\"\"\"
"""

    try:
        response = client.chat.complete(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1  # style metrics need to be consistent, not creative
        )
        raw = response.choices[0].message.content.strip()
        soft_metrics = _safe_parse_json(raw)

        # Validate scene_type is from allowed list
        if soft_metrics.get("scene_type") not in SCENE_TYPES:
            soft_metrics["scene_type"] = "UNKNOWN"

        # Merge: hard metrics + soft metrics
        return {**hard_metrics, **soft_metrics}

    except Exception as e:
        log.warning(f"Style LLM call failed: {e} — using defaults")
        return {**hard_metrics, **soft_defaults}


# ══════════════════════════════════════════════════════════════════════════════
# CALL 2 — ENTITIES + EVENTS (MERGED)
# ══════════════════════════════════════════════════════════════════════════════

def extract_entities_and_events(text: str, scene_meta: dict,
                                 context: dict, client: Mistral) -> dict:
    """
    Single LLM call that extracts both entities and events from one scene.

    Why merged: events reference characters, and characters' motivations
    shape which events are meaningful. Giving the LLM both tasks in one
    prompt with a structured two-section schema lets it use entity context
    while writing event descriptions — producing more accurate causal links
    than two separate calls would.

    The prompt explicitly tells the LLM to do Task 1 (entities) before
    Task 2 (events), so they remain structurally distinct.

    Args:
        text:       Raw scene text
        scene_meta: Scene dict from segmentation
        context:    Global context — serialized into the prompt
        client:     Mistral client

    Returns:
        dict with keys:
            "entities": { location, scene_summary, tension_level,
                          characters[], relationships[] }
            "events":   [ event objects ]
    """
    context_str = _build_context_string(context)
    scene_id    = scene_meta["scene_id"]

    prompt = f"""You are a narrative analyst extracting structured data from a scene.

=== CUMULATIVE STORY CONTEXT (everything known before this scene) ===
{context_str}

=== CURRENT SCENE ===
Scene ID: {scene_id}
Chapter: "{scene_meta['chapter_title']}"
Scene {scene_meta['scene_index']} of this chapter

\"\"\"{text[:SCENE_TEXT_LIMIT]}\"\"\"

=== YOUR TASK ===
Complete TWO tasks in order. Return as one JSON object with keys "entities" and "events".

---
TASK 1 — ENTITIES
Extract who is in this scene, where it takes place, and what the emotional stakes are.

Critical rules:
- If a character appears in KNOWN CHARACTERS above, use their EXACT existing char_id.
- Only create a new char_id (format: char_<firstname_lowercase>) for characters
  who are genuinely new — not previously listed.
- Extract ONLY traits explicitly stated in this scene's text.
- Separate persistent traits (species, hair color, eye color — things that never change)
  from evolving traits (clothing, injuries, emotional state — things that can change).
- Only include relationships that are NEW or have a CHANGED dynamic in this scene.
  Do not repeat stable relationships already listed in KNOWN RELATIONSHIPS.
- [Illustration: ...] markers in the text are visual elements — do not treat them
  as character dialogue or narrative events.

"entities": {{
    "location": "Exact setting description from the text",
    "scene_summary": "3-4 sentence summary of ONLY what explicitly happens in this scene",
    "tension_level": <integer 1-10 based on conflict and stakes in this scene>,
    "characters": [
        {{
            "char_id": "char_<name>",
            "names_used": ["Name1", "Nickname"],
            "role_in_scene": "What this character does in this scene",
            "emotional_state": "How they feel, based only on explicit text",
            "persistent_traits": {{"hair": "brown", "species": "human"}},
            "evolving_traits": {{"clothing": "torn dress", "injury": "cut on arm"}},
            "knowledge_gained": ["what this character learned in this scene"],
            "quote": "One short characteristic line of dialogue, or null"
        }}
    ],
    "relationships": [
        {{
            "from": "char_id1",
            "to": "char_id2",
            "type": "ALLY_WITH | RIVAL_OF | ENEMY_OF | LOVER_OF | SIBLING_OF | MENTOR_TO | SERVANT_OF | CAREGIVER_OF | OTHER",
            "dynamic": "One sentence describing their interaction in this scene (15-25 words)"
        }}
    ]
}}

---
TASK 2 — EVENTS
Extract the discrete events that occur in this scene — actions or occurrences
that change the state of the story.

Critical rules:
- Only extract DISCRETE, SPECIFIC events — things that actually happen, not states.
- event_id format: evt_<short_description_underscored> e.g. evt_jude_kills_dagdan
- For caused_by and causes: ONLY reference event_ids from KNOWN EVENTS listed above,
  or other events you are creating in this same scene.
- Do not invent causal links you cannot verify from the text.
- is_structural_pillar: set to true only if this event is completely irreversible
  and essential to the entire story's direction.
- unresolved_consequence: if this event sets something up that isn't resolved
  in this scene, describe it. Otherwise null.

"events": [
    {{
        "event_id": "evt_<unique_identifier>",
        "description": "Concise description of what happened (15-25 words)",
        "event_type": <EXACTLY ONE of: {EVENT_TYPES}>,
        "criticality_score": <integer 1-10: how much does this change the overall story>,
        "divergence_potential": <integer 1-10: how different would the story be if this hadn't happened>,
        "is_structural_pillar": <true | false>,
        "unresolved_consequence": "What this sets up for later, or null",
        "causes": [
            {{"event_id": "evt_id_of_consequence", "relationship": "TRIGGERS | ENABLES | REQUIRES"}}
        ],
        "caused_by": [
            {{"event_id": "evt_id_of_prerequisite", "relationship": "TRIGGERS | ENABLES | REQUIRES"}}
        ],
        "characters_involved": [
            {{"char_id": "char_id", "role": "trigger | affected | witness"}}
        ]
    }}
]

Return ONLY the JSON object. No markdown. No explanation.
"""

    empty_result = {"entities": {}, "events": []}

    try:
        response = client.chat.complete(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2  # low temp for structured extraction accuracy
        )
        raw = response.choices[0].message.content.strip()
        result = _safe_parse_json(raw)

        if not result:
            log.warning(f"{scene_id}: entity+event extraction returned empty JSON")
            return empty_result

        # Ensure expected top-level keys exist
        if "entities" not in result:
            result["entities"] = {}
        if "events" not in result:
            result["events"] = []

        return result

    except Exception as e:
        log.error(f"{scene_id}: entity+event extraction failed: {e}")
        return empty_result


# ══════════════════════════════════════════════════════════════════════════════
# CALL 3 — CONTEXT UPDATE (mutates in place)
# ══════════════════════════════════════════════════════════════════════════════

def update_context(context: dict, scene_result: dict) -> None:
    """
    Updates the global context with everything learned from this scene.
    Called AFTER extract_style and extract_entities_and_events.
    Mutates the context dict in place — returns nothing.

    Accumulate-not-overwrite: existing data is extended, never replaced.
    New names, new traits, new state snapshots are ADDED to existing entries.

    Args:
        context:      Global context dict (mutated in place)
        scene_result: The fully assembled SceneResult for this scene
    """
    scene_id    = scene_result["scene_id"]
    chapter_idx = scene_result["chapter_index"]
    scene_idx   = scene_result["scene_index"]

    # ── Update characters ─────────────────────────────────────────────────────
    for char in scene_result.get("characters", []):
        char_id = char.get("char_id")
        if not char_id:
            continue

        names_used       = char.get("names_used", [])
        persistent       = char.get("persistent_traits", {})
        evolving         = char.get("evolving_traits", {})
        emotional_state  = char.get("emotional_state", "")
        knowledge_gained = char.get("knowledge_gained", [])

        if char_id not in context["known_characters"]:
            # New character — initialize entry
            primary_name = names_used[0] if names_used else char_id
            context["known_characters"][char_id] = {
                "primary_name":      primary_name,
                "all_names":         list(names_used),
                "persistent_traits": dict(persistent),
                "state_history":     [],
                "first_appearance":  scene_id
            }
            log.info(f"  + New character: {char_id} ({primary_name})")

        existing = context["known_characters"][char_id]

        # Add new names without duplicating
        for name in names_used:
            if name and name not in existing["all_names"]:
                existing["all_names"].append(name)

        # Add new persistent traits without overwriting confirmed ones
        for key, value in persistent.items():
            if key not in existing["persistent_traits"]:
                existing["persistent_traits"][key] = value

        # Always append a state snapshot — even if emotional_state is empty
        # The decoder needs complete state history to query "state at scene X"
        state_snapshot = {
            "char_id":         char_id,
            "scene_id":        scene_id,
            "chapter_index":   chapter_idx,
            "scene_index":     scene_idx,
            "emotional_state": emotional_state,
            "goal":            char.get("role_in_scene", ""),
            "knowledge":       knowledge_gained,
            "evolving_traits": evolving
        }
        existing["state_history"].append(state_snapshot)

    # ── Update relationships ──────────────────────────────────────────────────
    for rel in scene_result.get("relationships", []):
        char_from = rel.get("from")
        char_to   = rel.get("to")
        rel_type  = rel.get("type")
        dynamic   = rel.get("dynamic", "")

        if not char_from or not char_to or not rel_type:
            continue

        # Normalize direction: A→B and B→A treated as same relationship
        pair_key = tuple(sorted([char_from, char_to]))

        # Find existing relationship of the same type between this pair
        existing_rel = None
        for r in context["known_relationships"]:
            existing_pair = tuple(sorted([r["from"], r["to"]]))
            if existing_pair == pair_key and r["type"] == rel_type:
                existing_rel = r
                break

        if existing_rel:
            # Only update if the dynamic has actually changed
            last_dynamic = (
                existing_rel["evolution"][-1]["dynamic"]
                if existing_rel.get("evolution")
                else existing_rel.get("dynamic", "")
            )
            if dynamic and dynamic != last_dynamic:
                existing_rel["evolution"].append({
                    "scene_id": scene_id,
                    "dynamic":  dynamic
                })
                existing_rel["dynamic"]      = dynamic
                existing_rel["last_updated"] = scene_id
                log.info(f"  ~ Relationship evolved: {char_from} → {char_to} ({rel_type})")
        else:
            # New relationship
            context["known_relationships"].append({
                "from":         char_from,
                "to":           char_to,
                "type":         rel_type,
                "dynamic":      dynamic,
                "first_seen":   scene_id,
                "last_updated": scene_id,
                "evolution":    [{"scene_id": scene_id, "dynamic": dynamic}]
            })
            log.info(f"  + New relationship: {char_from} → {char_to} ({rel_type})")

    # ── Update locations ──────────────────────────────────────────────────────
    location = scene_result.get("location", "")
    if location and location != "Unknown" and location not in context["known_locations"]:
        context["known_locations"].append(location)

    # ── Update known event IDs ────────────────────────────────────────────────
    # extract_events() in future scenes can reference these IDs in caused_by
    for event in scene_result.get("events", []):
        event_id = event.get("event_id")
        if event_id and event_id not in context["known_event_ids"]:
            context["known_event_ids"].append(event_id)

    # ── Update scene summaries ────────────────────────────────────────────────
    summary = scene_result.get("scene_summary", "")
    if summary:
        context["scene_summaries"].append({
            "scene_id": scene_id,
            "summary":  summary
        })


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_context_string(context: dict) -> str:
    """
    Serializes the global context dict into a readable string for LLM injection.

    Injects:
    - All known characters (with names and persistent traits)
    - Last MAX_CONTEXT_SUMMARIES scene summaries
    - All known relationships (current dynamic only, not full evolution)
    - All known locations
    - All known event IDs (so the LLM can form causal links)

    Keeps the string bounded — we don't inject full state histories or
    evolution arrays, which would make the prompt unboundedly large.
    """
    parts = []

    # Characters
    if context["known_characters"]:
        lines = ["KNOWN CHARACTERS:"]
        for char_id, char in context["known_characters"].items():
            traits_str = ", ".join(
                f"{k}: {v}" for k, v in char.get("persistent_traits", {}).items()
            )
            names_str = " / ".join(char.get("all_names", [char["primary_name"]]))
            # Most recent emotional state for continuity
            last_state = ""
            history = char.get("state_history", [])
            if history:
                last = history[-1]
                last_state = f", last state: {last.get('emotional_state', '')}"
            lines.append(
                f"  - {char_id}: {char['primary_name']} "
                f"(also known as: {names_str}) | {traits_str}{last_state}"
            )
        parts.append("\n".join(lines))

    # Recent scene summaries
    if context["scene_summaries"]:
        recent = context["scene_summaries"][-MAX_CONTEXT_SUMMARIES:]
        lines = ["RECENT SCENE SUMMARIES (most recent last):"]
        for s in recent:
            lines.append(f"  [{s['scene_id']}] {s['summary']}")
        parts.append("\n".join(lines))

    # Known relationships (current state only)
    if context["known_relationships"]:
        lines = ["KNOWN RELATIONSHIPS:"]
        for rel in context["known_relationships"]:
            lines.append(
                f"  - {rel['from']} → {rel['to']}: {rel['type']} | {rel['dynamic']}"
            )
        parts.append("\n".join(lines))

    # Known locations
    if context["known_locations"]:
        parts.append(
            "KNOWN LOCATIONS:\n  " +
            "\n  ".join(context["known_locations"][:20])  # cap at 20
        )

    # Known event IDs (for causal linking in Task 2)
    if context["known_event_ids"]:
        # Only show the last 30 event IDs — earlier ones are unlikely to be
        # directly caused by events in the current scene
        recent_events = context["known_event_ids"][-30:]
        parts.append(
            "KNOWN EVENTS (for causal linking):\n  " +
            "\n  ".join(recent_events)
        )

    return "\n\n".join(parts) if parts else "No prior context — this is the first scene."


def _finalize_events(events: list[dict], scene_id: str) -> list[dict]:
    """
    Post-processes the LLM's event list:
    - Ensures criticality_score and divergence_potential are ints
    - Auto-sets is_structural_pillar if criticality_score >= PILLAR_THRESHOLD
    - Sets is_mutable as the inverse of is_structural_pillar
    - Ensures all required fields exist with safe defaults
    """
    finalized = []
    for event in events:
        if not isinstance(event, dict):
            continue

        event_id = event.get("event_id", "")
        if not event_id:
            continue

        criticality   = _safe_int(event.get("criticality_score"),   default=5)
        divergence    = _safe_int(event.get("divergence_potential"), default=5)
        is_pillar     = event.get("is_structural_pillar", False)

        # Auto-upgrade to structural pillar if criticality is high enough
        if criticality >= PILLAR_THRESHOLD:
            is_pillar = True

        finalized.append({
            "event_id":               event_id,
            "description":            event.get("description", ""),
            "event_type":             event.get("event_type", "ACTION"),
            "scene_id":               scene_id,
            "criticality_score":      criticality,
            "divergence_potential":   divergence,
            "is_structural_pillar":   is_pillar,
            "is_mutable":             not is_pillar,
            "unresolved_consequence": event.get("unresolved_consequence"),
            "causes":                 event.get("causes", []),
            "caused_by":              event.get("caused_by", []),
            "characters_involved":    event.get("characters_involved", []),
        })

    return finalized


def _build_character_states(characters: list[dict], scene: dict) -> list[dict]:
    """
    Builds CharacterState snapshot objects from the characters extracted in this scene.
    One snapshot per character per scene — these become CharacterState nodes in Neo4j.

    These are what the decoder queries when it needs to know
    "what was character X's state at scene Y?"
    """
    states = []
    for char in characters:
        char_id = char.get("char_id")
        if not char_id:
            continue

        states.append({
            "char_id":         char_id,
            "scene_id":        scene["scene_id"],
            "chapter_index":   scene["chapter_index"],
            "scene_index":     scene["scene_index"],
            "emotional_state": char.get("emotional_state", ""),
            "goal":            char.get("role_in_scene", ""),
            "knowledge":       char.get("knowledge_gained", []),
            "evolving_traits": char.get("evolving_traits", {})
        })

    return states


def _safe_int(value, default: int = 5) -> int:
    """
    Safely casts a value to int.
    Returns default if the value is None, empty, or non-numeric.
    The LLM sometimes returns tension_level as "7" (string) instead of 7.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_parse_json(content: str) -> dict:
    """
    Parses JSON from LLM output.
    Strips markdown fences if present.
    Returns empty dict on failure — callers handle the empty case.
    """
    cleaned = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse failed: {e} — snippet: {cleaned[:200]}")
        return {}