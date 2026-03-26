"""
aggregation.py — Phase 4: Aggregation
Narraverse Encoder v2

Accepts: all SceneResult dicts + final global context (from extraction.py)
Outputs: full_book_report.json — the contract file for Neo4j ingestion

Aggregation runs AFTER all scenes are processed.
It never calls the LLM in a loop — only a small number of targeted
synthesis calls are made on already-compressed data.

Report schema (top-level keys):
{
    "metadata":        { book-level facts + flat style_fingerprint },
    "chapters":        [ chapter summaries with scene index ],
    "characters":      { char_id: full character profile with state_history },
    "relationships":   [ relationship objects with evolution ],
    "causal_graph":    { events[], divergence_points[], sequel_seeds[] },
    "tension_profile": { average, peaks, valleys, sequel_hooks }
}
"""

import json
import logging
from pathlib import Path
from statistics import mean, median, stdev
from collections import Counter
from mistralai import Mistral

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [aggregation] %(levelname)s — %(message)s"
)
log = logging.getLogger("aggregation")

# ── Constants ─────────────────────────────────────────────────────────────────

AGGREGATION_MODEL = "mistral-large-latest"

# Minimum number of appearances for a character to get a full LLM profile.
# Characters seen only once are included but don't get an LLM synthesis call.
MIN_APPEARANCES_FOR_PROFILE = 2

# Criticality threshold for structural pillar auto-detection in the causal graph
PILLAR_THRESHOLD = 9


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_report(scene_results: list[dict], context: dict,
                 chapters: list[dict], book_meta: dict,
                 client: Mistral, output_path: str) -> dict:
    """
    Main entry point for aggregation.
    Assembles all scene results and context into the final report.

    Args:
        scene_results: list of SceneResult dicts from extraction.extract_scene()
        context:       final global context dict (fully populated)
        chapters:      list of Chapter dicts from ingestion (for titles/word counts)
        book_meta:     dict with book_id, title, author — from encoder.py config
        client:        Mistral client for synthesis LLM calls
        output_path:   where to write full_book_report.json

    Returns:
        dict — the full report (also written to output_path)
    """
    log.info("Starting aggregation...")

    # ── Step 4.1: Style fingerprint ───────────────────────────────────────────
    log.info("Building style fingerprint...")
    style_fingerprint = build_style_fingerprint(scene_results)

    # ── Step 4.2: Tension profile ─────────────────────────────────────────────
    log.info("Building tension profile...")
    tension_profile = build_tension_profile(scene_results, client)

    # ── Step 4.3: Chapter summaries ───────────────────────────────────────────
    log.info("Building chapter summaries...")
    chapter_summaries = build_chapter_summaries(scene_results, chapters, client)

    # ── Step 4.4: Character profiles ─────────────────────────────────────────
    log.info("Building character profiles...")
    character_profiles = build_character_profiles(context, scene_results, client)

    # ── Step 4.5: Causal graph connection pass ───────────────────────────────
    log.info("Connecting causal graph...")
    causal_graph = connect_causal_graph(scene_results, context, client)

    # ── Step 4.6: Assemble metadata ───────────────────────────────────────────
    total_words   = sum(s.get("style", {}).get("total_words", 0) for s in scene_results)
    total_scenes  = len(scene_results)
    avg_tension   = round(mean([s["tension_level"] for s in scene_results
                                if isinstance(s.get("tension_level"), int)]), 2)

    metadata = {
        "book_id":                book_meta.get("book_id", "unknown"),
        "title":                  book_meta.get("title", "Unknown"),
        "author":                 book_meta.get("author", "Unknown"),
        "total_chapters":         len(chapters),
        "total_scenes":           total_scenes,
        "total_words":            total_words,
        "avg_tension":            avg_tension,
        "total_unique_characters": len(character_profiles),
        "total_unique_locations": len(context.get("known_locations", [])),
        "total_relationships":    len(context.get("known_relationships", [])),
        "total_causal_events":    len(causal_graph.get("events", [])),
        "style_fingerprint":      style_fingerprint,
    }

    # ── Step 4.7: Assemble final report ───────────────────────────────────────
    report = {
        "metadata":        metadata,
        "chapters":        chapter_summaries,
        "characters":      character_profiles,
        "relationships":   context.get("known_relationships", []),
        "causal_graph":    causal_graph,
        "tension_profile": tension_profile,
    }

    # ── Step 4.8: Validate before writing ────────────────────────────────────
    log.info("Validating report...")
    issues = validate_report(report)
    if issues:
        log.warning(f"Validation found {len(issues)} issue(s):")
        for issue in issues:
            log.warning(f"  - {issue}")
    else:
        log.info("Validation passed — no issues found")

    # ── Step 4.9: Write to disk ───────────────────────────────────────────────
    write_report(report, output_path)

    return report


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4.1 — STYLE FINGERPRINT
# ══════════════════════════════════════════════════════════════════════════════

def build_style_fingerprint(scene_results: list[dict]) -> dict:
    """
    Averages all per-scene style metrics into a single flat dict.
    This is stored on the Book node in Neo4j and injected directly
    into the decoder system prompt.

    No LLM call — pure Python aggregation.

    Returns a FLAT dict — not nested — so the decoder can access
    style_fingerprint["avg_sentence_length"] directly.
    """
    all_asl        = []
    all_adverb     = []
    all_dialogue   = []
    all_pacing     = []
    all_pov        = []
    all_sensory    = []
    all_tone       = []
    all_scene_type = []

    for scene in scene_results:
        style = scene.get("style", {})
        if not style:
            continue

        _collect_float(all_asl,      style.get("avg_sentence_length"))
        _collect_float(all_adverb,   style.get("adverb_density"))
        _collect_float(all_dialogue, style.get("dialogue_ratio"))

        if style.get("pacing"):
            all_pacing.append(style["pacing"])
        if style.get("pov"):
            all_pov.append(style["pov"])
        if style.get("sensory_focus"):
            all_sensory.append(style["sensory_focus"])
        if style.get("scene_type"):
            all_scene_type.append(style["scene_type"])
        if isinstance(style.get("tone_keywords"), list):
            all_tone.extend(style["tone_keywords"])

    # Dominant values — most frequent wins
    dominant_pacing      = Counter(all_pacing).most_common(1)[0][0]  if all_pacing      else "Unknown"
    primary_pov          = Counter(all_pov).most_common(1)[0][0]     if all_pov         else "Unknown"
    primary_sensory      = Counter(all_sensory).most_common(1)[0][0] if all_sensory     else "Unknown"
    top_tone_keywords    = [t for t, _ in Counter(all_tone).most_common(5)]

    return {
        # Averaged metrics
        "avg_sentence_length":  round(mean(all_asl),      2) if all_asl      else 0,
        "adverb_density":       round(mean(all_adverb),   2) if all_adverb   else 0,
        "dialogue_ratio":       round(mean(all_dialogue), 2) if all_dialogue else 0,

        # Dominant categorical values
        "dominant_pace":        dominant_pacing,
        "primary_pov":          primary_pov,
        "primary_sensory_focus": primary_sensory,
        "tone_keywords":        top_tone_keywords,

        # Distribution data (useful for author bible)
        "pacing_distribution":      dict(Counter(all_pacing)),
        "scene_type_distribution":  dict(Counter(all_scene_type)),
        "pov_distribution":         dict(Counter(all_pov)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4.2 — TENSION PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def build_tension_profile(scene_results: list[dict], client: Mistral) -> dict:
    """
    Aggregates all scene tension scores into a book-level tension profile.

    One LLM call: given the highest and lowest tension scenes,
    asks the LLM to describe the overall tension arc and generate sequel hooks.

    Returns:
        dict with average, peaks, valleys, arc description, sequel hooks,
        and sequel_potential_rating
    """
    tension_data = []
    for scene in scene_results:
        level = scene.get("tension_level")
        if isinstance(level, int):
            tension_data.append({
                "scene_id": scene["scene_id"],
                "tension":  level,
                "summary":  scene.get("scene_summary", "")
            })

    if not tension_data:
        log.warning("No tension data found — skipping tension profile")
        return {}

    avg_tension = round(mean([t["tension"] for t in tension_data]), 2)

    # Top peaks and lowest valleys
    sorted_by_tension = sorted(tension_data, key=lambda x: x["tension"], reverse=True)
    peaks   = sorted_by_tension[:5]
    valleys = sorted_by_tension[-3:]

    # Build LLM context
    peaks_str   = "\n".join(
        f"  [{t['scene_id']}] tension={t['tension']}: {t['summary']}"
        for t in peaks
    )
    valleys_str = "\n".join(
        f"  [{t['scene_id']}] tension={t['tension']}: {t['summary']}"
        for t in valleys
    )

    prompt = f"""Analyze the tension arc of this narrative from the scene data provided.
Base your analysis ONLY on the data below.

HIGH TENSION SCENES (peaks):
{peaks_str}

LOW TENSION SCENES (valleys):
{valleys_str}

Return ONLY a valid JSON object:
{{
    "overall_tension_pattern": "How tension builds and releases across the story (40-50 words)",
    "climax_description": "The highest tension moment and what makes it climactic (30-40 words)",
    "resolution_style": "How the story resolves its central tension (20-30 words)",
    "sequel_hooks": [
        "Unresolved tension or open question that could drive a sequel",
        "Second unresolved thread",
        "Third unresolved thread"
    ],
    "sequel_potential_rating": <integer 1-10>
}}
"""

    try:
        response = client.chat.complete(
            model=AGGREGATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        arc_data = _safe_parse_json(response.choices[0].message.content)
    except Exception as e:
        log.error(f"Tension profile LLM call failed: {e}")
        arc_data = {}

    return {
        "average_tension":          avg_tension,
        "highest_tension_scenes":   [{"scene_id": t["scene_id"], "tension": t["tension"]} for t in peaks],
        "lowest_tension_scenes":    [{"scene_id": t["scene_id"], "tension": t["tension"]} for t in valleys],
        "overall_tension_pattern":  arc_data.get("overall_tension_pattern", ""),
        "climax_description":       arc_data.get("climax_description", ""),
        "resolution_style":         arc_data.get("resolution_style", ""),
        "sequel_hooks":             arc_data.get("sequel_hooks", []),
        "sequel_potential_rating":  arc_data.get("sequel_potential_rating", 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4.3 — CHAPTER SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════

def build_chapter_summaries(scene_results: list[dict],
                             chapters: list[dict],
                             client: Mistral) -> list[dict]:
    """
    Builds a summary object for each chapter.
    One LLM call per chapter — given that chapter's scene summaries,
    generates a 4-6 sentence chapter-level summary.

    Returns:
        list[dict] — one entry per chapter, ordered by chapter_index
    """
    # Group scenes by chapter
    from collections import defaultdict
    chapter_scenes = defaultdict(list)
    for scene in scene_results:
        chapter_scenes[scene["chapter_index"]].append(scene)

    chapter_summaries = []

    for chapter in sorted(chapters, key=lambda c: c["chapter_index"]):
        ch_idx   = chapter["chapter_index"]
        ch_title = chapter["title"]
        scenes   = chapter_scenes.get(ch_idx, [])

        if not scenes:
            log.warning(f"No scenes found for chapter {ch_idx} '{ch_title}'")
            continue

        log.info(f"  Summarizing chapter {ch_idx}: '{ch_title}' ({len(scenes)} scenes)")

        scene_summaries_text = "\n".join(
            f"Scene {s['scene_index']}: {s.get('scene_summary', '')}"
            for s in scenes
        )

        prompt = f"""Create a comprehensive summary for this chapter based ONLY on the scene summaries provided.

Chapter: "{ch_title}"

Scene summaries:
{scene_summaries_text}

Return ONLY a valid JSON object:
{{
    "chapter_summary": "A 4-6 sentence summary covering all major events and character developments in this chapter"
}}
"""
        try:
            response = client.chat.complete(
                model=AGGREGATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            result  = _safe_parse_json(response.choices[0].message.content)
            summary = result.get("chapter_summary", "Summary generation failed")
        except Exception as e:
            log.error(f"Chapter summary failed for '{ch_title}': {e}")
            summary = "Summary generation failed"

        # Chapter-level tension stats
        tensions = [s["tension_level"] for s in scenes
                    if isinstance(s.get("tension_level"), int)]

        chapter_summaries.append({
            "chapter_index": ch_idx,
            "title":         ch_title,
            "total_scenes":  len(scenes),
            "word_count":    chapter.get("word_count", 0),
            "avg_tension":   round(mean(tensions), 2) if tensions else 0,
            "summary":       summary,
            "scene_ids":     [s["scene_id"] for s in scenes],
        })

    log.info(f"Chapter summaries complete — {len(chapter_summaries)} chapters")
    return chapter_summaries


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4.4 — CHARACTER PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def build_character_profiles(context: dict, scene_results: list[dict],
                              client: Mistral) -> dict:
    """
    Builds a complete character profile for every character in the global context.

    For characters with MIN_APPEARANCES_FOR_PROFILE or more appearances:
    one LLM call generates overall_role, character_arc, personality_summary.

    For minor characters (fewer appearances):
    profile is assembled from data alone — no LLM call.

    Always includes:
    - persistent_traits (accumulated across all scenes)
    - state_history (all CharacterState snapshots)
    - final_emotional_state (from last scene appearance)
    - image_reference_prompt (built from persistent_traits for image gen)
    - all_names, first_appearance

    Returns:
        dict — {char_id: profile_dict}
    """
    profiles = {}

    for char_id, char_data in context["known_characters"].items():
        state_history    = char_data.get("state_history", [])
        persistent       = char_data.get("persistent_traits", {})
        all_names        = char_data.get("all_names", [char_data["primary_name"]])
        first_appearance = char_data.get("first_appearance", "")

        # Final emotional state = last state snapshot
        final_emotional_state = ""
        final_goal = ""
        if state_history:
            last = state_history[-1]
            final_emotional_state = last.get("emotional_state", "")
            final_goal            = last.get("goal", "")

        # Build image reference prompt from persistent traits
        image_reference_prompt = _build_image_prompt(
            char_data["primary_name"], persistent
        )

        # Find relationships involving this character
        char_relationships = [
            rel for rel in context.get("known_relationships", [])
            if rel["from"] == char_id or rel["to"] == char_id
        ]

        base_profile = {
            "character_id":          char_id,
            "primary_name":          char_data["primary_name"],
            "all_names":             all_names,
            "persistent_traits":     persistent,
            "final_emotional_state": final_emotional_state,
            "final_goal":            final_goal,
            "first_appearance":      first_appearance,
            "total_appearances":     len(state_history),
            "image_reference_prompt": image_reference_prompt,
            "state_history":         state_history,
            "relationships_summary": [
                {
                    "with":    rel["to"] if rel["from"] == char_id else rel["from"],
                    "type":    rel["type"],
                    "dynamic": rel["dynamic"]
                }
                for rel in char_relationships
            ]
        }

        # LLM synthesis for significant characters
        if len(state_history) >= MIN_APPEARANCES_FOR_PROFILE:
            log.info(f"  Generating profile for {char_id} ({len(state_history)} appearances)")

            all_emotional_states = [
                s.get("emotional_state", "") for s in state_history
                if s.get("emotional_state")
            ]
            all_roles = [
                s.get("goal", "") for s in state_history if s.get("goal")
            ]
            rel_text = "\n".join(
                f"  - {r['with']}: {r['type']} | {r['dynamic']}"
                for r in base_profile["relationships_summary"]
            ) or "No relationships recorded"

            prompt = f"""Create a character profile based ONLY on the data provided.
Do NOT invent information not present below.

Character: {char_data['primary_name']} ({char_id})
Also known as: {', '.join(all_names)}
Persistent traits: {persistent}
Total appearances: {len(state_history)}
Emotional states observed (in order): {all_emotional_states}
Roles observed: {list(set(all_roles))}
Final state: emotional={final_emotional_state}, goal={final_goal}

Relationships:
{rel_text}

Return ONLY a valid JSON object:
{{
    "overall_role": "Their role in the story (20-30 words)",
    "character_arc": "How they develop across the story (30-40 words)",
    "personality_summary": "Core personality traits and behavioral patterns (40-50 words)",
    "key_traits": ["trait1", "trait2", "trait3"]
}}
"""
            try:
                response = client.chat.complete(
                    model=AGGREGATION_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                synthesis = _safe_parse_json(response.choices[0].message.content)
                base_profile.update({
                    "overall_role":       synthesis.get("overall_role", ""),
                    "character_arc":      synthesis.get("character_arc", ""),
                    "personality_summary": synthesis.get("personality_summary", ""),
                    "key_traits":         synthesis.get("key_traits", []),
                })
            except Exception as e:
                log.error(f"Profile LLM call failed for {char_id}: {e}")
                base_profile.update({
                    "overall_role": "", "character_arc": "",
                    "personality_summary": "", "key_traits": []
                })
        else:
            base_profile.update({
                "overall_role": "", "character_arc": "",
                "personality_summary": "", "key_traits": []
            })

        profiles[char_id] = base_profile

    log.info(f"Character profiles complete — {len(profiles)} characters")
    return profiles


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4.5 — CAUSAL GRAPH CONNECTION PASS
# ══════════════════════════════════════════════════════════════════════════════

def connect_causal_graph(scene_results: list[dict],
                          context: dict,
                          client: Mistral) -> dict:
    """
    Lightweight post-processing pass on the causal graph.

    During per-scene extraction, events can only form causal links to events
    already in known_event_ids at the time of extraction. This means:
    - Event A in scene 5 cannot reference Event B in scene 8 as a consequence
      even if the book makes that link explicit in scene 5's text.

    This pass collects all events, then asks the LLM ONE time to:
    1. Identify cross-scene causal links that extraction couldn't see
    2. Flag divergence points (events where story could have gone differently)
    3. Identify sequel seeds (unresolved events that set up future stories)
    4. Validate and remove any dangling event ID references

    One LLM call total — not per scene. Works on summaries not raw text,
    so context window is manageable.

    Returns:
        dict with keys: events[], divergence_points[], sequel_seeds[], graph_metrics{}
    """
    # Collect all events from all scenes
    all_events = []
    for scene in scene_results:
        for event in scene.get("events", []):
            all_events.append(event)

    if not all_events:
        log.warning("No events found — causal graph will be empty")
        return {"events": [], "divergence_points": [], "sequel_seeds": [], "graph_metrics": {}}

    log.info(f"Causal graph connection pass on {len(all_events)} events...")

    # Build a compact event summary for the LLM
    event_ids = {e["event_id"] for e in all_events}
    event_summary = "\n".join(
        f"  [{e['event_id']}] scene={e.get('scene_id', '?')} "
        f"criticality={e.get('criticality_score', '?')} "
        f"pillar={e.get('is_structural_pillar', False)}: "
        f"{e.get('description', '')}"
        for e in all_events
    )

    # Recent scene summaries for narrative context
    scene_context = "\n".join(
        f"  [{s['scene_id']}]: {s.get('scene_summary', '')}"
        for s in scene_results[-20:]  # last 20 scenes for context
    )

    prompt = f"""You are a narrative causality analyst.

Below are all the events extracted from the book, in order.
Your task is to identify THREE things that per-scene extraction could not see:

1. CROSS-SCENE CAUSAL LINKS: events that cause other events in a DIFFERENT scene.
   Only add links that are clearly supported by the event descriptions.

2. DIVERGENCE POINTS: events where a different choice would have led to a
   significantly different story. These are the What-If candidates.

3. SEQUEL SEEDS: events with unresolved consequences that set up a future story.

ALL EVENTS (in story order):
{event_summary}

RECENT SCENE CONTEXT:
{scene_context}

Return ONLY a valid JSON object:
{{
    "cross_scene_links": [
        {{
            "from_event_id": "evt_id",
            "to_event_id": "evt_id",
            "relationship": "TRIGGERS | ENABLES | REQUIRES",
            "explanation": "Why this causal link exists (10-15 words)"
        }}
    ],
    "divergence_points": [
        {{
            "event_id": "evt_id",
            "decision_made": "What happened in the original story",
            "alternatives": ["Alternative outcome 1", "Alternative outcome 2"],
            "divergence_potential": <integer 1-10>,
            "alternate_timeline_hint": "What would have happened instead (20-30 words)"
        }}
    ],
    "sequel_seeds": [
        {{
            "event_id": "evt_id",
            "unresolved_consequence": "What this sets up for a sequel (20-30 words)",
            "sequel_potential": <integer 1-10>
        }}
    ]
}}

IMPORTANT:
- Only reference event_ids from the list above.
- Only create causal links you can clearly justify.
- Leave arrays empty if you cannot find clear examples.
"""

    try:
        response = client.chat.complete(
            model=AGGREGATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        graph_additions = _safe_parse_json(response.choices[0].message.content)
    except Exception as e:
        log.error(f"Causal graph connection pass failed: {e}")
        graph_additions = {}

    # Apply cross-scene links to the events list
    all_events = _apply_cross_scene_links(
        all_events, graph_additions.get("cross_scene_links", [])
    )

    # Validate — remove any dangling event ID references
    all_events = _validate_event_references(all_events, event_ids)

    # Calculate graph metrics
    graph_metrics = _calculate_graph_metrics(all_events)

    log.info(
        f"Causal graph complete: {len(all_events)} events, "
        f"{graph_metrics.get('total_causal_links', 0)} causal links, "
        f"{len(graph_additions.get('divergence_points', []))} divergence points"
    )

    return {
        "events":            all_events,
        "divergence_points": graph_additions.get("divergence_points", []),
        "sequel_seeds":      graph_additions.get("sequel_seeds", []),
        "graph_metrics":     graph_metrics,
    }


def _apply_cross_scene_links(events: list[dict],
                              cross_links: list[dict]) -> list[dict]:
    """
    Applies cross-scene causal links identified in the connection pass
    to the corresponding event objects.
    """
    event_map = {e["event_id"]: e for e in events}

    for link in cross_links:
        from_id = link.get("from_event_id")
        to_id   = link.get("to_event_id")
        rel     = link.get("relationship", "TRIGGERS")

        if from_id in event_map and to_id in event_map:
            # Add to causes of the source event
            causes = event_map[from_id].setdefault("causes", [])
            if not any(c["event_id"] == to_id for c in causes):
                causes.append({"event_id": to_id, "relationship": rel})

            # Add to caused_by of the target event
            caused_by = event_map[to_id].setdefault("caused_by", [])
            if not any(c["event_id"] == from_id for c in caused_by):
                caused_by.append({"event_id": from_id, "relationship": rel})

    return list(event_map.values())


def _validate_event_references(events: list[dict],
                                event_ids: set[str]) -> list[dict]:
    """
    Removes any causal link that references an event_id not in the known set.
    Prevents dangling references from breaking Neo4j ingestion.
    """
    for event in events:
        event["causes"] = [
            c for c in event.get("causes", [])
            if c.get("event_id") in event_ids
        ]
        event["caused_by"] = [
            c for c in event.get("caused_by", [])
            if c.get("event_id") in event_ids
        ]
    return events


def _calculate_graph_metrics(events: list[dict]) -> dict:
    """Calculates summary statistics about the causal graph structure."""
    if not events:
        return {}

    total_causes    = sum(len(e.get("causes", []))    for e in events)
    total_caused_by = sum(len(e.get("caused_by", [])) for e in events)
    pillars         = [e for e in events if e.get("is_structural_pillar")]
    mutable         = [e for e in events if e.get("is_mutable")]

    criticality_scores = [
        e["criticality_score"] for e in events
        if isinstance(e.get("criticality_score"), int)
    ]

    return {
        "total_events":              len(events),
        "total_causal_links":        total_causes + total_caused_by,
        "structural_pillars_count":  len(pillars),
        "mutable_events_count":      len(mutable),
        "avg_criticality_score":     round(mean(criticality_scores), 2) if criticality_scores else 0,
        "avg_connections_per_event": round((total_causes + total_caused_by) / len(events), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4.6 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_report(report: dict) -> list[str]:
    """
    Validates the assembled report before writing.

    Checks:
    - All char_ids referenced in events exist in characters dict
    - All event_ids in causal links exist in the events list
    - All scene_ids in chapter scene lists are consistent
    - All characters have at least a primary_name and character_id

    Returns:
        list[str] — list of issue descriptions (empty = no issues)
    """
    issues = []

    char_ids   = set(report.get("characters", {}).keys())
    event_ids  = {
        e["event_id"]
        for e in report.get("causal_graph", {}).get("events", [])
    }

    # Check character references in events
    for event in report.get("causal_graph", {}).get("events", []):
        for ci in event.get("characters_involved", []):
            cid = ci.get("char_id")
            if cid and cid not in char_ids:
                issues.append(
                    f"Event '{event['event_id']}' references unknown char_id '{cid}'"
                )

    # Check causal link references
    for event in report.get("causal_graph", {}).get("events", []):
        for link in event.get("causes", []) + event.get("caused_by", []):
            eid = link.get("event_id")
            if eid and eid not in event_ids:
                issues.append(
                    f"Event '{event['event_id']}' has dangling causal link to '{eid}'"
                )

    # Check character profiles completeness
    for char_id, profile in report.get("characters", {}).items():
        if not profile.get("primary_name"):
            issues.append(f"Character '{char_id}' is missing primary_name")
        if not profile.get("character_id"):
            issues.append(f"Character '{char_id}' is missing character_id field")

    # Check chapter scene_ids are consistent
    chapter_scene_ids = set()
    for chapter in report.get("chapters", []):
        for sid in chapter.get("scene_ids", []):
            if sid in chapter_scene_ids:
                issues.append(f"Duplicate scene_id found: '{sid}'")
            chapter_scene_ids.add(sid)

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4.7 — WRITE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(report: dict, output_path: str) -> None:
    """
    Writes the final report to disk as JSON.
    Creates parent directories if they don't exist.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    size_kb = path.stat().st_size / 1024
    log.info(f"Report written to {path} ({size_kb:.1f} KB)")

    # Print summary to console
    meta = report.get("metadata", {})
    print("\n" + "=" * 65)
    print(f"NARRAVERSE ENCODER — REPORT COMPLETE")
    print(f"  Title:      {meta.get('title', '?')}")
    print(f"  Author:     {meta.get('author', '?')}")
    print(f"  Chapters:   {meta.get('total_chapters', 0)}")
    print(f"  Scenes:     {meta.get('total_scenes', 0)}")
    print(f"  Words:      {meta.get('total_words', 0):,}")
    print(f"  Characters: {meta.get('total_unique_characters', 0)}")
    print(f"  Locations:  {meta.get('total_unique_locations', 0)}")
    print(f"  Relations:  {meta.get('total_relationships', 0)}")
    print(f"  Events:     {meta.get('total_causal_events', 0)}")
    print(f"  Avg Tension:{meta.get('avg_tension', 0)}/10")
    print(f"  Output:     {path}")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_image_prompt(name: str, persistent_traits: dict) -> str:
    """
    Builds a text prompt for image generation from persistent character traits.
    Stored on the character profile for use by the image pipeline.
    """
    if not persistent_traits:
        return f"{name}, fictional character"

    trait_parts = []
    # Prioritize visually relevant traits
    priority_keys = ["species", "gender", "age", "hair", "eyes",
                     "height", "build", "skin", "distinguishing_features"]

    for key in priority_keys:
        if key in persistent_traits:
            trait_parts.append(f"{persistent_traits[key]}")

    # Add any remaining traits not in priority list
    for key, value in persistent_traits.items():
        if key not in priority_keys and value:
            trait_parts.append(str(value))

    traits_str = ", ".join(trait_parts) if trait_parts else "fictional character"
    return f"{name}, {traits_str}, fantasy character, detailed portrait"


def _collect_float(collection: list, value) -> None:
    """Safely appends a float value to a collection, ignoring None and non-numeric."""
    if value is None:
        return
    try:
        collection.append(float(value))
    except (ValueError, TypeError):
        pass


def _safe_parse_json(content: str) -> dict:
    """
    Parses JSON from LLM output.
    Strips markdown fences if present.
    Returns empty dict on failure.
    """
    cleaned = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse failed: {e} — snippet: {cleaned[:200]}")
        return {}