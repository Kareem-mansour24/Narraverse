"""
encoder.py — Main orchestrator
Narraverse Encoder v2

Entry point for the entire encoding pipeline.
Imports and calls all 4 modules in order, passing outputs between them.
Saves intermediate outputs after each phase so you can inspect and debug
without re-running the whole pipeline.

Usage:
    python encoder.py --file "The Cruel Prince.pdf" --title "The Cruel Prince" --author "Holly Black"

Optional flags:
    --start-from-phase 2    Skip phase 1, load saved phase 1 output instead
    --stop-after-phase 2    Stop after phase 2, don't continue
    --output-dir ./output   Where to save all intermediate and final files
    --no-neo4j              Skip Neo4j ingestion (ingest.py) at the end

Intermediate files saved:
    output/phase1_chapters.json     → after ingestion
    output/phase2_scenes.json       → after segmentation
    output/phase3_scene_results.json → after extraction (all scenes)
    output/full_book_report.json    → after aggregation (final contract file)
"""

import json
import logging
import time
from pathlib import Path
from mistralai import Mistral

# ── Module imports ────────────────────────────────────────────────────────────
from ingestion    import load_book
from segmentation import segment_book
from extraction   import extract_scene
from aggregation  import build_report

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [encoder] %(levelname)s — %(message)s"
)
log = logging.getLogger("encoder")

# ── API key ───────────────────────────────────────────────────────────────────
# Replace with your actual key or load from environment
MISTRAL_API_KEY = "ID2mVb3cRw4oYRlJz0PXgIom6yly2L1B"


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    FILE_PATH   = "Encoder\The Cruel Prince - Holly Black.pdf"
    TITLE       = "The Cruel Prince"
    AUTHOR      = "Holly Black"
    OUTPUT_DIR  = Path(__file__).parent / "output"
    START_PHASE = 1   # set to 2/3/4 to skip earlier phases
    STOP_PHASE  = 4   # set to 1/2/3 to stop early
    # ─────────────────────────────────────────────────────────

    client = Mistral(api_key=MISTRAL_API_KEY)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths for intermediate outputs
    phase1_path  = output_dir / "phase1_chapters.json"
    phase2_path  = output_dir / "phase2_scenes.json"
    phase3_path  = output_dir / "phase3_scene_results.json"
    report_path  = output_dir / "full_book_report.json"

    book_meta = {
        "book_id": _make_book_id(TITLE, AUTHOR),
        "title":   TITLE,
        "author":  AUTHOR,
    }

    start_phase = START_PHASE
    stop_phase  = STOP_PHASE

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — INGESTION
    # ══════════════════════════════════════════════════════════════════════════
    if start_phase <= 1:
        chapters = run_phase1(FILE_PATH, client, phase1_path)
    else:
        log.info(f"Skipping phase 1 — loading from {phase1_path}")
        chapters = load_json(phase1_path)

    inspect_phase1(chapters)

    if stop_phase == 1:
        log.info("Stopping after phase 1 as requested.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — SEGMENTATION
    # ══════════════════════════════════════════════════════════════════════════
    if start_phase <= 2:
        scenes = run_phase2(chapters, client, phase2_path)
    else:
        log.info(f"Skipping phase 2 — loading from {phase2_path}")
        scenes = load_json(phase2_path)

    inspect_phase2(scenes)

    if stop_phase == 2:
        log.info("Stopping after phase 2 as requested.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — EXTRACTION
    # ══════════════════════════════════════════════════════════════════════════
    if start_phase <= 3:
        scene_results, context = run_phase3(scenes, client, phase3_path)
    else:
        log.info(f"Skipping phase 3 — loading from {phase3_path}")
        scene_results, context = load_phase3(phase3_path)

    inspect_phase3(scene_results, context)

    if stop_phase == 3:
        log.info("Stopping after phase 3 as requested.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4 — AGGREGATION
    # ══════════════════════════════════════════════════════════════════════════
    report = run_phase4(
        scene_results=scene_results,
        context=context,
        chapters=chapters,
        book_meta=book_meta,
        client=client,
        report_path=report_path
    )

    inspect_phase4(report)

    log.info("Encoding complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1(file_path: str, client: Mistral, output_path: Path) -> list[dict]:
    """
    Phase 1: Ingestion
    Reads the book file and produces clean chapter objects.
    Saves output to phase1_chapters.json.
    """
    log.info("\n── PHASE 1: INGESTION ─────────────────────────────────────")
    t0 = time.time()

    chapters = load_book(file_path, MISTRAL_API_KEY)

    save_json(chapters, output_path)
    log.info(f"Phase 1 complete in {time.time() - t0:.1f}s — {len(chapters)} chapters")
    log.info(f"Saved: {output_path}")

    return chapters


def run_phase2(chapters: list[dict], client: Mistral,
               output_path: Path) -> list[dict]:
    """
    Phase 2: Segmentation
    Splits every chapter into scenes.
    Saves output to phase2_scenes.json.
    """
    log.info("\n── PHASE 2: SEGMENTATION ──────────────────────────────────")
    t0 = time.time()

    scenes = segment_book(chapters, client)

    save_json(scenes, output_path)
    log.info(f"Phase 2 complete in {time.time() - t0:.1f}s — {len(scenes)} scenes")
    log.info(f"Saved: {output_path}")

    return scenes


def run_phase3(scenes: list[dict], client: Mistral,
               output_path: Path) -> tuple[list[dict], dict]:
    """
    Phase 3: Extraction
    Runs style + entity + event extraction on every scene.
    Saves output to phase3_scene_results.json.

    Saves progress after every scene — if the run crashes at scene 80,
    you can resume from scene 81 using --start-from-phase 3 after
    manually editing phase3_scene_results.json.

    Returns:
        (scene_results, context) — both needed by phase 4
    """
    log.info("\n── PHASE 3: EXTRACTION ────────────────────────────────────")
    t0 = time.time()

    # Initialize global context — born empty, grows with every scene
    context = {
        "known_characters":    {},
        "known_relationships": [],
        "known_locations":     [],
        "known_event_ids":     [],
        "scene_summaries":     []
    }

    scene_results = []
    total = len(scenes)

    for i, scene in enumerate(scenes, start=1):
        scene_t0 = time.time()
        log.info(f"\n  [{i}/{total}] {scene['scene_id']} — {scene['word_count']} words")

        try:
            result = extract_scene(scene, context, client)
            scene_results.append(result)

            log.info(
                f"  ✓ {scene['scene_id']} — "
                f"{len(result.get('characters', []))} chars, "
                f"{len(result.get('events', []))} events, "
                f"tension={result.get('tension_level')}, "
                f"type={result.get('scene_type')} "
                f"({time.time() - scene_t0:.1f}s)"
            )

        except Exception as e:
            log.error(f"  ✗ {scene['scene_id']} extraction failed: {e}")
            # Add a placeholder so scene IDs remain consistent
            scene_results.append(_empty_scene_result(scene))

        # Save progress after every scene
        # This way a crash doesn't lose all previous work
        _save_phase3_progress(scene_results, context, output_path)

    elapsed = time.time() - t0
    log.info(
        f"\nPhase 3 complete in {elapsed:.1f}s "
        f"({elapsed / total:.1f}s per scene average)"
    )
    log.info(f"Saved: {output_path}")

    return scene_results, context


def run_phase4(scene_results: list[dict], context: dict,
               chapters: list[dict], book_meta: dict,
               client: Mistral, report_path: Path) -> dict:
    """
    Phase 4: Aggregation
    Builds the final report from all scene results and context.
    Writes full_book_report.json.
    """
    log.info("\n── PHASE 4: AGGREGATION ───────────────────────────────────")
    t0 = time.time()

    report = build_report(
        scene_results=scene_results,
        context=context,
        chapters=chapters,
        book_meta=book_meta,
        client=client,
        output_path=str(report_path)
    )

    log.info(f"Phase 4 complete in {time.time() - t0:.1f}s")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# INSPECTION FUNCTIONS
# Each prints a human-readable summary of the phase output to the console.
# These help you verify the output is correct before running the next phase.
# ══════════════════════════════════════════════════════════════════════════════

def inspect_phase1(chapters: list[dict]) -> None:
    """
    Prints a summary of the ingestion output.
    Shows: chapter count, titles, word counts, extraction methods.
    Flags any chapter with fewer than 500 words (likely a partial extraction).
    """
    print("\n" + "─" * 65)
    print("PHASE 1 OUTPUT — CHAPTERS")
    print("─" * 65)
    print(f"Total chapters: {len(chapters)}\n")

    for ch in chapters:
        flag = " ⚠ LOW WORD COUNT" if ch.get("word_count", 0) < 500 else ""
        print(
            f"  [{ch['chapter_index']:02d}] {ch['title'][:50]:<50} "
            f"{ch.get('word_count', 0):>6} words  "
            f"[{ch.get('method', '?')}]{flag}"
        )

    # Check for [Illustration: ...] markers
    total_illustrations = sum(
        ch["text"].count("[Illustration:")
        for ch in chapters if ch.get("text")
    )
    print(f"\n  Inline illustration markers: {total_illustrations}")

    # Sample: first 300 chars of chapter 1
    if chapters:
        print(f"\n  Sample — first 300 chars of chapter 1:")
        print(f"  {chapters[0].get('text', '')[:300]!r}")

    print("─" * 65)
    input("\nPress Enter to continue to Phase 2... ")


def inspect_phase2(scenes: list[dict]) -> None:
    """
    Prints a summary of the segmentation output.
    Shows: total scenes, scenes per chapter, word count distribution.
    Flags any scene with fewer than 50 words (likely a segmentation artifact).
    """
    print("\n" + "─" * 65)
    print("PHASE 2 OUTPUT — SCENES")
    print("─" * 65)
    print(f"Total scenes: {len(scenes)}\n")

    # Group by chapter
    from collections import defaultdict
    by_chapter = defaultdict(list)
    for s in scenes:
        by_chapter[s["chapter_index"]].append(s)

    for ch_idx in sorted(by_chapter.keys()):
        ch_scenes = by_chapter[ch_idx]
        title = ch_scenes[0].get("chapter_title", f"Chapter {ch_idx}")
        word_counts = [s["word_count"] for s in ch_scenes]
        avg_words = sum(word_counts) // len(word_counts) if word_counts else 0
        print(
            f"  Ch {ch_idx:02d} | {title[:40]:<40} "
            f"{len(ch_scenes):>3} scenes  avg {avg_words:>5} words/scene"
        )

    # Flag suspiciously short scenes
    short_scenes = [s for s in scenes if s.get("word_count", 0) < 50]
    if short_scenes:
        print(f"\n  ⚠ {len(short_scenes)} scene(s) under 50 words (possible artifacts):")
        for s in short_scenes[:5]:
            print(f"    {s['scene_id']}: {s['word_count']} words — {s['text'][:80]!r}")

    # Word count distribution
    word_counts = [s["word_count"] for s in scenes]
    if word_counts:
        print(f"\n  Word count per scene:")
        print(f"    Min:    {min(word_counts)}")
        print(f"    Max:    {max(word_counts)}")
        print(f"    Avg:    {sum(word_counts) // len(word_counts)}")

    # Sample: first scene text
    if scenes:
        print(f"\n  Sample — first scene ({scenes[0]['scene_id']}):")
        print(f"  {scenes[0].get('text', '')[:300]!r}")

    print("─" * 65)
    input("\nPress Enter to continue to Phase 3... ")


def inspect_phase3(scene_results: list[dict], context: dict) -> None:
    """
    Prints a summary of the extraction output.
    Shows: character registry, relationship count, event count,
    tension distribution, and a sample scene result.
    Flags scenes where extraction failed (empty characters or events).
    """
    print("\n" + "─" * 65)
    print("PHASE 3 OUTPUT — EXTRACTION")
    print("─" * 65)

    # Context summary
    print(f"Global context after all scenes:")
    print(f"  Known characters:    {len(context['known_characters'])}")
    print(f"  Known relationships: {len(context['known_relationships'])}")
    print(f"  Known locations:     {len(context['known_locations'])}")
    print(f"  Known event IDs:     {len(context['known_event_ids'])}")
    print(f"  Scene summaries:     {len(context['scene_summaries'])}")

    # Character registry
    print(f"\nCharacter registry:")
    for char_id, char in context["known_characters"].items():
        appearances = len(char.get("state_history", []))
        print(
            f"  {char_id:<25} {char['primary_name']:<20} "
            f"{appearances:>3} appearances  "
            f"aliases: {', '.join(char.get('all_names', []))}"
        )

    # Extraction quality check
    failed_scenes = [
        s for s in scene_results
        if not s.get("characters") and not s.get("events")
    ]
    if failed_scenes:
        print(f"\n  ⚠ {len(failed_scenes)} scene(s) with empty characters AND events:")
        for s in failed_scenes[:5]:
            print(f"    {s['scene_id']}")

    # Tension distribution
    tensions = [
        s["tension_level"] for s in scene_results
        if isinstance(s.get("tension_level"), int)
    ]
    if tensions:
        from statistics import mean
        print(f"\nTension distribution:")
        print(f"  Average: {mean(tensions):.2f}")
        print(f"  Min:     {min(tensions)}")
        print(f"  Max:     {max(tensions)}")
        buckets = {i: 0 for i in range(1, 11)}
        for t in tensions:
            if 1 <= t <= 10:
                buckets[t] += 1
        bar = "  Distribution: "
        for level, count in buckets.items():
            bar += f"{level}:{'█' * count} "
        print(bar)

    # Event summary
    all_events = [e for s in scene_results for e in s.get("events", [])]
    pillars    = [e for e in all_events if e.get("is_structural_pillar")]
    print(f"\nEvent summary:")
    print(f"  Total events:       {len(all_events)}")
    print(f"  Structural pillars: {len(pillars)}")
    print(f"  Mutable events:     {len(all_events) - len(pillars)}")

    # Sample: one scene result (pick middle of book)
    if scene_results:
        sample = scene_results[len(scene_results) // 2]
        print(f"\nSample scene result ({sample['scene_id']}):")
        print(f"  Summary:   {sample.get('scene_summary', '')[:150]}")
        print(f"  Location:  {sample.get('location', '')}")
        print(f"  Tension:   {sample.get('tension_level')}")
        print(f"  Type:      {sample.get('scene_type')}")
        print(f"  Characters: {[c.get('char_id') for c in sample.get('characters', [])]}")
        print(f"  Events:    {[e.get('event_id') for e in sample.get('events', [])]}")

    print("─" * 65)
    input("\nPress Enter to continue to Phase 4... ")


def inspect_phase4(report: dict) -> None:
    """
    Prints a summary of the final report.
    Shows: all top-level keys, character profiles, causal graph metrics,
    tension profile, and style fingerprint.
    """
    print("\n" + "─" * 65)
    print("PHASE 4 OUTPUT — FULL REPORT")
    print("─" * 65)

    meta = report.get("metadata", {})
    print(f"Book:       {meta.get('title')} by {meta.get('author')}")
    print(f"Chapters:   {meta.get('total_chapters')}")
    print(f"Scenes:     {meta.get('total_scenes')}")
    print(f"Words:      {meta.get('total_words', 0):,}")
    print(f"Characters: {meta.get('total_unique_characters')}")
    print(f"Locations:  {meta.get('total_unique_locations')}")
    print(f"Relations:  {meta.get('total_relationships')}")
    print(f"Events:     {meta.get('total_causal_events')}")
    print(f"Avg tension:{meta.get('avg_tension')}/10")

    # Style fingerprint
    sf = meta.get("style_fingerprint", {})
    print(f"\nStyle fingerprint:")
    print(f"  POV:             {sf.get('primary_pov')}")
    print(f"  Avg sent len:    {sf.get('avg_sentence_length')}")
    print(f"  Adverb density:  {sf.get('adverb_density')}%")
    print(f"  Dialogue ratio:  {sf.get('dialogue_ratio')}%")
    print(f"  Dominant pace:   {sf.get('dominant_pace')}")
    print(f"  Sensory focus:   {sf.get('primary_sensory_focus')}")
    print(f"  Tone keywords:   {sf.get('tone_keywords')}")

    # Tension profile
    tp = report.get("tension_profile", {})
    print(f"\nTension profile:")
    print(f"  Average:         {tp.get('average_tension')}/10")
    print(f"  Sequel potential:{tp.get('sequel_potential_rating')}/10")
    print(f"  Sequel hooks:")
    for hook in tp.get("sequel_hooks", []):
        print(f"    - {hook}")

    # Causal graph
    cg = report.get("causal_graph", {})
    gm = cg.get("graph_metrics", {})
    print(f"\nCausal graph:")
    print(f"  Total events:    {gm.get('total_events')}")
    print(f"  Causal links:    {gm.get('total_causal_links')}")
    print(f"  Pillars:         {gm.get('structural_pillars_count')}")
    print(f"  Mutable events:  {gm.get('mutable_events_count')}")
    print(f"  Divergence pts:  {len(cg.get('divergence_points', []))}")
    print(f"  Sequel seeds:    {len(cg.get('sequel_seeds', []))}")

    # Character profiles summary
    print(f"\nCharacter profiles:")
    for char_id, profile in report.get("characters", {}).items():
        print(
            f"  {char_id:<25} {profile.get('primary_name', ''):<20} "
            f"{profile.get('total_appearances', 0):>3} appearances  "
            f"final state: {profile.get('final_emotional_state', 'unknown')}"
        )

    print("─" * 65)
    print(f"\nFull report saved to: {Path('output') / 'full_book_report.json'}")
    print("Ready for Neo4j ingestion (ingest.py).")


# ══════════════════════════════════════════════════════════════════════════════
# FILE I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_json(data, path: Path) -> None:
    """Saves any JSON-serializable object to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    """Loads a JSON file from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected intermediate file not found: {path}\n"
            f"Run from phase 1 or lower --start-from-phase value."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_phase3_progress(scene_results: list[dict],
                           context: dict, path: Path) -> None:
    """
    Saves phase 3 progress after every scene.
    Stores both scene_results and context in the same file
    so resuming is possible without re-running extraction.
    """
    save_json(
        {"scene_results": scene_results, "context": context},
        path
    )


def load_phase3(path: Path) -> tuple[list[dict], dict]:
    """Loads saved phase 3 output and returns (scene_results, context)."""
    data = load_json(path)
    return data["scene_results"], data["context"]


def _empty_scene_result(scene: dict) -> dict:
    """
    Returns a minimal scene result for scenes where extraction failed.
    Preserves scene metadata so the scene_id chain remains intact.
    """
    return {
        "scene_id":           scene["scene_id"],
        "chapter_index":      scene["chapter_index"],
        "scene_index":        scene["scene_index"],
        "global_scene_index": scene["global_scene_index"],
        "chapter_title":      scene["chapter_title"],
        "scene_summary":      "",
        "tension_level":      5,
        "location":           "Unknown",
        "scene_type":         "UNKNOWN",
        "is_mutable":         True,
        "style":              {},
        "characters":         [],
        "character_states":   [],
        "relationships":      [],
        "events":             [],
        "_extraction_failed": True
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _make_book_id(title: str, author: str) -> str:
    """Generates a clean book_id string from title and author."""
    import re
    combined = f"{title}_{author}".lower()
    clean = re.sub(r"[^a-z0-9_]", "_", combined)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:60]  # cap length


if __name__ == "__main__":
    main()