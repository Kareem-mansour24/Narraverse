"""
segmentation.py — Phase 2: Semantic scene segmentation
Narraverse Encoder v2

Accepts: list[Chapter] from ingestion.py
Outputs: list[Scene] — flat ordered list of all scenes across the book

A scene boundary is any shift in:
- Time ("The next morning...", "Three days later...")
- Location ("Back at the castle...", "The forest was dark...")
- Narrative focus / POV shift

Scene schema (contract with extraction.py):
{
    "scene_id":           str,   — "ch{N}_sc{N}" e.g. "ch3_sc2"
    "chapter_index":      int,   — which chapter this belongs to
    "chapter_title":      str,   — chapter title for context injection
    "scene_index":        int,   — position within this chapter, 1-based
    "global_scene_index": int,   — position across the whole book, 1-based
    "text":               str,   — raw scene prose (may contain [Illustration: ...])
    "word_count":         int,
    "is_first_scene":     bool,  — first scene of its chapter
    "is_last_scene":      bool,  — last scene of its chapter
}
"""

import json
import logging
from pathlib import Path
from mistralai import Mistral

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [segmentation] %(levelname)s — %(message)s"
)
log = logging.getLogger("segmentation")

# ── Constants ─────────────────────────────────────────────────────────────────

# Model used for scene boundary detection
SEGMENTATION_MODEL = "mistral-large-latest"

# Minimum characters for a scene slice to be kept.
# Filters out artifacts from boundary detection.
SCENE_MIN_CHARS = 50

# Maximum characters sent to the LLM per detection call.
# For long chapters we split into overlapping windows of this size.
WINDOW_SIZE = 8000

# Overlap between windows when a chapter is split.
# Ensures scene boundaries near the split point are not missed.
WINDOW_OVERLAP = 500


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def segment_book(chapters: list[dict], client: Mistral) -> list[dict]:
    """
    Main entry point for segmentation.
    Splits every chapter into scenes and returns a flat ordered list.

    Args:
        chapters:  list[dict] from ingestion.load_book()
        client:    Mistral client instance

    Returns:
        list[dict] — all scenes across the book, matching the scene schema
    """
    all_scenes = []
    global_scene_counter = 0

    for chapter in chapters:
        chapter_index = chapter["chapter_index"]
        chapter_title = chapter["title"]
        chapter_text  = chapter["text"]

        log.info(
            f"Segmenting chapter {chapter_index}: '{chapter_title}' "
            f"({chapter['word_count']} words)"
        )

        # Split chapter into raw scene text blocks
        raw_scenes = split_chapter(chapter_text, chapter_title, client)

        if not raw_scenes:
            log.warning(
                f"No scenes detected in chapter {chapter_index} "
                f"'{chapter_title}' — treating whole chapter as one scene"
            )
            raw_scenes = [chapter_text]

        # Assign metadata to each scene
        chapter_scenes = assign_scene_meta(
            raw_scenes=raw_scenes,
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            global_counter_start=global_scene_counter
        )

        global_scene_counter += len(chapter_scenes)
        all_scenes.extend(chapter_scenes)

        log.info(
            f"  → {len(chapter_scenes)} scenes "
            f"(global #{global_scene_counter - len(chapter_scenes) + 1}"
            f"–{global_scene_counter})"
        )

    log.info(f"Segmentation complete — {len(all_scenes)} total scenes")
    return all_scenes


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2.1 — SCENE BOUNDARY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def split_chapter(chapter_text: str, chapter_title: str,
                  client: Mistral) -> list[str]:
    """
    Splits a chapter's text into a list of scene text blocks.

    For short chapters (under WINDOW_SIZE chars): one LLM call.
    For long chapters: splits into overlapping windows, detects boundaries
    in each window, merges and deduplicates, then slices the full text.

    Args:
        chapter_text:   Full text of the chapter
        chapter_title:  Used in logging and the LLM prompt for context
        client:         Mistral client

    Returns:
        list[str] — raw scene text blocks, in order
    """
    text_len = len(chapter_text)

    if text_len <= WINDOW_SIZE:
        # Short chapter — single detection call
        boundary_indices = detect_scene_boundaries(
            chapter_text, chapter_title, client
        )
    else:
        # Long chapter — overlapping window strategy
        log.info(
            f"  Long chapter ({text_len} chars) — "
            f"using overlapping windows (size={WINDOW_SIZE}, overlap={WINDOW_OVERLAP})"
        )
        boundary_indices = _detect_boundaries_windowed(
            chapter_text, chapter_title, client
        )

    return slice_scenes(chapter_text, boundary_indices)


def detect_scene_boundaries(text: str, chapter_title: str,
                             client: Mistral) -> list[int]:
    """
    Asks the LLM to identify where new scenes begin in the given text.

    The LLM returns the exact first 10-15 words of each scene start.
    We then find those phrases in the original text to get character indices.
    This approach avoids asking the LLM to count characters or line numbers.

    Args:
        text:           Chapter text (or window of it)
        chapter_title:  Injected into prompt for context
        client:         Mistral client

    Returns:
        list[int] — sorted character indices of scene start positions
                    always includes 0 (start of text)
    """
    prompt = f"""You are a structural editor analyzing a chapter from a novel.

Chapter: "{chapter_title}"

Your task: identify where new scenes begin in the text below.

A new scene starts when there is a clear shift in:
- Time (e.g. "The next morning", "Three days later", "That night")
- Location (e.g. "Back at the palace", "The forest stretched ahead")
- Narrative focus or POV

Rules:
- Base your analysis ONLY on what is explicitly in the text.
- Do NOT invent scene breaks that aren't signaled in the text.
- Always include the very first sentence of the text as the first scene start.
- [Illustration: ...] markers are part of the text — do not treat them as scene breaks.

Return ONLY a valid JSON object with this exact structure:
{{
    "scene_starts": [
        "exact first 10-15 words of scene 1",
        "exact first 10-15 words of scene 2"
    ]
}}

Text to analyze:
\"\"\"{text}\"\"\"
"""

    try:
        response = client.chat.complete(
            model=SEGMENTATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        data = _safe_parse_json(content)
        start_phrases = data.get("scene_starts", [])

    except Exception as e:
        log.error(f"Scene boundary detection failed: {e}")
        return [0]  # treat whole text as one scene

    if not start_phrases:
        log.warning("LLM returned no scene starts — treating as one scene")
        return [0]

    log.info(f"  LLM detected {len(start_phrases)} scene start phrase(s)")

    # Convert phrases to character indices by finding them in the original text
    indices = _phrases_to_indices(text, start_phrases)
    return indices


def _detect_boundaries_windowed(chapter_text: str, chapter_title: str,
                                 client: Mistral) -> list[int]:
    """
    Handles long chapters by running boundary detection on overlapping windows.

    Window strategy:
    - Window 1: chars 0 → WINDOW_SIZE
    - Window 2: chars (WINDOW_SIZE - WINDOW_OVERLAP) → (WINDOW_SIZE * 2 - WINDOW_OVERLAP)
    - etc.

    Indices from each window are converted to their absolute position in the
    full chapter text before merging. Duplicates (from the overlap region)
    are removed by deduplication with a tolerance of 100 chars.

    Returns:
        list[int] — sorted, deduplicated absolute character indices
    """
    all_indices = set()
    step = WINDOW_SIZE - WINDOW_OVERLAP
    text_len = len(chapter_text)
    window_start = 0
    window_num = 0

    while window_start < text_len:
        window_end = min(window_start + WINDOW_SIZE, text_len)
        window_text = chapter_text[window_start:window_end]
        window_num += 1

        log.info(
            f"  Window {window_num}: chars {window_start}–{window_end} "
            f"({window_end - window_start} chars)"
        )

        local_indices = detect_scene_boundaries(
            window_text,
            f"{chapter_title} (window {window_num})",
            client
        )

        # Convert local indices to absolute positions in the full text
        for local_idx in local_indices:
            absolute_idx = window_start + local_idx
            all_indices.add(absolute_idx)

        window_start += step

    # Deduplicate indices that are within 100 chars of each other
    # (caused by the same scene boundary being detected in both overlapping windows)
    sorted_indices = sorted(all_indices)
    deduplicated = _deduplicate_indices(sorted_indices, tolerance=100)

    log.info(
        f"  Windowed detection complete: {len(deduplicated)} unique boundaries "
        f"(from {len(sorted_indices)} raw)"
    )
    return deduplicated


def _phrases_to_indices(text: str, phrases: list[str]) -> list[int]:
    """
    Finds the character index of each phrase in the text.
    Skips phrases that can't be found (LLM hallucinated or paraphrased).
    Always ensures index 0 is included.
    Returns sorted, deduplicated list of indices.
    """
    indices = [0]  # always start from the beginning

    for phrase in phrases:
        if not phrase or not isinstance(phrase, str):
            continue

        # Try exact match first
        idx = text.find(phrase)

        # If exact match fails, try with the first 8 words
        # (LLM sometimes adds/drops punctuation at the end)
        if idx == -1:
            short_phrase = " ".join(phrase.split()[:8])
            if short_phrase:
                idx = text.find(short_phrase)

        if idx != -1 and idx not in indices:
            indices.append(idx)
        elif idx == -1:
            log.debug(f"  Phrase not found in text: '{phrase[:50]}...'")

    return sorted(set(indices))


def _deduplicate_indices(indices: list[int], tolerance: int = 100) -> list[int]:
    """
    Removes indices that are within `tolerance` characters of a previous index.
    Keeps the earlier one (lower index).
    """
    if not indices:
        return []

    deduped = [indices[0]]
    for idx in indices[1:]:
        if idx - deduped[-1] > tolerance:
            deduped.append(idx)

    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2.2 — SCENE SLICING
# ══════════════════════════════════════════════════════════════════════════════

def slice_scenes(text: str, boundary_indices: list[int]) -> list[str]:
    """
    Slices the chapter text at the given boundary indices.

    Filters out any slices shorter than SCENE_MIN_CHARS —
    these are artifacts from boundary detection, not real scenes.

    Args:
        text:             Full chapter text
        boundary_indices: Sorted list of character positions where scenes start

    Returns:
        list[str] — raw scene text blocks, in order
    """
    if not boundary_indices:
        return [text.strip()] if text.strip() else []

    # Ensure sorted and unique
    indices = sorted(set(boundary_indices))

    # Ensure we always start from the beginning
    if indices[0] != 0:
        indices.insert(0, 0)

    scenes = []
    for i, start in enumerate(indices):
        end = indices[i + 1] if i + 1 < len(indices) else len(text)
        chunk = text[start:end].strip()

        if len(chunk) >= SCENE_MIN_CHARS:
            scenes.append(chunk)
        else:
            log.debug(
                f"  Discarding short slice at index {start} "
                f"({len(chunk)} chars) — likely an artifact"
            )

    return scenes


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2.3 — SCENE METADATA ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def assign_scene_meta(raw_scenes: list[str], chapter_index: int,
                      chapter_title: str,
                      global_counter_start: int) -> list[dict]:
    """
    Wraps raw scene text blocks into full scene objects with metadata.

    Args:
        raw_scenes:           list of raw scene text strings
        chapter_index:        which chapter these scenes belong to
        chapter_title:        chapter title (injected into extraction prompts later)
        global_counter_start: how many scenes have been processed before this chapter

    Returns:
        list[dict] — scene objects matching the scene schema
    """
    total_in_chapter = len(raw_scenes)
    scenes = []

    for local_index, text in enumerate(raw_scenes, start=1):
        global_index = global_counter_start + local_index

        scene = {
            "scene_id":           f"ch{chapter_index}_sc{local_index}",
            "chapter_index":      chapter_index,
            "chapter_title":      chapter_title,
            "scene_index":        local_index,
            "global_scene_index": global_index,
            "text":               text,
            "word_count":         len(text.split()),
            "is_first_scene":     local_index == 1,
            "is_last_scene":      local_index == total_in_chapter,
        }
        scenes.append(scene)

    return scenes


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_parse_json(content: str) -> dict:
    """
    Parses JSON from LLM output safely.
    Strips markdown fences if present.
    Returns empty dict on failure — callers handle the empty case.
    """
    cleaned = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse failed: {e} — raw content: {cleaned[:200]}")
        return {}