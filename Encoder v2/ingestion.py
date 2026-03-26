"""
ingestion.py — Phase 1: Multi-modal ingestion
Narraverse Encoder v2

Accepts: PDF (digital or scanned), EPUB
Outputs: list[Chapter] — clean chapter objects ready for segmentation

Illustration captions are kept INLINE in the text at their exact position,
formatted as [Illustration: description]. This preserves narrative references
like "the route shown above" that depend on illustration placement.

Chapter schema (contract with segmentation.py):
{
    "chapter_index":  int,   — 1-based, ordered
    "title":          str,   — from TOC or epub spine
    "text":           str,   — narrative text with inline [Illustration: ...] markers
    "word_count":     int,   — word count of narrative text only (illustrations excluded)
    "method":         str,   — "direct" | "ocr" | "hybrid" | "epub" | "ocr-fallback"
}
"""

import re
import base64
import time
import logging
from pathlib import Path
from collections import defaultdict

import fitz  # PyMuPDF

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False

from mistralai import Mistral

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ingestion] %(levelname)s — %(message)s"
)
log = logging.getLogger("ingestion")

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum characters on a page to consider it "has direct text".
# ONLY used to decide whether to run the full-page OCR fallback.
# Does NOT gate whether embedded images are processed — those always are.
TEXT_PRESENCE_THRESHOLD = 50

# Minimum characters for a chapter block to be kept.
# Filters out TOC pages, copyright pages, and blank pages.
CHAPTER_MIN_CHARS = 200

# Mistral vision model for OCR + captioning
VISION_MODEL = "pixtral-12b-2409"

# ── Pixtral unified prompt ────────────────────────────────────────────────────
VISION_PROMPT = """
You are an expert multimodal document analysis AI.

You will receive an image that may contain:
- readable printed or handwritten text, or
- a non-text visual element such as a map, chart, diagram, or illustration.

Your task:
1. If the image contains readable text:
   - Extract the text exactly as printed, preserving spelling, punctuation, and line breaks when possible.
   - Correct obvious OCR noise (e.g., "th1s" → "this").
   - Do NOT summarize or describe — output only the text itself.

2. If the image does NOT contain meaningful readable text:
   - Provide a concise, neutral one-sentence caption describing the visual content.
   - Format the caption exactly as: [Illustration: short description]

3. Never output both text and a caption. Choose one based on the image type.
4. Avoid commentary, speculation, or metadata.

Output format:
<text or [Illustration: ...]>
"""


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_book(file_path: str, api_key: str) -> list[dict]:
    """
    Main entry point for ingestion.
    Detects input format and routes to the correct extraction path.

    Args:
        file_path:  Path to the book file (.pdf or .epub)
        api_key:    Mistral API key — needed for Pixtral OCR

    Returns:
        list[dict] — chapters, each matching the chapter schema above
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Book file not found: {path}")

    client = Mistral(api_key=api_key)
    input_type = detect_input(path)
    log.info(f"Detected input type: {input_type} — {path.name}")

    if input_type in ("pdf_digital", "pdf_scanned"):
        chapters = extract_pdf(path, client, scanned=(input_type == "pdf_scanned"))
    elif input_type == "epub":
        chapters = extract_epub(path)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    log.info(f"Ingestion complete — {len(chapters)} chapters")
    return chapters


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1.1 — FORMAT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_input(path: Path) -> str:
    """
    Determines the input type from the file extension and content.

    For PDFs: samples up to 10 pages to check for a text layer.
    If the majority have extractable text → digital PDF.
    Otherwise → scanned PDF (all pages processed via Pixtral).

    Returns one of:
        "pdf_digital"  — PDF with a usable text layer
        "pdf_scanned"  — PDF without a usable text layer
        "epub"         — EPUB file
    """
    ext = path.suffix.lower()

    if ext == ".epub":
        if not EPUB_AVAILABLE:
            raise ImportError(
                "ebooklib and beautifulsoup4 are required for EPUB support.\n"
                "Run: pip install ebooklib beautifulsoup4"
            )
        return "epub"

    if ext == ".pdf":
        return _detect_pdf_type(path)

    raise ValueError(
        f"Unsupported file type: '{ext}'. "
        f"Narraverse encoder accepts .pdf and .epub files only."
    )


def _detect_pdf_type(path: Path) -> str:
    """
    Samples up to 10 pages of the PDF.
    If more than half have extractable text → digital.
    Otherwise → scanned.
    """
    with fitz.open(path) as doc:
        total = len(doc)
        sample_size = min(10, total)
        pages_with_text = 0

        for i in range(sample_size):
            text = doc[i].get_text("text").strip()
            if len(text) >= TEXT_PRESENCE_THRESHOLD:
                pages_with_text += 1

        ratio = pages_with_text / sample_size
        log.info(
            f"PDF type detection: {pages_with_text}/{sample_size} "
            f"sampled pages have direct text (ratio {ratio:.2f})"
        )
        return "pdf_digital" if ratio >= 0.5 else "pdf_scanned"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1.2 + 1.3 — PDF EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf(path: Path, client: Mistral, scanned: bool = False) -> list[dict]:
    """
    Extracts all pages from a PDF and groups them into chapters.

    Per-page strategy:
    1. Always attempt direct fitz text extraction first (unless scanned=True)
    2. Always process every embedded image with Pixtral — results are inserted
       inline at their position in the text, preserving reading order
    3. If still no content after 1 and 2: full-page OCR fallback via Pixtral

    Args:
        path:     Path to the PDF file
        client:   Mistral client instance
        scanned:  If True, skips direct extraction and uses full-page OCR
                  for every page from the start

    Returns:
        list[dict] — chapters matching the chapter schema
    """
    import shutil
    pages_data = []

    with fitz.open(path) as doc:
        total_pages = len(doc)
        chapter_map = _extract_chapter_map(doc)

        log.info(
            f"Processing {total_pages} pages — "
            f"{len(set(v for v in chapter_map.values() if v))} chapters in TOC"
        )

        temp_dir = path.parent / f"_narraverse_temp_{path.stem}"
        temp_dir.mkdir(exist_ok=True)

        try:
            for i, page in enumerate(doc):
                page_num = i + 1
                log.info(f"  Page {page_num}/{total_pages}...")

                page_data = _extract_page(
                    page=page,
                    page_num=page_num,
                    client=client,
                    temp_dir=temp_dir,
                    force_ocr=scanned
                )
                page_data["chapter_title"] = chapter_map.get(page_num, "Unknown")
                pages_data.append(page_data)

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                log.info("Cleaned up temp image directory")

    return _build_chapters_from_pages(pages_data)


def _extract_chapter_map(doc) -> dict:
    """
    Reads the PDF table of contents.
    Returns {page_number: chapter_title}.
    Falls back to empty dict if no TOC exists.
    """
    toc = doc.get_toc()
    if not toc:
        log.warning("No TOC found in PDF — all content will be treated as one chapter")
        return {}

    chapters = []
    for level, title, page_num in toc:
        if level == 1:
            chapters.append((page_num, title))

    chapters.append((len(doc) + 1, None))  # sentinel: end of book

    chapter_map = {}
    for idx in range(len(chapters) - 1):
        start, title = chapters[idx]
        end, _ = chapters[idx + 1]
        for page_num in range(start, end):
            chapter_map[page_num] = title

    log.info(f"TOC parsed: {len(chapters) - 1} chapters")
    return chapter_map


def _extract_page(page, page_num: int, client: Mistral,
                  temp_dir: Path, force_ocr: bool = False) -> dict:
    """
    Extracts all content from a single PDF page into one ordered text string.

    The key design decision here:
    - Embedded image content (OCR text OR illustration captions) is inserted
      INLINE between surrounding text parts, preserving page reading order.
    - [Illustration: ...] markers stay in the text — they are not extracted
      to a separate field. A paragraph that says "as shown on the map above"
      needs the map marker to appear above it, not stripped away.

    Returns:
        {
            "page_number": int,
            "text":        str,  — ordered: direct text + inline image content
            "method":      str
        }
    """
    content_parts = []  # ordered — position within this list = position on page
    methods_used = set()

    # ── Step 1: Direct text extraction ───────────────────────────────────────
    direct_text = ""
    if not force_ocr:
        direct_text = page.get_text("text").strip()
        if len(direct_text) >= TEXT_PRESENCE_THRESHOLD:
            content_parts.append(direct_text)
            methods_used.add("direct")

    # ── Step 2: Process every embedded image with Pixtral ────────────────────
    # We always do this — no threshold check.
    # Rationale: an image may contain a caption, map legend, or narrative text
    # that has NO presence in the text layer whatsoever. Skipping it based on
    # how much direct text exists would silently lose that content.
    #
    # Results are appended after the direct text in content_parts,
    # which preserves the top-to-bottom reading order of the page.
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list):
        try:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]

            img_path = temp_dir / f"p{page_num}_img{img_index + 1}.{img_ext}"
            img_path.write_bytes(image_bytes)

            ocr_result = _vision_ocr(client, img_path)

            if ocr_result:
                # Whether it's narrative text or [Illustration: ...],
                # insert it inline — not stripped to a side list
                content_parts.append(ocr_result)
                methods_used.add("pixtral-image")

            time.sleep(0.5)  # rate limit protection

        except Exception as e:
            log.warning(f"  Image {img_index + 1} on page {page_num} failed: {e}")

    # ── Step 3: Full-page OCR fallback ───────────────────────────────────────
    # Only triggered if we have NOTHING after steps 1 and 2.
    # This handles fully scanned pages with no text layer and no image xrefs.
    if not content_parts:
        log.info(f"  Page {page_num} — running full-page OCR fallback")
        try:
            pix = page.get_pixmap(dpi=200)  # 200 DPI for better OCR accuracy
            fallback_path = temp_dir / f"p{page_num}_fullpage.png"
            pix.save(str(fallback_path))

            ocr_result = _vision_ocr(client, fallback_path)
            if ocr_result:
                content_parts.append(ocr_result)
                methods_used.add("pixtral-fallback")

        except Exception as e:
            log.error(f"  Full-page OCR fallback failed on page {page_num}: {e}")

    # ── Assemble ──────────────────────────────────────────────────────────────
    raw_combined = "\n\n".join(content_parts).strip()
    clean = normalize_text(raw_combined)

    return {
        "page_number": page_num,
        "text": clean,
        "method": _resolve_method(methods_used)
    }


def _resolve_method(methods_used: set) -> str:
    """Maps the set of methods used on a page to a single descriptive tag."""
    if "pixtral-fallback" in methods_used:
        return "ocr-fallback"
    if "pixtral-image" in methods_used and "direct" in methods_used:
        return "hybrid"
    if "pixtral-image" in methods_used:
        return "ocr"
    return "direct"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1.4 — EPUB EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_epub(path: Path) -> list[dict]:
    """
    Extracts chapters from an EPUB file.

    EPUBs are ZIP archives of HTML files — each ITEM_DOCUMENT is one chapter.
    BeautifulSoup strips HTML tags and extracts clean text.

    <img> tags are replaced with inline [Illustration: ...] markers using
    the alt attribute or a sibling <figcaption>, preserving their position
    in the text exactly as with the PDF path.

    Returns:
        list[dict] — chapters matching the chapter schema
    """
    if not EPUB_AVAILABLE:
        raise ImportError(
            "ebooklib and beautifulsoup4 are required for EPUB support.\n"
            "Run: pip install ebooklib beautifulsoup4"
        )

    book = epub.read_epub(str(path))
    chapters = []
    chapter_index = 1

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        try:
            html_content = item.get_content()
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove non-narrative structural elements
            for tag in soup.find_all(["nav", "header", "footer", "aside"]):
                tag.decompose()

            # Replace <img> tags with inline [Illustration: ...] markers.
            # Priority for caption text: figcaption > alt attribute > generic "image"
            for img_tag in soup.find_all("img"):
                alt = img_tag.get("alt", "").strip()
                figcaption = None
                parent = img_tag.parent
                if parent and parent.name == "figure":
                    figcaption = parent.find("figcaption")

                caption_text = (
                    figcaption.get_text().strip() if figcaption
                    else alt if alt
                    else "image"
                )
                img_tag.replace_with(f"[Illustration: {caption_text}]")

            raw_text = soup.get_text(separator="\n").strip()

            if len(raw_text) < CHAPTER_MIN_CHARS:
                continue  # skip TOC, copyright, blank pages

            title_tag = soup.find(["h1", "h2", "h3"])
            title = title_tag.get_text().strip() if title_tag else f"Chapter {chapter_index}"

            clean_text = normalize_text(raw_text)
            word_count = _count_narrative_words(clean_text)

            chapters.append({
                "chapter_index": chapter_index,
                "title": title,
                "text": clean_text,
                "word_count": word_count,
                "method": "epub"
            })

            chapter_index += 1

        except Exception as e:
            log.warning(f"Failed to process EPUB item '{item.get_name()}': {e}")
            continue

    log.info(f"EPUB extraction complete — {len(chapters)} chapters")
    return chapters


# ══════════════════════════════════════════════════════════════════════════════
# VISION OCR — shared by PDF and image paths
# ══════════════════════════════════════════════════════════════════════════════

def _vision_ocr(client: Mistral, image_path: Path) -> str | None:
    """
    Sends an image to Pixtral-12b for unified OCR / captioning.

    Returns:
        str  — extracted narrative text, OR [Illustration: caption]
        None — on failure or empty response
    """
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        ext = image_path.suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": VISION_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime};base64,{b64}"
                    }
                ]
            }
        ]

        response = client.chat.complete(model=VISION_MODEL, messages=messages)
        result = response.choices[0].message.content.strip()
        return result if result else None

    except Exception as e:
        log.error(f"Pixtral call failed for {image_path.name}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION + CHAPTER BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """
    Cleans raw extracted text without removing any narrative content.

    Removes:
    - Standalone page numbers (lines that are only digits)
    - Very short non-dialogue lines likely to be running headers or footers
    - 3+ consecutive blank lines → collapsed to one blank line

    Preserves:
    - [Illustration: ...] markers exactly as-is
    - All dialogue, punctuation, and narrative prose
    - Single blank lines (some books use them as scene separators)
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Always preserve illustration markers at their exact position
        if stripped.startswith("[Illustration:"):
            cleaned.append(stripped)
            continue

        # Skip standalone page numbers
        if re.fullmatch(r"\d{1,4}", stripped):
            continue

        # Skip very short lines that are likely headers/footers
        # Preserve short dialogue lines: "Yes." "No." "Fine." etc.
        if len(stripped) < 4 and not re.search(r'[.!?"\']', stripped):
            continue

        cleaned.append(stripped)

    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _count_narrative_words(text: str) -> int:
    """
    Word count of narrative text only — excludes [Illustration: ...] markers.
    This keeps word_count meaningful as a measure of actual prose length.
    """
    without_markers = re.sub(r"\[Illustration:[^\]]*\]", "", text)
    return len(without_markers.split())


def _build_chapters_from_pages(pages_data: list[dict]) -> list[dict]:
    """
    Groups page dicts into chapter dicts using the chapter_title on each page.

    Pages with chapter_title == "Unknown" or None are skipped —
    these are front matter, back matter, or blank pages.

    The order of chapters follows the order they first appear in the page list,
    which matches the book's reading order.

    Returns:
        list[dict] — chapters matching the chapter schema
    """
    groups = defaultdict(list)
    seen_titles = []  # insertion-order tracking

    for page in pages_data:
        title = page.get("chapter_title")
        if not title or title == "Unknown":
            continue
        if title not in groups:
            seen_titles.append(title)
        groups[title].append(page)

    if not groups:
        log.warning("No chapter groupings found — check TOC extraction")
        return []

    chapters = []
    for chapter_index, title in enumerate(seen_titles, start=1):
        pages = groups[title]

        combined_text = "\n\n".join(
            p["text"] for p in pages if p["text"]
        ).strip()

        if len(combined_text) < CHAPTER_MIN_CHARS:
            log.debug(f"Skipping short chapter '{title}' ({len(combined_text)} chars)")
            continue

        word_count = _count_narrative_words(combined_text)
        methods = [p.get("method", "direct") for p in pages]

        chapters.append({
            "chapter_index": chapter_index,
            "title": title,
            "text": combined_text,
            "word_count": word_count,
            "method": _most_complex_method(methods)
        })

    log.info(f"Built {len(chapters)} chapters from {len(pages_data)} pages")
    return chapters


def _most_complex_method(methods: list[str]) -> str:
    """
    Returns the most complex extraction method from a list.
    Priority: ocr-fallback > hybrid > ocr > epub > direct
    """
    priority = {
        "ocr-fallback": 4,
        "hybrid":        3,
        "ocr":           2,
        "epub":          1,
        "direct":        0
    }
    return max(methods, key=lambda m: priority.get(m, 0), default="direct")