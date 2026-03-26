import json
import re
import logging
from pathlib import Path
from mistralai import Mistral
from collections import Counter
from statistics import mean, median, stdev

# === Configuration ===
API_KEY = "ID2mVb3cRw4oYRlJz0PXgIom6yly2L1B"
MODEL_NAME = "pixtral-12b-2409"
INPUT_FILE = Path("The Cruel Prince - Holly Black-pixtral-chapters.json")
OUTPUT_FILE = Path("full_book_scene_analysis_report_full1.json")

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# === Initialize Mistral ===
client = Mistral(api_key=API_KEY)


# ==============================================================================
# 1. SMART SCENE DETECTION
# ==============================================================================
def smart_split_scenes(chapter_text, chapter_name):
    """
    Uses the LLM to semantically identify scene breaks.
    """
    logging.info(f"🔍 Asking AI to identify scene breaks in {chapter_name}...")

    prompt = f"""
    You are a structural editor. Analyze the text below and identify where new scenes begin.
    A new scene is defined by a change in:
    - Time (e.g., "The next morning...")
    - Location (e.g., "Back at the castle...")
    - Narrative Focus

    CRITICAL: Base your analysis ONLY on what is explicitly present in the text. Do not invent or assume scene breaks.

    Return ONLY a JSON object with a key "scene_starts". 
    The value must be a list of strings, where each string is the EXACT first 10-15 words of the sentence that starts a new scene.
    Always include the very first sentence of the chapter as the first scene.

    Text to Analyze:
    "{chapter_text[:8000]}..." 
    """

    try:
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()

        try:
            data = json.loads(content)
            start_phrases = data.get("scene_starts", [])
        except json.JSONDecodeError:
            clean_json = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            start_phrases = data.get("scene_starts", [])

    except Exception as e:
        logging.error(f"❌ AI Scene detection failed: {e}")
        return [{"scene_index": 1, "content": chapter_text}]

    logging.info(f"✅ AI detected {len(start_phrases)} scenes.")

    indices = []
    for phrase in start_phrases:
        idx = chapter_text.find(phrase)
        if idx != -1:
            indices.append(idx)

    indices.sort()

    if not indices or indices[0] != 0:
        indices.insert(0, 0)
    indices = sorted(list(set(indices)))

    scenes = []
    for i in range(len(indices)):
        start_idx = indices[i]
        end_idx = indices[i + 1] if i + 1 < len(indices) else len(chapter_text)
        chunk = chapter_text[start_idx:end_idx].strip()

        if len(chunk) > 50:
            scenes.append({
                "scene_index": len(scenes) + 1,
                "content": chunk
            })

    return scenes


# ==============================================================================
# 2. ANALYSIS ENGINES
# ==============================================================================
def analyze_style_neuro_symbolic(text):
    """
    Analyzes pacing, tone, hard metrics, and scene type classification.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b\w+\b', text.lower())

    asl = (len(words) / len(sentences)) if sentences else 0
    adverbs = len([w for w in words if w.endswith('ly')])
    adv_density = (adverbs / len(words)) * 100 if words else 0

    prompt = f"""
    You are a literary analyst. Analyze the writing style of this text in detail.
    
    CRITICAL INSTRUCTION: Base your analysis EXCLUSIVELY on what is present in the provided text. 
    Do NOT infer, assume, or add information not explicitly stated in the text.
    
    Return ONLY a valid JSON object (no markdown formatting) with these keys:
    - "pacing": Choose EXACTLY ONE from this list: ["Fast", "Medium", "Slow", "Variable"]
    - "pov": Choose EXACTLY ONE from this list: ["1st Person", "2nd Person", "3rd Person Limited", "3rd Person Omniscient", "Multiple POV", "Epistolary", "Stream of Consciousness"]
    - "tone": Provide a detailed multi-layered tone analysis (30-40 words) based ONLY on language, word choice, and atmosphere present in the text. Include primary tone, undertones, emotional texture, and atmosphere.
    - "sensory_focus": Choose EXACTLY ONE primary sensory focus from this list: ["Visual", "Auditory", "Internal Monologue", "Tactile", "Olfactory", "Kinesthetic", "Mixed"]
    - "scene_type": Choose the ONE most dominant type from this list based ONLY on what occurs in the text:
      ["ACTION", "DIALOGUE", "INTROSPECTION", "DESCRIPTION", "EXPOSITION", "TRANSITION", "ROMANCE", "CONFLICT", "REVELATION", "WORLD_BUILDING"]

    CRITICAL: For "pacing", "pov", "sensory_focus", and "scene_type", return ONLY the exact label from the list with NO explanations, descriptions, or additional text.

    Text Snippet:
    "{text[:1500]}..."
    """
    try:
        res = client.chat.complete(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        soft_metrics = res.choices[0].message.content
    except Exception as e:
        soft_metrics = f'{{"error": "{e}"}}'

    return {
        "hard_metrics": {
            "avg_sentence_length": round(asl, 2),
            "adverb_percentage": f"{round(adv_density, 2)}%",
            "total_words": len(words)
        },
        "soft_metrics": soft_metrics
    }


def analyze_scene_entities(text, scene_index, chapter_name, context):
    """
    Extracts Location, Characters, and Inter-character Relationships.
    Uses full narrative context from ALL previous scenes.
    """
    # Build comprehensive context
    context_str = ""
    
    if context["known_characters"]:
        context_str += "\n=== KNOWN CHARACTERS (from all previous chapters/scenes) ===\n"
        for char_id, char_info in context["known_characters"].items():
            names_str = f"Names used: {', '.join(char_info.get('all_names', [char_info['primary_name']]))}"
            visual_str = ""
            if char_info.get('visual_traits'):
                visual_str = f" | Visual: {'; '.join(char_info['visual_traits'])}"
            context_str += f"- {char_id}: {char_info['primary_name']} ({names_str}){visual_str}\n"
            context_str += f"  First seen: {char_info['first_appearance']}\n"
    
    if context["previous_summaries"]:
        # Include last 10 summaries for immediate context
        recent_summaries = context["previous_summaries"][-10:]
        context_str += "\n=== RECENT SCENES SUMMARY ===\n"
        for summary in recent_summaries:
            context_str += f"{summary['chapter']} - Scene {summary['scene_index']}: {summary['summary']}\n"
    
    if context["known_relationships"]:
        context_str += "\n=== ESTABLISHED RELATIONSHIPS ===\n"
        for rel in context["known_relationships"]:
            context_str += f"- {rel['from']} → {rel['to']}: {rel['type']} ({rel['dynamic']})\n"
    
    if context["known_locations"]:
        context_str += "\n=== KNOWN LOCATIONS ===\n"
        for loc in context["known_locations"]:
            context_str += f"- {loc}\n"
    
    prompt = f"""
    Analyze the narrative entities in this specific scene from {chapter_name}.
    
    {context_str}
    
    CRITICAL INSTRUCTIONS:
    1. Analyze ONLY what is explicitly present in the provided text. Do NOT invent, assume, or infer information.
    2. CHARACTERS: If a character from "KNOWN CHARACTERS" appears, MUST use their existing ID. Only create new IDs for characters explicitly mentioned in this scene who are NEW.
    3. CHARACTER DETAILS: Extract ALL explicitly mentioned details:
       - Any names, nicknames, or titles used in the text
       - Physical descriptions (hair color, eye color, height, clothing, scars, etc.) ONLY if explicitly mentioned
       - Do NOT describe visual traits that are not in the text
    4. RELATIONSHIPS: Check "ESTABLISHED RELATIONSHIPS" - do NOT repeat existing relationships unless the dynamic has explicitly changed in this scene.
    5. LOCATIONS: Use exact descriptions from the text. Reference known locations if explicitly revisiting them.
    6. Character IDs format: char_<name_lowercase> (e.g., char_jude, char_cardan)
    7. SCENE SUMMARY: Summarize ONLY the events that explicitly occur in this scene text. Do not add context or implications.
    
    Return ONLY a valid JSON object (no markdown formatting):

    {{
        "location": "Exact description of the physical setting as described in the text",
        "scene_summary": "Detailed 3-4 sentence recap of ONLY the events, actions, and dialogue that explicitly occur in this scene text",
        "tension_level": "Rate 1-10 based on conflict, stakes, or emotional intensity explicitly present in the scene",
        "characters": [
            {{
                "id": "char_<name>",
                "names_used": ["Name1", "Nickname", "Title"],
                "visual_traits": ["trait explicitly mentioned in text", "another visual detail from text"],
                "role": "Role in this specific scene based ONLY on their actions in the text",
                "emotional_state": "Current feeling based ONLY on explicit descriptions or dialogue",
                "quote": "Short characteristic line of dialogue from the text (if any)"
            }}
        ],
        "relationships": [
            {{
                "from": "char_<name1>",
                "to": "char_<name2>",
                "type": "RELATIONSHIP_TYPE",
                "dynamic": "Description of their interaction/tension in this scene based ONLY on explicit text (15-25 words)"
            }}
        ]
    }}

    IMPORTANT: Only include visual_traits if they are EXPLICITLY described in the text. Empty array if none mentioned.
    Only include NEW relationships or relationships with CHANGED dynamics explicitly shown in this scene.
    Relationship types: CAREGIVER_OF, ENEMY_OF, ALLY_WITH, LOVER_OF, SIBLING_OF, MENTOR_TO, SERVANT_OF, RIVAL_OF, etc.

    Text:
    "{text[:3000]}..."
    """
    try:
        res = client.chat.complete(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        return res.choices[0].message.content
    except Exception as e:
        return f'{{"error": "{e}"}}'


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def clean_json_str(s):
    """Remove Markdown fences and parse JSON"""
    s = s.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(s)
    except Exception:
        return {"raw_output": s, "error": "JSON parse failed"}


def update_context(context, scene_analysis, chapter_name):
    """Updates the running context with new scene data - UPDATES not overwrites"""
    # Update characters - ACCUMULATE data, don't overwrite
    for char in scene_analysis.get("characters", []):
        char_id = char.get("id")
        names_used = char.get("names_used", [])
        visual_traits = char.get("visual_traits", [])
        
        if char_id:
            if char_id not in context["known_characters"]:
                # New character - initialize
                primary_name = names_used[0] if names_used else "Unknown"
                context["known_characters"][char_id] = {
                    "primary_name": primary_name,
                    "all_names": list(set(names_used)),
                    "visual_traits": list(set(visual_traits)),
                    "first_appearance": f"{chapter_name} - Scene {scene_analysis['scene_index']}",
                    "appearances": []
                }
                logging.info(f"   ➕ New character: {char_id} ({primary_name})")
            else:
                # Existing character - UPDATE by adding new info
                existing = context["known_characters"][char_id]
                
                # Add new names (don't overwrite)
                for name in names_used:
                    if name not in existing["all_names"]:
                        existing["all_names"].append(name)
                        logging.info(f"   🔄 Added name to {char_id}: '{name}'")
                
                # Add new visual traits (don't overwrite)
                for trait in visual_traits:
                    if trait not in existing["visual_traits"]:
                        existing["visual_traits"].append(trait)
                        logging.info(f"   👁️ Added visual trait to {char_id}: '{trait}'")
            
            # Track this appearance
            context["known_characters"][char_id]["appearances"].append({
                "chapter": chapter_name,
                "scene": scene_analysis['scene_index'],
                "role": char.get("role", ""),
                "emotional_state": char.get("emotional_state", "")
            })
    
    # Update relationships - TRACK EVOLUTION over time
    for rel in scene_analysis.get("relationships", []):
        char_from = rel.get("from")
        char_to = rel.get("to")
        rel_type = rel.get("type")
        dynamic = rel.get("dynamic", "")
        
        if not char_from or not char_to or not rel_type:
            continue
        
        # Create normalized key (handles bidirectional relationships)
        # Sort the IDs to ensure A-B and B-A map to the same relationship
        pair_key = tuple(sorted([char_from, char_to]))
        
        # Find if this relationship pair already exists
        existing_rel = None
        existing_index = None
        
        for idx, existing in enumerate(context["known_relationships"]):
            existing_pair = tuple(sorted([existing["from"], existing["to"]]))
            
            # Same pair and same type = update this relationship
            if existing_pair == pair_key and existing["type"] == rel_type:
                existing_rel = existing
                existing_index = idx
                break
        
        if existing_rel:
            # Relationship exists - track its evolution
            if "evolution" not in existing_rel:
                # First time tracking evolution - convert existing to evolution format
                existing_rel["evolution"] = [
                    {
                        "chapter": existing_rel.get("first_seen", "Unknown"),
                        "scene": existing_rel.get("first_scene", 0),
                        "dynamic": existing_rel["dynamic"]
                    }
                ]
            
            # Check if dynamic has actually changed
            last_dynamic = existing_rel["evolution"][-1]["dynamic"]
            if dynamic != last_dynamic:
                # Dynamic has evolved - add new entry
                existing_rel["evolution"].append({
                    "chapter": chapter_name,
                    "scene": scene_analysis['scene_index'],
                    "dynamic": dynamic
                })
                # Update current dynamic
                existing_rel["dynamic"] = dynamic
                existing_rel["last_updated"] = f"{chapter_name} - Scene {scene_analysis['scene_index']}"
                logging.info(f"   🔄 Relationship evolved: {char_from} → {char_to} ({rel_type})")
            
        else:
            # New relationship - initialize with evolution tracking
            new_rel = {
                "from": char_from,
                "to": char_to,
                "type": rel_type,
                "dynamic": dynamic,
                "first_seen": f"{chapter_name} - Scene {scene_analysis['scene_index']}",
                "first_scene": scene_analysis['scene_index'],
                "last_updated": f"{chapter_name} - Scene {scene_analysis['scene_index']}",
                "evolution": [
                    {
                        "chapter": chapter_name,
                        "scene": scene_analysis['scene_index'],
                        "dynamic": dynamic
                    }
                ]
            }
            context["known_relationships"].append(new_rel)
            logging.info(f"   ➕ New relationship: {char_from} → {char_to} ({rel_type})")
    
    # Update locations
    loc = scene_analysis.get("location", "")
    if loc and loc not in context["known_locations"]:
        context["known_locations"].append(loc)
    
    # Update summaries with tension
    context["previous_summaries"].append({
        "chapter": chapter_name,
        "scene_index": scene_analysis["scene_index"],
        "summary": scene_analysis.get("scene_summary", ""),
        "tension_level": scene_analysis.get("tension_level", 5)
    })


def generate_chapter_summary(chapter_scenes):
    """Generate a comprehensive summary for a single chapter"""
    scenes_text = "\n\n".join([
        f"Scene {s['scene_index']}: {s['scene_summary']}"
        for s in chapter_scenes
    ])
    
    prompt = f"""
    Create a comprehensive summary of this chapter based ONLY on the scene summaries provided.
    
    Do NOT add information not present in the summaries.
    
    Scene summaries:
    {scenes_text}
    
    Return ONLY a JSON object:
    {{
        "chapter_summary": "A comprehensive 4-6 sentence summary covering all major events and developments in this chapter"
    }}
    """
    
    try:
        res = client.chat.complete(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        result = clean_json_str(res.choices[0].message.content)
        return result.get("chapter_summary", "Summary generation failed")
    except Exception as e:
        logging.error(f"Failed to generate chapter summary: {e}")
        return "Summary generation failed"


def generate_book_summary(all_chapter_summaries):
    """Generate a comprehensive summary for the entire book"""
    chapters_text = "\n\n".join([
        f"{s['chapter_name']}: {s['summary']}"
        for s in all_chapter_summaries
    ])
    
    prompt = f"""
    Create a comprehensive summary of the entire book based ONLY on the chapter summaries provided.
    
    Do NOT add information not present in the summaries.
    
    Chapter summaries:
    {chapters_text}
    
    Return ONLY a JSON object:
    {{
        "book_summary": "A comprehensive 8-12 sentence summary covering the entire narrative arc, major plot points, character developments, and key themes"
    }}
    """
    
    try:
        res = client.chat.complete(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        result = clean_json_str(res.choices[0].message.content)
        return result.get("book_summary", "Summary generation failed")
    except Exception as e:
        logging.error(f"Failed to generate book summary: {e}")
        return "Summary generation failed"


def generate_tension_profile(all_chapters_analysis):
    """Generate tension arc and profile from all scenes"""
    logging.info("📊 Generating tension profile...")
    
    # Collect all tension data
    tension_data = []
    for chapter in all_chapters_analysis:
        for scene in chapter["scenes"]:
            tension_level = scene.get("tension_level", 5)
            try:
                tension_num = int(tension_level) if isinstance(tension_level, (int, str)) else 5
            except:
                tension_num = 5
            
            tension_data.append({
                "chapter": chapter["chapter_name"],
                "scene": scene["scene_index"],
                "tension": tension_num,
                "summary": scene.get("scene_summary", "")
            })
    
    # Find peaks and valleys
    high_tension_scenes = sorted([t for t in tension_data if t["tension"] >= 8], key=lambda x: x["tension"], reverse=True)[:5]
    low_tension_scenes = sorted([t for t in tension_data if t["tension"] <= 3], key=lambda x: x["tension"])[:3]
    
    # Build context for AI analysis
    context_text = "HIGH TENSION SCENES:\n"
    for ht in high_tension_scenes:
        context_text += f"- {ht['chapter']} Scene {ht['scene']} (Tension: {ht['tension']}): {ht['summary']}\n"
    
    context_text += "\nLOW TENSION SCENES:\n"
    for lt in low_tension_scenes:
        context_text += f"- {lt['chapter']} Scene {lt['scene']} (Tension: {lt['tension']}): {lt['summary']}\n"
    
    prompt = f"""
    Analyze the tension profile of this narrative based ONLY on the scene data provided.
    
    {context_text}
    
    Return ONLY a JSON object:
    {{
        "overall_tension_pattern": "Description of how tension builds and releases throughout the story (40-50 words)",
        "climax_description": "Description of the highest tension moment(s) (30-40 words)",
        "resolution_style": "How the story resolves tension (20-30 words)",
        "sequel_hooks": ["Unresolved tension 1", "Unresolved tension 2", "Potential conflict for sequel"],
        "sequel_potential_rating": "Rate 1-10 how suitable this story is for a sequel based on unresolved elements"
    }}
    """
    
    try:
        res = client.chat.complete(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        result = clean_json_str(res.choices[0].message.content)
        
        # Add quantitative data
        avg_tension = mean([t["tension"] for t in tension_data])
        
        return {
            "average_tension": round(avg_tension, 2),
            "highest_tension_scenes": [{"chapter": ht["chapter"], "scene": ht["scene"], "level": ht["tension"]} for ht in high_tension_scenes],
            "lowest_tension_scenes": [{"chapter": lt["chapter"], "scene": lt["scene"], "level": lt["tension"]} for lt in low_tension_scenes],
            "overall_tension_pattern": result.get("overall_tension_pattern", ""),
            "climax_description": result.get("climax_description", ""),
            "resolution_style": result.get("resolution_style", ""),
            "sequel_hooks": result.get("sequel_hooks", []),
            "sequel_potential_rating": result.get("sequel_potential_rating", "N/A")
        }
    except Exception as e:
        logging.error(f"Failed to generate tension profile: {e}")
        return {"error": str(e)}


def generate_character_profiles(global_context):
    """Generate comprehensive character profiles from all accumulated data"""
    logging.info("👥 Generating comprehensive character profiles...")
    
    character_profiles = {}
    
    for char_id, char_data in global_context["known_characters"].items():
        # Gather all appearance data
        all_roles = [app["role"] for app in char_data["appearances"] if app.get("role")]
        all_emotions = [app["emotional_state"] for app in char_data["appearances"] if app.get("emotional_state")]
        
        # Build context for this character
        char_context = f"""
        Character ID: {char_id}
        Primary Name: {char_data['primary_name']}
        All Names Used: {', '.join(char_data['all_names'])}
        Visual Traits: {', '.join(char_data['visual_traits']) if char_data['visual_traits'] else 'None explicitly described'}
        First Appearance: {char_data['first_appearance']}
        Total Appearances: {len(char_data['appearances'])}
        Roles Played: {', '.join(set(all_roles))}
        Emotional States Observed: {', '.join(set(all_emotions))}
        """
        
        # Find relationships involving this character
        character_relationships = []
        for rel in global_context["known_relationships"]:
            if rel["from"] == char_id:
                character_relationships.append(f"→ {rel['to']}: {rel['type']} ({rel['dynamic']})")
            elif rel["to"] == char_id:
                character_relationships.append(f"← {rel['from']}: {rel['type']} ({rel['dynamic']})")
        
        rel_text = "\n".join(character_relationships) if character_relationships else "No relationships explicitly defined"
        
        prompt = f"""
        Create a comprehensive character profile based ONLY on the data provided. Do NOT invent or assume information.
        
        {char_context}
        
        Relationships:
        {rel_text}
        
        Return ONLY a JSON object:
        {{
            "overall_role": "Their role in the story (20-30 words, based ONLY on observed roles)",
            "character_arc": "How they develop through the story (30-40 words, based ONLY on appearances)",
            "key_traits": ["trait1", "trait2", "trait3"],
            "personality_summary": "Comprehensive personality description (40-50 words, based ONLY on emotional states and roles)"
        }}
        """
        
        try:
            res = client.chat.complete(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
            profile = clean_json_str(res.choices[0].message.content)
            
            character_profiles[char_id] = {
                "character_id": char_id,
                "primary_name": char_data['primary_name'],
                "all_names": char_data['all_names'],
                "visual_traits": char_data['visual_traits'],
                "first_appearance": char_data['first_appearance'],
                "total_appearances": len(char_data['appearances']),
                "overall_role": profile.get("overall_role", ""),
                "character_arc": profile.get("character_arc", ""),
                "key_traits": profile.get("key_traits", []),
                "personality_summary": profile.get("personality_summary", ""),
                "all_appearances": char_data['appearances']
            }
            
            logging.info(f"   ✅ Generated profile for {char_id}")
            
        except Exception as e:
            logging.error(f"   ❌ Failed to generate profile for {char_id}: {e}")
            character_profiles[char_id] = {
                "character_id": char_id,
                "primary_name": char_data['primary_name'],
                "all_names": char_data['all_names'],
                "visual_traits": char_data['visual_traits'],
                "error": str(e)
            }
    
    return character_profiles


def generate_author_bible(all_chapters_analysis):
    """Generate comprehensive writing style analysis combining all metrics"""
    logging.info("📖 Generating Author Bible...")
    
    # Collect all metrics
    all_asl = []
    all_adverb_pct = []
    all_words = []
    all_scene_types = []
    all_povs = []
    all_pacing = []
    all_tones = []
    all_sensory = []
    
    for chapter in all_chapters_analysis:
        for scene in chapter["scenes"]:
            # Hard metrics
            all_asl.append(scene["style_metrics"]["avg_sentence_length"])
            adv_str = scene["style_metrics"]["adverb_percentage"].replace("%", "")
            try:
                all_adverb_pct.append(float(adv_str))
            except:
                pass
            all_words.append(scene["style_metrics"]["total_words"])
            
            # Soft metrics
            qual = scene.get("style_qualitative", {})
            if qual.get("scene_type"):
                all_scene_types.append(qual["scene_type"])
            if qual.get("pov"):
                all_povs.append(qual["pov"])
            if qual.get("pacing"):
                all_pacing.append(qual["pacing"])
            if qual.get("tone"):
                all_tones.append(qual["tone"])
            if qual.get("sensory_focus"):
                # Handle both string and list formats
                sensory = qual["sensory_focus"]
                if isinstance(sensory, list):
                    all_sensory.extend(sensory)
                elif isinstance(sensory, str):
                    all_sensory.append(sensory)
                else:
                    all_sensory.append(str(sensory))
    
    # Calculate statistics
    avg_asl = mean(all_asl) if all_asl else 0
    median_asl = median(all_asl) if all_asl else 0
    stdev_asl = stdev(all_asl) if len(all_asl) > 1 else 0
    
    avg_adverb = mean(all_adverb_pct) if all_adverb_pct else 0
    
    avg_scene_words = mean(all_words) if all_words else 0
    median_scene_words = median(all_words) if all_words else 0
    
    # Distributions
    scene_type_dist = Counter(all_scene_types)
    total_scenes = len(all_scene_types)
    scene_type_percentages = {
        st: {
            "count": count,
            "percentage": round((count / total_scenes) * 100, 2)
        }
        for st, count in scene_type_dist.items()
    }
    
    pov_dist = Counter(all_povs)
    pov_percentages = {
        pov: {
            "count": count,
            "percentage": round((count / len(all_povs)) * 100, 2) if all_povs else 0
        }
        for pov, count in pov_dist.items()
    }
    
    pacing_dist = Counter([p.split(',')[0].strip() if ',' in p else p.split('-')[0].strip() for p in all_pacing])
    sensory_dist = Counter(all_sensory)
    
    # Generate AI synthesis
    metrics_summary = f"""
    Average Sentence Length: {avg_asl:.2f} words
    Adverb Usage: {avg_adverb:.2f}%
    Average Scene Length: {avg_scene_words:.0f} words
    
    Scene Type Distribution: {dict(scene_type_dist)}
    POV Distribution: {dict(pov_dist)}
    Pacing Distribution: {dict(pacing_dist)}
    Sensory Focus Distribution: {dict(sensory_dist)}
    
    Sample Tones: {all_tones[:5]}
    """
    
    prompt = f"""
    Create a comprehensive "Author Bible" - a writing style guide based ONLY on the metrics provided.
    
    {metrics_summary}
    
    Return ONLY a JSON object:
    {{
        "prose_signature": "Distinctive characteristics of the prose style (40-50 words)",
        "narrative_voice": "Description of the narrative perspective and voice (30-40 words)",
        "typical_scene_structure": "Common patterns in scene construction (30-40 words)",
        "emotional_range": "Range and patterns of emotional expression (30-40 words)",
        "stylistic_strengths": ["strength1", "strength2", "strength3"],
        "writing_recommendations_for_sequel": ["recommendation1", "recommendation2", "recommendation3"]
    }}
    """
    
    try:
        res = client.chat.complete(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        ai_analysis = clean_json_str(res.choices[0].message.content)
    except Exception as e:
        logging.error(f"Failed AI synthesis: {e}")
        ai_analysis = {"error": str(e)}
    
    return {
        "quantitative_metrics": {
            "sentence_structure": {
                "average_sentence_length": round(avg_asl, 2),
                "median_sentence_length": round(median_asl, 2),
                "sentence_length_variation": round(stdev_asl, 2)
            },
            "word_usage": {
                "average_adverb_percentage": f"{round(avg_adverb, 2)}%",
                "average_scene_word_count": round(avg_scene_words, 0),
                "median_scene_word_count": round(median_scene_words, 0)
            },
            "scene_composition": {
                "scene_type_distribution": scene_type_percentages,
                "most_common_scene_type": scene_type_dist.most_common(1)[0][0] if scene_type_dist else "N/A"
            }
        },
        "qualitative_patterns": {
            "point_of_view": {
                "distribution": pov_percentages,
                "primary_pov": pov_dist.most_common(1)[0][0] if pov_dist else "N/A"
            },
            "pacing_patterns": {
                "distribution": dict(pacing_dist),
                "dominant_pace": pacing_dist.most_common(1)[0][0] if pacing_dist else "N/A"
            },
            "sensory_preferences": {
                "distribution": dict(sensory_dist),
                "primary_sensory_focus": sensory_dist.most_common(1)[0][0] if sensory_dist else "N/A"
            }
        },
        "ai_synthesis": ai_analysis
    }


# ==============================================================================
# 3. CAUSAL GRAPH EXTRACTION (POST-PROCESSING)
# ==============================================================================
def extract_causal_graph(all_chapters_analysis, global_context):
    """
    Analyzes all scene data to build a comprehensive causal graph.
    
    Called AFTER all chapters are processed.
    Uses existing rich scene data (summaries + characters + relationships + tension).
    """
    logging.info("🔗 Building causal event graph from all scenes...")
    
    # Collect all scene data into structured format
    scene_events_data = []
    
    for chapter in all_chapters_analysis:
        chapter_name = chapter["chapter_name"]
        
        for scene in chapter["scenes"]:
            scene_events_data.append({
                "id": f"scene_{chapter_name.replace(' ', '_')}_{scene['scene_index']}",
                "chapter": chapter_name,
                "scene_index": scene["scene_index"],
                "summary": scene["scene_summary"],
                "location": scene["location"],
                "characters": [char["id"] for char in scene.get("characters", [])],
                "tension": scene["tension_level"],
                "scene_type": scene.get("scene_type", "UNKNOWN")
            })
    logging.info(f"   📊 Collected {len(scene_events_data)} total scenes")
    # Build context summary for LLM
    context_summary = _build_causal_context(scene_events_data, global_context)
    logging.info(f"   📝 Context size: {len(context_summary)} characters")
    # Generate causal graph
    prompt = f"""
    You are a narrative causality analyst. Analyze this story's event structure.
    
    {context_summary}
    
    Create a causal dependency graph by:
    1. Identifying major EVENTS (not states/descriptions - must be actions that change story)
    2. Mapping causal relationships between events
    3. Identifying critical vs flexible story elements
    
    CRITICAL: Base analysis ONLY on provided scene data. Do NOT invent events.
    
    Return ONLY a valid JSON object:
    {{
        "events": [
            {{
                "id": "evt_<unique_identifier>",
                "description": "Concise description of what happened (15-25 words)",
                "source_scene": "scene_id where this occurred",
                "chapter": "chapter name",
                "event_type": "ACTION | DISCOVERY | DECISION | REVELATION | CONFLICT | TRANSFORMATION",
                "story_impact": "Rate 1-10 how much this changes the overall narrative",
                "reversibility": "Rate 1-10 how easily this could be undone (1=permanent, 10=easily reversed)",
                
                "caused_by": [
                    {{
                        "event_id": "evt_prerequisite",
                        "relationship": "ENABLES | REQUIRES | TRIGGERS",
                        "explanation": "Why this prerequisite was necessary (10-15 words)"
                    }}
                ],
                
                "causes": [
                    {{
                        "event_id": "evt_consequence",
                        "relationship": "ENABLES | REQUIRES | TRIGGERS",
                        "explanation": "How this led to the consequence (10-15 words)"
                    }}
                ],
                
                "prevents": [
                    {{
                        "alternative": "What path/action this event blocked (15-20 words)",
                        "why_blocked": "Mechanism of blocking (10-15 words)"
                    }}
                ],
                
                "required_for": [
                    {{
                        "event_id": "evt_dependent",
                        "why_required": "Why the dependent event needs this (10-15 words)"
                    }}
                ]
            }}
        ],
        
        "critical_path": [
            {{
                "event_id": "evt_id",
                "why_critical": "Why this event is essential to the story's ending (15-25 words)",
                "criticality_score": "Rate 1-10"
            }}
        ],
        
        "flexible_events": [
            {{
                "event_id": "evt_id",
                "why_flexible": "Why this could change without breaking the story (15-25 words)",
                "flexibility_score": "Rate 1-10"
            }}
        ],
        
        "causal_chains": [
            {{
                "chain_id": "chain_<n>",
                "description": "Description of this causal sequence (20-30 words)",
                "event_sequence": ["evt_1", "evt_2", "evt_3"],
                "chain_type": "LINEAR | BRANCHING | CONVERGENT",
                "story_function": "What narrative purpose this chain serves (15-25 words)"
            }}
        ],
        
        "divergence_points": [
            {{
                "event_id": "evt_id",
                "decision_made": "What choice was made",
                "alternatives": ["Alternative choice 1", "Alternative choice 2"],
                "divergence_potential": "Rate 1-10 how different story would be with alternate choice",
                "alternate_timeline_description": "What would happen with alternate choice (25-35 words)"
            }}
        ],
        
        "sequel_seeds": [
            {{
                "event_id": "evt_id",
                "unresolved_consequence": "What this sets up for a sequel (20-30 words)",
                "sequel_potential": "Rate 1-10"
            }}
        ]
    }}
    
    IMPORTANT:
    - Only create event nodes for DISCRETE, SPECIFIC actions/occurrences
    - Link events only when there's CLEAR, EXPLICIT causality
    - If you cannot determine causality from the data, leave arrays empty
    - Criticality score = how essential to reaching the story's ending
    - Flexibility score = how much this could change without breaking causality
    """
    
    try:
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        causal_data = clean_json_str(response.choices[0].message.content)
        
        # Validate and enrich the graph
        validated_graph = _validate_causal_graph(causal_data)
        
        # Calculate graph metrics
        graph_metrics = _calculate_graph_metrics(validated_graph)
        
        logging.info(f"✅ Causal graph built: {len(validated_graph.get('events', []))} events")
        
        return {
            "causal_graph": validated_graph,
            "graph_metrics": graph_metrics
        }
        
    except Exception as e:
        logging.error(f"❌ Causal graph extraction failed: {e}")
        return {
            "causal_graph": {"error": str(e)},
            "graph_metrics": {}
        }


def _build_causal_context(scene_events_data, global_context):
    """
    Builds a concise context summary for the LLM.
    Includes scene summaries + character states + relationship dynamics.
    """
    context = "=== ALL SCENES ===\n"
    
    for scene in scene_events_data:
        context += f"\n[{scene['id']}] {scene['chapter']} - Scene {scene['scene_index']}\n"
        context += f"Summary: {scene['summary']}\n"
        context += f"Location: {scene['location']}\n"
        context += f"Characters: {', '.join(scene['characters'])}\n"
        context += f"Tension: {scene['tension']}/10\n"
        context += f"Type: {scene['scene_type']}\n"
    
    # Add character context for understanding motivations
    context += "\n\n=== CHARACTER REGISTRY ===\n"
    for char_id, char_info in global_context.get("known_characters", {}).items():
        context += f"\n{char_id}: {char_info['primary_name']}\n"
        context += f"First appears: {char_info['first_appearance']}\n"
        if char_info.get("appearances"):
            last = char_info["appearances"][-1]
            context += f"Last state: {last.get('emotional_state', 'Unknown')}\n"
    
    # Add relationship dynamics for understanding conflicts
    context += "\n\n=== KEY RELATIONSHIPS ===\n"
    for rel in global_context.get("known_relationships", []):
        context += f"\n{rel['from']} → {rel['to']}: {rel['type']}\n"
        context += f"Dynamic: {rel['dynamic']}\n"
        if rel.get("evolution"):
            context += f"Evolution: {len(rel['evolution'])} changes\n"
    
    return context


def _validate_causal_graph(causal_data):
    """
    Validates event IDs and ensures all references exist.
    Fixes common issues like dangling references.
    """
    events = causal_data.get("events", [])
    event_ids = {evt["id"] for evt in events}
    
    # Validate all event references
    for event in events:
        # Check caused_by references
        event["caused_by"] = [
            cb for cb in event.get("caused_by", [])
            if cb.get("event_id") in event_ids
        ]
        
        # Check causes references
        event["causes"] = [
            c for c in event.get("causes", [])
            if c.get("event_id") in event_ids
        ]
        
        # Check required_for references
        event["required_for"] = [
            rf for rf in event.get("required_for", [])
            if rf.get("event_id") in event_ids
        ]
    
    # Validate critical_path
    if "critical_path" in causal_data:
        causal_data["critical_path"] = [
            cp for cp in causal_data["critical_path"]
            if cp.get("event_id") in event_ids
        ]
    
    # Validate flexible_events
    if "flexible_events" in causal_data:
        causal_data["flexible_events"] = [
            fe for fe in causal_data["flexible_events"]
            if fe.get("event_id") in event_ids
        ]
    
    # Validate causal_chains
    if "causal_chains" in causal_data:
        for chain in causal_data["causal_chains"]:
            chain["event_sequence"] = [
                evt_id for evt_id in chain.get("event_sequence", [])
                if evt_id in event_ids
            ]
    
    return causal_data


def _calculate_graph_metrics(validated_graph):
    """
    Calculates useful metrics about the causal graph structure.
    """
    events = validated_graph.get("events", [])
    
    if not events:
        return {}
    
    # Count causal relationships
    total_prerequisites = sum(len(evt.get("caused_by", [])) for evt in events)
    total_consequences = sum(len(evt.get("causes", [])) for evt in events)
    total_prevents = sum(len(evt.get("prevents", [])) for evt in events)
    total_requirements = sum(len(evt.get("required_for", [])) for evt in events)
    
    # Find most connected events (bottlenecks)
    event_connections = []
    for evt in events:
        connection_count = (
            len(evt.get("caused_by", [])) +
            len(evt.get("causes", [])) +
            len(evt.get("required_for", []))
        )
        event_connections.append({
            "event_id": evt["id"],
            "description": evt["description"],
            "connections": connection_count
        })
    
    event_connections.sort(key=lambda x: x["connections"], reverse=True)
    # Calculate story impact distribution
    impacts = []
    for evt in events:
        impact = evt.get("story_impact", 5)
        try:
            impacts.append(int(impact) if isinstance(impact, str) else impact)
        except (ValueError, TypeError):
            impacts.append(5)
    avg_impact = sum(impacts) / len(impacts) if impacts else 0

    # Calculate reversibility distribution
    reversibilities = []
    for evt in events:
        rev = evt.get("reversibility", 5)
        try:
            reversibilities.append(int(rev) if isinstance(rev, str) else rev)
        except (ValueError, TypeError):
            reversibilities.append(5)
    avg_reversibility = sum(reversibilities) / len(reversibilities) if reversibilities else 0

    return {
        "total_events": len(events),
        "total_causal_links": total_prerequisites + total_consequences,
        "total_prerequisites": total_prerequisites,
        "total_consequences": total_consequences,
        "total_blocked_alternatives": total_prevents,
        "total_requirements": total_requirements,
        "average_connections_per_event": round(
            (total_prerequisites + total_consequences) / len(events), 2
        ) if events else 0,
        "most_connected_events": event_connections[:10],
        "average_story_impact": round(avg_impact, 2),
        "average_reversibility": round(avg_reversibility, 2),
        "critical_path_length": len(validated_graph.get("critical_path", [])),
        "flexible_events_count": len(validated_graph.get("flexible_events", [])),
        "causal_chains_count": len(validated_graph.get("causal_chains", [])),
        "divergence_points_count": len(validated_graph.get("divergence_points", [])),
        "sequel_seeds_count": len(validated_graph.get("sequel_seeds", []))
    }


def process_chapter(chapter, context, global_scene_counter):
    """
    Process a single chapter with full context from all previous chapters.
    Returns: (chapter_analysis, updated_global_scene_counter)
    """
    chapter_name = chapter["chapter"]
    logging.info(f"\n{'='*70}")
    logging.info(f"📚 PROCESSING: {chapter_name}")
    logging.info(f"{'='*70}")
    
    # Split into scenes
    scenes = smart_split_scenes(chapter["content"], chapter_name)
    
    if not scenes:
        logging.error(f"❌ Failed to split scenes in {chapter_name}")
        return None, global_scene_counter
    
    # Analyze each scene
    chapter_scenes_analysis = []
    
    for scene in scenes:
        local_scene_idx = scene["scene_index"]
        global_scene_counter += 1
        
        logging.info(f"🔍 Analyzing {chapter_name} - Scene {local_scene_idx} (Global Scene #{global_scene_counter})...")
        logging.info(f"   📊 Context: {len(context['known_characters'])} chars, {len(context['known_relationships'])} rels, {len(context['previous_summaries'])} scenes")
        
        # Style Analysis
        style_data = analyze_style_neuro_symbolic(scene["content"])
        
        # Entity Analysis (with FULL book context)
        entities_data_raw = analyze_scene_entities(
            scene["content"], 
            local_scene_idx,
            chapter_name,
            context
        )
        
        # Parse outputs
        parsed_soft_metrics = clean_json_str(style_data['soft_metrics'])
        parsed_entities = clean_json_str(entities_data_raw)
        
        # Build scene analysis
        scene_analysis = {
            "scene_index": local_scene_idx,
            "global_scene_index": global_scene_counter,
            "chapter": chapter_name,
            "location": parsed_entities.get("location", "Unknown"),
            "scene_summary": parsed_entities.get("scene_summary", ""),
            "tension_level": parsed_entities.get("tension_level", 5),
            "scene_type": parsed_soft_metrics.get("scene_type", "UNKNOWN"),
            "full_text": scene["content"],
            "style_metrics": style_data['hard_metrics'],
            "style_qualitative": parsed_soft_metrics,
            "characters": parsed_entities.get("characters", []),
            "relationships": parsed_entities.get("relationships", [])
        }
        
        chapter_scenes_analysis.append(scene_analysis)
        
        # Update context for next scene
        update_context(context, scene_analysis, chapter_name)
        
        logging.info(f"✅ Scene {local_scene_idx}: {len(parsed_entities.get('characters', []))} chars, {len(parsed_entities.get('relationships', []))} rels")
    
    # Generate chapter summary
    chapter_summary = generate_chapter_summary(chapter_scenes_analysis)
    
    # Calculate chapter metrics
    total_words = sum([s["style_metrics"]["total_words"] for s in chapter_scenes_analysis])
    total_asl = sum([s["style_metrics"]["avg_sentence_length"] for s in chapter_scenes_analysis])
    avg_asl = total_asl / len(chapter_scenes_analysis) if chapter_scenes_analysis else 0
    
    adverb_percentages = []
    for s in chapter_scenes_analysis:
        adv_str = s["style_metrics"]["adverb_percentage"].replace("%", "")
        try:
            adverb_percentages.append(float(adv_str))
        except:
            pass
    avg_adverb = sum(adverb_percentages) / len(adverb_percentages) if adverb_percentages else 0
    
    # Scene type distribution
    scene_types = [s.get("scene_type", "UNKNOWN") for s in chapter_scenes_analysis]
    scene_type_counts = Counter(scene_types)
    total_scenes_count = len(chapter_scenes_analysis)
    scene_type_distribution = {
        scene_type: {
            "count": count,
            "percentage": round((count / total_scenes_count) * 100, 2)
        }
        for scene_type, count in scene_type_counts.items()
    }
    
    # POV distribution
    povs = [s.get("style_qualitative", {}).get("pov", "Unknown") for s in chapter_scenes_analysis if s.get("style_qualitative", {}).get("pov")]
    pov_counts = Counter(povs)
    pov_distribution = {
        pov: {
            "count": count,
            "percentage": round((count / len(povs)) * 100, 2) if povs else 0
        }
        for pov, count in pov_counts.items()
    }
    
    chapter_analysis = {
        "chapter_name": chapter_name,
        "chapter_summary": chapter_summary,
        "total_scenes": len(chapter_scenes_analysis),
        "chapter_metrics": {
            "total_words": total_words,
            "avg_sentence_length": round(avg_asl, 2),
            "avg_adverb_percentage": f"{round(avg_adverb, 2)}%",
            "scene_type_distribution": scene_type_distribution,
            "pov_distribution": pov_distribution
        },
        "scenes": chapter_scenes_analysis
    }
    
    return chapter_analysis, global_scene_counter


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    if not INPUT_FILE.exists():
        logging.error(f"❌ File not found: {INPUT_FILE}")
        return

    logging.info("📖 Loading all chapters...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    logging.info(f"✅ Loaded {len(chapters)} chapters")

    # Initialize global context (persists across ALL chapters)
    global_context = {
        "known_characters": {},
        "known_relationships": [],
        "known_locations": [],
        "previous_summaries": []
    }
    
    # Process all chapters
    all_chapters_analysis = []
    global_scene_counter = 0
    
    for chapter in chapters:
        chapter_analysis, global_scene_counter = process_chapter(
            chapter, 
            global_context, 
            global_scene_counter
        )
        
        if chapter_analysis:
            all_chapters_analysis.append(chapter_analysis)
    
    # Generate comprehensive character profiles
    character_profiles = generate_character_profiles(global_context)
    
    # Generate book summary
    chapter_summaries_list = [
        {"chapter_name": chap["chapter_name"], "summary": chap["chapter_summary"]}
        for chap in all_chapters_analysis
    ]
    book_summary = generate_book_summary(chapter_summaries_list)
    
    # Generate tension profile
    tension_profile = generate_tension_profile(all_chapters_analysis)
    
    # Generate Author Bible
    author_bible = generate_author_bible(all_chapters_analysis)
    
    # ========================================================================
    # NEW: Extract causal graph (POST-PROCESSING)
    # ========================================================================
    causal_graph_data = extract_causal_graph(all_chapters_analysis, global_context)
    # ========================================================================
    
    # Calculate overall statistics
    total_scenes = sum([chap["total_scenes"] for chap in all_chapters_analysis])
    total_words = sum([chap["chapter_metrics"]["total_words"] for chap in all_chapters_analysis])
    avg_scenes_per_chapter = total_scenes / len(all_chapters_analysis) if all_chapters_analysis else 0
    
    final_report = {
        "metadata": {
            "book_title": "The Cruel Prince - Holly Black",
            "total_chapters": len(all_chapters_analysis),
            "total_scenes": total_scenes,
            "average_scenes_per_chapter": round(avg_scenes_per_chapter, 2),
            "total_words": total_words,
            "total_unique_characters": len(global_context["known_characters"]),
            "total_unique_locations": len(global_context["known_locations"]),
            "total_relationships": len(global_context["known_relationships"]),
            "total_causal_events": causal_graph_data.get("graph_metrics", {}).get("total_events", 0),  # NEW
            "splitting_method": "LLM-Semantic-Detection",
            "analysis_note": "All analysis based exclusively on text content with cumulative context"
        },
        "book_summary": book_summary,
        "author_bible": author_bible,
        "tension_profile": tension_profile,
        "character_profiles": character_profiles,
        "relationship_registry": global_context["known_relationships"],
        "location_registry": global_context["known_locations"],
        
        # NEW: Add causal graph to report
        "causal_graph": causal_graph_data["causal_graph"],
        "graph_metrics": causal_graph_data["graph_metrics"],
        
        "chapters": all_chapters_analysis
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    # Enhanced output reporting
    print("\n" + "=" * 70)
    print(f"✅ FULL BOOK ANALYSIS COMPLETE: {OUTPUT_FILE}")
    print(f"   📚 Total Chapters: {len(all_chapters_analysis)}")
    print(f"   📊 Total Scenes: {total_scenes}")
    print(f"   📈 Avg Scenes/Chapter: {avg_scenes_per_chapter:.2f}")
    print(f"   📝 Total Words: {total_words:,}")
    print(f"   👥 Unique Characters: {len(character_profiles)}")
    print(f"   🔗 Total Relationships: {len(global_context['known_relationships'])}")
    print(f"   📍 Unique Locations: {len(global_context['known_locations'])}")
    
    # NEW: Causal graph metrics
    graph_metrics = causal_graph_data.get("graph_metrics", {})
    if graph_metrics:
        print(f"\n🔗 CAUSAL GRAPH METRICS:")
        print(f"   📍 Total Events: {graph_metrics.get('total_events', 0)}")
        print(f"   🔀 Causal Links: {graph_metrics.get('total_causal_links', 0)}")
        print(f"   🎯 Critical Path Length: {graph_metrics.get('critical_path_length', 0)}")
        print(f"   🌿 Flexible Events: {graph_metrics.get('flexible_events_count', 0)}")
        print(f"   🔱 Divergence Points: {graph_metrics.get('divergence_points_count', 0)}")
        print(f"   📖 Sequel Seeds: {graph_metrics.get('sequel_seeds_count', 0)}")
        print(f"   📊 Avg Connections/Event: {graph_metrics.get('average_connections_per_event', 0)}")
    
    print(f"\n   ⚡ Avg Tension Level: {tension_profile.get('average_tension', 'N/A')}")
    print(f"   📖 Sequel Potential: {tension_profile.get('sequel_potential_rating', 'N/A')}/10")
    print("\n📖 BOOK SUMMARY:")
    print(f"   {book_summary}")
    print("\n🎭 AUTHOR BIBLE HIGHLIGHTS:")
    if "ai_synthesis" in author_bible and not author_bible["ai_synthesis"].get("error"):
        print(f"   Prose: {author_bible['ai_synthesis'].get('prose_signature', 'N/A')}")
        print(f"   Voice: {author_bible['ai_synthesis'].get('narrative_voice', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
