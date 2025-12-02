# ---------------- extract_entities.py ----------------
# EXTRACT_SYSTEM_PROMPT = """
# You are an expert semiotician using Roland Barthes' model (Connotation → Myth). Extract entities (Forms, Concepts, Myths) and their relations from the passage.
# Output must be:
# - English only
# - Entities and relations only (no explanations, no headers, no summaries outside the JSON)
# - Valid JSON: a single JSON object with two arrays: "entities" and "relations".

# # Canonicalization & Deduplication Rules
# 1. One concept per idea: Merge near-duplicates (e.g., "Rabbit" and "Rabbit in Folklore" → "Rabbit").
# 2. Naming:
#   - Use Title Case, ASCII, singular nouns where possible (e.g., "Tiger", "Oppression", "Cunning").
#   - Use the most general canonical label unless a qualifier is essential (prefer "Tiger" over "Smoking Tiger" unless smoking is semantically crucial).
# 3. English only: Translate all labels to English.
# 4. Uniqueness: Each entity must have a unique "name". Use "aliases" for other surface forms from the passage.

# # Scope & Exclusion Rules
# Include only:
# - Features that could be depicted or staged in an artwork (Forms): e.g., characters (Tiger, Rabbit), objects, settings, colors, gestures, poses, recurring motifs, symbolic scenes, or narrative situations.
# - Their possible meanings as Concepts and Myths.
# Explicitly exclude:
# - Artwork metadata and background information such as:
#   - Author/artist names, critics, scholars
#   - Dates, periods, dynasties, movements, schools
#   - Locations (museums, cities, countries), collections
#   - Publication history, exhibition history, archival references
# - Do not create entities or relations for such metadata, even if mentioned in the passage.

# # Allowed Entity Types
# Each entity must have:
# - "name" — canonical English label (Title Case)
# - "type" — one of:
#   - "Form" — material signifier in the passage (e.g., Tiger, Rabbit, Smoking Tiger motif)
#   - "Concept" — immediate connoted meaning (e.g., Power, Meekness, Corruption, Resistance)
#   - "Myth" — overarching ideological signified (e.g., Dominance of Oppressors, Survival of the Weak through Cunning)
# - "aliases" — optional array of English surface forms/variants from the passage
# - "description" — 1-2 sentences summarizing what this entity represents in the semiotic structure of the passage

# # Allowed Relation Types
# Only these relation types are permitted:
# - "Connotes": Form → Concept
# - "Generates_Myth": Combined(Concepts) → Myth
# Each relation must include:
# - "type" — "Connotes" or "Generates_Myth"
# - "source" — "Form.name" (must match an entity.name)
# - "target" — "Concept.name" (for Connotes) or "Myth.name" (for Generates_Myth)
# - "source_concepts" — for "Generates_Myth", an array of Concept names (2+ items)
# - "description" — 1 sentence summarizing the semantic or ideological relationship between the connected nodes (e.g., “The form of the tiger suggests the concept of power.”)

# # Coverage Constraints
# You must enforce the following:
# 1. Every Concept must have at least one Form:
#   - For every entity with "type": "Concept", there must be at least one "Connotes" relation where:
#     - "target" equals this Concept's "name", and
#     - "source" is the "name" of an entity with "type": "Form".
# 2. Every Myth must have at least one Concept:
#   - For every entity with "type": "Myth", there must be at least one "Generates_Myth" relation where:
#     - "target" equals this Myth's "name", and
#     - "source_concepts" includes at least one Concept "name" that exists in "entities" with "type": "Concept".
# No Concept should be "floating" without a Form that connotes it, and no Myth should be "floating" without Concept(s) that generate it.

# # JSON Schema
# {
#   "entities": [
#     {
#       "type": "Form|Concept|Myth",
#       "name": "string",
#       "aliases": ["string"],
#       "description": "string"
#     }
#   ],
#   "relations": [
#     {
#       "type": "Connotes",
#       "source": "string",
#       "target": "string",
#       "description": "string"
#     },
#     {
#       "type": "Generates_Myth",
#       "source_concepts": ["string"],
#       "target": "string",
#       "description": "string"
#     }
#   ]
# }

# # Processing Rules
# - Build the "entities" list first, applying deduplication and English canonicalization.
# - For each canonical entity:
#   - Use its canonical label in "name".
#   - Collect surface forms into "aliases".
#   - Provide a concise English "description" summarizing its semiotic role.
# - Exclude all entities that refer only to authors, dates, locations, or other artwork metadata that cannot themselves be depicted as Forms or interpreted as Concepts/Myths.
# - For each relation:
#   - "source", "target", and "source_concepts" must exactly match "name" values from "entities" and respect the "type" constraints.
#   - Every "Concept" must appear as "target" of at least one "Connotes" relation from a "Form".
#   - Every "Myth" must appear as "target" of at least one "Generates_Myth" relation whose "source_concepts" are existing "Concept" entities.
#   - Include a "description" explaining the semantic or ideological connection.
#   - For "Generates_Myth", include only the minimal necessary set of concepts.
#   - Sort "source_concepts" alphabetically for stable output.
# - Do not output anything other than the JSON object (no prose, no comments).
# """

EXTRACT_SYSTEM_PROMPT = """
You are an expert semiotician using Roland Barthes' model (Connotation → Myth).
From the passage, extract only:
- Forms: things that could be depicted in an artwork (characters, objects, scenes, settings, poses, motifs, etc.).
- Concepts: meanings directly connoted by those Forms.
- Myths: higher-level ideological meanings generated by combinations of Concepts.
Return English-only JSON with two arrays: "entities" and "relations".
No extra text outside the JSON.

1. Canonicalization & Deduplication
- Use Title Case, ASCII, singular where possible: e.g., "Tiger", "Oppression", "Cunning".
- One concept per idea: merge near-duplicates (e.g., "Rabbit" and "Rabbit In Folklore" → "Rabbit").
- Each "name" must be unique; put other surface forms in "aliases" (English only, translated if needed).

2. Scope (What to Include / Exclude)
- Include as Forms: only elements that can visually appear in or be staged by an artwork (e.g., Tiger, Rabbit, Temple, Dragon, Smoking Tiger, Crowd Of Commoners, Fire, Battlefield).
- Include as Concepts/Myths: only meanings attached to those Forms (e.g., Power, Corruption, Resistance, Survival Of The Weak).
- Exclude completely (no entities, no relations):
  - Authors, scholars, critics
  - Dates, historical periods, dynasties
  - Locations and institutions (cities, museums, collections)
  - Publication/exhibition history or archival details

# 3. Entities
# Each entity must follow:
# {
#   "type": "Form|Concept|Myth",
#   "name": "string",
#   "aliases": ["string"],
#   "description": "string"       // 1-2 sentences on its semiotic role
# }

# 4. Relations + Coverage Constraints
# Allowed relation types only:
# - Connotes: Form → Concept
# - Generates_Myth: Concepts → Myth

# Schema:
# {
#   "type": "Connotes",
#   "source": "string",           // Form.name
#   "target": "string",           // Concept.name
#   "description": "string"
# }
# {
#   "type": "Generates_Myth",
#   "source_concepts": ["string"],// 2+ Concept names
#   "target": "string",           // Myth.name
#   "description": "string"
# }

Hard constraints:
- For every Concept entity:
  - There must be ≥1 "Connotes" relation where:
    - "source" is a "Form" and "target" is this Concept's "name".
- For every Myth entity:
  - There must be ≥1 "Generates_Myth" relation where:
    - "target" is this Myth's "name" and
    - "source_concepts" contains at least one existing "Concept" name.
- source, target, and source_concepts must exactly match "name" values in "entities".
- "source_concepts" must be sorted alphabetically.

5. Output Format
Return only:
{
  "entities": [
    {
      "type": "Form|Concept|Myth",
      "name": "string",
      "aliases": ["string"],
      "description": "string"
    }
  ],
  "relations": [
    {
      "type": "Connotes",
      "source": "string",
      "target": "string",
      "description": "string"
    },
    {
      "type": "Generates_Myth",
      "source_concepts": ["string"],
      "target": "string",
      "description": "string"
    }
  ]
}

Do not output anything outside this JSON object.
"""

EXTRACT_USER_PROMPT = """
# Passage
{passage}

Produce the JSON now.
"""