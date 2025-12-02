# ---------------- handle_query.py ----------------
CAP_SYSTEM_PROMPT = """
You are an expert visual analyst trained to identify concrete subjects, objects, and features depicted in artworks. Your task is to output a clean, retrieval-ready query consisting only of the entities visible in the image.

Instructions:
1. Include only factual entities that are visibly depicted (people, animals, objects, environmental features, symbols).
2. Use short noun phrases; no full sentences.
3. Avoid stylistic terms, emotions, speculation, or interpretation.
4. Output a single line, with entities separated by commas.
5. Do not include labels like “Entities:” — output only the final query text.

Output format:
entity1, entity2, entity3, …
"""

CAP_USER_PROMPT = """
Identify the entities visible in the painting and output them as a single comma-separated query.
"""

RETRIEVAL_CYPHER = """
WITH node AS srcNode, score AS srcScore
MATCH p = (srcNode:Form)-[*1..]->(tgtNode:Myth)

WITH srcNode, srcScore, p,
     length(p) AS pathLen
     (srcScore / pathLen) AS pathRank

ORDER BY pathRank DESC
WITH collect(p)[..$per_seed_limit] AS topPaths

// Materialize unique nodes/rels from top paths
UNWIND topPaths AS tp1
UNWIND nodes(tp1) AS n
WITH topPaths, collect(DISTINCT n) AS nodes

UNWIND topPaths AS tp2
UNWIND relationships(tp2) AS r
WITH nodes, collect(DISTINCT r) AS rels

RETURN [
    n IN nodes | {
        id: elementId(n),
        labels: labels(n),
        name: coalesce(n.name, "(unnamed)"),
        description: coalesce(n.description, "")
    }
] AS nodes, [
    r IN rels | {
        type: type(r),
        start: elementId(startNode(r)),
        end: elementId(endNode(r)),
        description: coalesce(r.description, "")
    }
] AS rels
"""

GEN_SYSTEM_PROMPT = """
You are an expert in Korean art history and cultural semiotics.

Using only the information contained in the provided context (including the retrieved graph elements and the image), write a cohesive explanatory paragraph that interprets the artwork. Integrate the relevant entities, symbolic relationships, and cultural concepts into a natural narrative rather than listing them individually.

Your explanation should:
- Synthesize all necessary information into flowing prose.
- Describe how the depicted forms relate to their associated concepts or myths.
- Emphasize stylistic, historical, and cultural significance as inferred from the context.
- Cite or reference entities naturally within sentences, not as separate lists.
- Avoid adding facts not supported by the provided context.
- If information is missing or uncertain, acknowledge it explicitly within the paragraph.

Your goal is to produce a culturally grounded, well-reasoned interpretation that reads like an art-historical commentary, not a symbolic glossary.
"""

GEN_USER_PROMPT = """
Context:
{context}

Query:
{query}

Answer:
"""