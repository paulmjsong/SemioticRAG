# ---------------- extract_entities.py ----------------
EXTRACTION_PROMPT = """
You are an expert semiotician using Roland Barthes' model (Connotation → Myth). Extract entities (Forms, Concepts, Myths) and their relations from the passage.
Output must be:
- English only
- Entities and relations only (no explanations, no headers, no summaries)
- Valid JSON: a single JSON object with two arrays: "entities" and "relations".

# Canonicalization & Deduplication Rules
1. One concept per idea: Merge near-duplicates (e.g., "Rabbit" and "Rabbit in Folklore" → "Rabbit").
2. Naming:
  - Use Title Case, ASCII, singular nouns where possible (e.g., "Tiger", "Oppression", "Cunning").
  - Use the most general canonical label unless a qualifier is essential (prefer "Tiger" over "Smoking Tiger" unless smoking is semantically crucial).
3. English only: Translate all labels to English.
4. Uniqueness: Each entity must have a unique "name". Use "aliases" for other surface forms from the passage.

# Allowed Entity Types
Each entity must have:
- "name" — canonical English label (Title Case)
- "type" — one of:
  - "Form" — material signifier in the passage (e.g., Tiger, Rabbit, Smoking Tiger motif)
  - "Concept" — immediate connoted meaning (e.g., Power, Meekness, Corruption, Resistance)
  - "Myth" — overarching ideological signified (e.g., Dominance of Oppressors, Survival of the Weak through Cunning)
- "aliases" — optional array of English surface forms/variants from the passage

# Allowed Relation Types
Only these relation types are permitted:
- "Connotes": Form → Concept
- "Generates_Myth": Combined(Concepts) → Myth

# JSON Schema
{
  "entities": [
    {
      "type": "Form|Concept|Myth",
      "name": "string",           // canonical English label, Title Case, unique
      "aliases": ["string"]       // optional; English variants/surface forms
    }
  ],
  "relations": [
    {
      "type": "Connotes",
      "source": "string",         // Form.name (must match an entity.name)
      "target": "string"          // Concept.name (must match an entity.name)
    },
    {
      "type": "Generates_Myth",
      "source_concepts": [
        "string"                  // Concept.name values; 2+ items
      ],
      "target": "string"          // Myth.name (must match an entity.name)
    }
  ]
}

# Processing Rules
- Build the "entities" list first, applying deduplication and English canonicalization.
- For each canonical entity:
  - Put its canonical label in "name".
  - Collect other surface forms into "aliases" (English only, translated if needed).
- No IDs: do not output any "id" field.
- For each relation:
  - "source", "target", and "source_concepts" must exactly match the "name" of some entity in "entities".
  - For "Generates_Myth", include only the minimal set of concepts needed to generate that myth.
  - Sort "source_concepts" alphabetically for stable, repeatable output.
- Do not output anything other than the JSON object (no prose, no comments).

# Passage
[Korean text here]

Produce the JSON now.
"""

# ---------------- construct_database.py ----------------
# None

# ---------------- handle_query.py ----------------
# CREATING G1
IMG2GRAPH_PROMPT = """
You are an expert in image understanding and knowledge graph construction.
Analyze the given image and output only valid JSON that represents the detected entities and relationships as a knowledge graph.
- Each entity must have a unique "id", "name", and "type".
- Each relationship must specify "source", "target", and "relation", where "source" and "target" reference entity "id" values.
- Include a top-level "entities" array and a "relationships" array.
- Do not include any text outside of the JSON block.

Return format example:
{
  "entities": [
    { "id": "e1", "name": "Tiger", "type": "Animal" },
    { "id": "e2", "name": "Rabbit", "type": "Animal" }
  ],
  "relationships": [
    { "source": "e2", "target": "e1", "relation": "offers_pipe_to" }
  ]
}
"""

# RETRIEVING G2
RETRIEVAL_CYPHER = """
WITH node, score
MATCH p = (node)__PATTERN__(nbr)
WITH node, score AS seedScore, p, nbr, length(p) AS hops

// Hop decay
WITH node, seedScore, p, nbr, hops,
     CASE
       WHEN $hop_decay IS NULL OR $hop_decay <= 0 THEN 1.0
       ELSE exp( (hops - 1) * log( toFloat($hop_decay) ) )
     END AS hop_w,
     toFloat($lambda) AS lambda

// Text similarity for endpoint nbr
WITH node, seedScore, p, nbr, hop_w, lambda, coalesce($q_embed, []) AS qv
WITH node, seedScore, p, nbr, hop_w, lambda, qv,
     // dot(qv, nbr.embedding), ||qv||, ||nbr.embedding||
     reduce(d=0.0, i IN range(0, size(qv)-1) | d + coalesce(qv[i],0.0) * coalesce(nbr.embedding[i],0.0)) AS dotN,
     sqrt(reduce(s=0.0, i IN range(0, size(qv)-1) | s + coalesce(qv[i],0.0)^2)) AS normQ,
     sqrt(reduce(s=0.0, i IN range(0, size(coalesce(nbr.embedding,[]))-1) | s + coalesce(nbr.embedding[i],0.0)^2)) AS normN
WITH node, seedScore, p, nbr, hop_w, lambda,
     CASE WHEN normQ = 0 OR normN = 0 THEN 0.0 ELSE dotN / (normQ * normN) END AS textSim

// Per-path node rank
WITH node, p, nbr,
     hop_w * ( lambda * seedScore + (1 - lambda) * toFloat(textSim) ) AS nodeRank
ORDER BY nodeRank DESC

// Keep top paths per seed
WITH node, collect({ p: p, rank: nodeRank })[..$per_seed_limit] AS top

// Aggregate best node scores
CALL (top) {
  WITH top
  UNWIND top AS k1
  UNWIND nodes(k1.p) AS n1
  WITH elementId(n1) AS nid, k1.rank AS kr1
  WITH nid, max(kr1) AS bestNodeRank
  RETURN collect({nid: nid, rank: bestNodeRank}) AS nodeRankList
}

// Aggregate best relationship scores
WITH node, top, nodeRankList, coalesce($q_embed, []) AS qv
CALL (top, qv) {
  WITH top, qv
  UNWIND top AS k2
  UNWIND relationships(k2.p) AS r2
  WITH r2, startNode(r2) AS s, endNode(r2) AS e, qv
  WITH r2, s, e, qv,
       reduce(sd=0.0, i IN range(0, size(qv)-1) | sd + coalesce(qv[i],0.0) * coalesce(s.embedding[i],0.0)) AS dotS,
       sqrt(reduce(ss=0.0, i IN range(0, size(qv)-1) | ss + coalesce(qv[i],0.0)^2)) AS normQ,
       sqrt(reduce(se=0.0, i IN range(0, size(coalesce(s.embedding,[]))-1) | se + coalesce(s.embedding[i],0.0)^2)) AS normS,
       reduce(ed=0.0, i IN range(0, size(qv)-1) | ed + coalesce(qv[i],0.0) * coalesce(e.embedding[i],0.0)) AS dotE,
       sqrt(reduce(ee=0.0, i IN range(0, size(coalesce(e.embedding,[]))-1) | ee + coalesce(e.embedding[i],0.0)^2)) AS normE
  WITH r2,
       CASE WHEN normQ = 0 OR normS = 0 THEN 0.0 ELSE dotS / (normQ * normS) END AS simS,
       CASE WHEN normQ = 0 OR normE = 0 THEN 0.0 ELSE dotE / (normQ * normE) END AS simE
  WITH elementId(r2) AS rid, (simS + simE)/2.0 AS simAvg
  RETURN collect({rid: rid, rank: toFloat(simAvg)}) AS relRankList
}

// Materialize unique nodes/rels from top paths
UNWIND top AS k3
UNWIND nodes(k3.p) AS n
WITH node, nodeRankList, relRankList, collect(DISTINCT n) AS allNodes, top

UNWIND top AS k4
UNWIND relationships(k4.p) AS r
WITH node, nodeRankList, relRankList, allNodes, collect(DISTINCT r) AS allRels

// Build nodes with best rank lookup
UNWIND allNodes AS n
WITH node, nodeRankList, relRankList, allRels, n,
     COUNT { (n)--() } AS deg_n,
     coalesce( head([m IN nodeRankList WHERE m.nid = elementId(n) | m.rank]), 0.0 ) AS bestNodeRank
WITH
     collect({
       id: elementId(n),
       labels: labels(n),
       name: coalesce(n.name, "(unnamed)"),
       description: coalesce(n.description, ""),
       degree: deg_n,
       rank: CASE WHEN n = node AND bestNodeRank < 1.0 THEN 1.0 ELSE bestNodeRank END,
       isSeed: n = node
     }) AS nodes, allRels, relRankList

// Build relationships with best rank lookup
UNWIND allRels AS r
WITH nodes, relRankList, r, startNode(r) AS s, endNode(r) AS e
WITH nodes,
     collect({
       id: elementId(r),
       type: type(r),
       start: elementId(s),
       end: elementId(e),
       description: coalesce(r.description, ""),
       rank: coalesce( head([m IN relRankList WHERE m.rid = elementId(r) | m.rank]), 0.0 )
     }) AS rels

// Final pruning and ordering
UNWIND nodes AS n
WITH n, rels
ORDER BY n.rank DESC
WITH collect(n) AS nodes, rels

UNWIND rels AS r
WITH nodes, r
ORDER BY r.rank DESC
RETURN nodes, collect(r) AS rels
"""

# FINAL ANSWER
SYSTEM_PROMPT = (
  "You are an expert in Korean art history. "
  "Rely ONLY on the provided subgraph facts. "
  "Cite entities by their names as shown in the node lines. "
  "If the graph does not contain enough information, plainly say so."
)