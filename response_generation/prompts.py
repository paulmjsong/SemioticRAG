# ---------------- handle_query.py ----------------
IMG2GRAPH_PROMPT = """
You are an expert in image understanding and knowledge graph construction.
Analyze the given image and output only valid JSON that represents the detected entities and relationships as a knowledge graph.
- Each entity must have a unique "type" and "name".
- Each relationship must specify "source", "target", and "relation", where "source" and "target" reference entity "id" values.
- Include a top-level "entities" array and a "relationships" array.
- Do not include any text outside of the JSON block.

Return format example:
{
  "entities": [
    { "type": "Animal", "name": "Tiger" },
    { "type": "Animal", "name": "Rabbit" }
  ],
  "relations": [
    { "type": "offers_pipe_to", "source": "Rabbit", "target": "Tiger" }
  ]
}
"""

IMG2TEXT_PROMPT = """
You are an expert visual analyst trained to describe the key subjects, objects, and features depicted in artworks and paintings.
Your goal is to generate a concise but information-rich caption that enables accurate semantic retrieval of related context (e.g., art history, symbolism, or artist background) in a RAG system.

Instructions:
Given an input image of a painting:
1. Identify the primary subjects (people, animals, landscapes, buildings, objects, etc.).
2. Describe relevant visual features that provide contextual clues (e.g., setting, posture, clothing, activity, symbols, colors, or props).
3. Avoid subjective interpretation (e.g., “beautiful,” “sad,” “mysterious”) or stylistic commentary (e.g., “Impressionist style”).
4. Focus on factual, observable content only.
5. Write the caption in 1-3 sentences or under 50 words.

Example 1:
Input: Painting of a woman sitting near a window with a cat on her lap.
Output:
A woman in a long dress sits beside a window holding a cat on her lap, with light entering from the left.


Example 2:
Input: Painting showing several soldiers crossing a river with a flag.
Output:
A group of soldiers cross a river in a boat, one raising a flag while others row through icy waters.
"""

RETRIEVAL_CYPHER = """
WITH node, score
MATCH p = __PATTERN__
WITH node, score AS seedScore, p, nbr,
     toFloat($lambda) AS lambda,
     coalesce($q_embed, []) AS qv

// Text similarity for endpoint nbr
WITH node, seedScore, p, nbr, lambda, qv,
     // dot(qv, nbr.embedding), ||qv||, ||nbr.embedding||
     reduce(d=0.0, i IN range(0, size(qv)-1) | d + coalesce(qv[i],0.0) * coalesce(nbr.embedding[i],0.0)) AS dotN,
     sqrt(reduce(s=0.0, i IN range(0, size(qv)-1) | s + coalesce(qv[i],0.0)^2)) AS normQ,
     sqrt(reduce(s=0.0, i IN range(0, size(coalesce(nbr.embedding,[]))-1) | s + coalesce(nbr.embedding[i],0.0)^2)) AS normN
WITH node, seedScore, p, nbr, lambda,
     CASE WHEN normQ = 0 OR normN = 0 THEN 0.0 ELSE dotN / (normQ * normN) END AS textSim

// Per-path node rank
WITH node, p, nbr,
     lambda * seedScore + (1 - lambda) * toFloat(textSim) AS nodeRank
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
     coalesce( head([m IN nodeRankList WHERE m.nid = elementId(n) | m.rank]), 0.0 ) AS bestNodeRank
WITH collect({
       id: elementId(n),
       labels: labels(n),
       name: coalesce(n.name, "(unnamed)"),
       description: coalesce(n.description, ""),
       rank: CASE WHEN n = node AND bestNodeRank < 1.0 THEN 1.0 ELSE bestNodeRank END
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

GENERATION_PROMPT = (
  # "You are an expert in Korean art history. "
  # "Rely ONLY on the provided subgraph facts. "
  # "Cite entities by their names as shown in the node lines. "
  # "If the graph does not contain enough information, plainly say so."
  "You are an expert in Korean art history. "
  "Use all relevant facts from the provided context to craft detailed, well-informed answers. "
  "Cite specific entities and cultural elements where appropriate. "
  "Focus on conveying both stylistic and cultural significance. "
  "If critical information is missing, acknowledge that explicitly. Do not add information beyond the provided context."
)