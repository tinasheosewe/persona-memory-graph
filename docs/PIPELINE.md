# Character KG Pipeline Guide

This doc explains how the toolkit turns raw documents into a character-focused knowledge graph, how the LLM is used, and how to pull facts back out.

## Components
- **Segmentation**: `kg.pipeline.segment_text` splits raw text into paragraph-aware `Segment` objects with stable IDs.
- **Extractor**: any `BaseExtractor` that returns `GraphExtractionResult` (`nodes` + `edges`). Provided:
  - `LLMExtractor` for structured JSON extraction via OpenAI-compatible chat models.
  - `MockExtractor` for local testing.
- **Builders**:
  - `GraphBuilder` (SQLAlchemy) persists to Postgres tables (`kg_nodes`, `kg_edges`, `source_segments`).
  - `CypherGraphBuilder` pushes to Neo4j or Memgraph via Cypher `MERGE`.
- **Schema**: `kg/schema.py` holds SQLAlchemy models that mirror the schema described in the README.
- **Types**: `kg/types.py` defines `Segment`, `NodeCandidate`, `EdgeCandidate`.

## Data model (shared intent across stores)
- Node types: `Character`, `Person`, `Event`, `Concept`, `Organization`, `Work`, `Place`, `Period`, `SourceSegment`, **`Episode`**, **`Principle`**.
- Edge vocab: the fixed list in `kg/types.py` (`RelationLiteral`), for example `RELATES_TO`, `BELIEVES`, `OPPOSES`, `INFLUENCES`, `REFERENCES`, `SUPPORTS`, `CONTRADICTS`, `PART_OF`, `PRECEDES`, `CAUSES`, `EVIDENCED_BY`. A relation outside the list is mapped to `RELATES_TO`, and the extraction result keeps the original name in `meta.original_relation`.
- Partitioning: nodes, edges and Neo4j entities carry a `character_key` (default `default`) so several characters can share one database.
- Source linkage: every node/edge can carry `source_ids` that point to the `SourceSegment` rows/nodes created from the text.

### Episodic / principle modeling
- **Episode** nodes: name is a short label; `meta` may include `context`, `tension`, `response`, `rationale`, `outcome`, `confidence`, `canon_status`. Attach a `source_id` for provenance.
- **Principle** nodes: name/claim of the principle; `meta` may include `scope`, `support` (episode IDs/names or source IDs), `exceptions`, `confidence`. Use edges to link supporting or exception episodes.
- Suggested edges: `EVIDENCED_BY` (Episode → SourceSegment), `DERIVED_FROM` or `SUPPORTS` (Principle → Episode), `EXCEPTION_OF` (Episode → Principle), `APPLIES_TO` (Principle → Episode).

## Pipeline: document → graph (Postgres)
1. **Ingest text**: call `GraphBuilder.build_from_text(text, work_name, character=...)`.
2. **Segment**: `segment_text` packs whole sentences into chunks up to `max_words` (default 120), never splitting a sentence, with `overlap_sentences` (default 1) to carry context between chunks. IDs look like `book-i-0-0`.
3. **Ensure Work node**: `_ensure_work_node` creates/returns the `Work` node for `work_name`.
4. **Store segments**: `_store_segments` writes `source_segments` rows tied to the `Work` node.
5. **Extract**: `extractor.extract(segments, character=...)` returns `NodeCandidate` and `EdgeCandidate` lists.
6. **Upsert nodes**: `_upsert_nodes` merges by `(name, type)`, unioning aliases/source_ids and filling summary/meta.
7. **Upsert edges**: `_upsert_edges` merges by `(from_id, to_id, relation)`, updating description/weight/source_ids.
8. **Commit**: single transaction via the provided SQLAlchemy session.

## Pipeline: document → graph (Neo4j/Memgraph)
Two options:
- **Single extraction, dual write (recommended)**: run `GraphBuilder.build_from_text(..., return_segments=True)` to get `(extraction, segments)`, then call `CypherGraphBuilder.project(segments, extraction, work_name, work_meta)`. This avoids a second LLM call and keeps Postgres canonical.
- **Graph-only write**: call `CypherGraphBuilder.build_from_text(text, work_name, character=...)` to segment, extract, and push directly to the graph.

What `CypherGraphBuilder` writes:
1. `Work` + `SourceSegment` nodes: `MERGE` a `Work` node and `HAS_SEGMENT` relationships to `SourceSegment` nodes (properties: `id`, `content`, `location`, `character_key`).
2. Entity nodes: `MERGE` by `name` and `character_key` with a label matching `type` (e.g., `:Character`, `:Event`). Properties: `type`, `alias_names`, `summary`, `source_ids`.
3. Edges: `MERGE` relationships using sanitized uppercase relation names (e.g., `OPPOSES`, `REFERENCES`), updating `description`, `weight`, `source_ids`.

## LLM extraction contract
`LLMExtractor` sends a prompt with the segments (each prefixed by its id) and a JSON-schema `response_format`, and expects JSON:
```json
{
  "nodes": [
    {"name": "Character Name", "type": "Character", "summary": "...", "alias_names": [], "meta": {}, "source_id": "book-i-0-0"}
  ],
  "edges": [
    {"from_name": "Character Name", "to_name": "Book I", "relation": "REFERENCES", "description": "Discusses work", "source_id": "book-i-0-0", "confidence": 0.9}
  ]
}
```
Tips:
- Provide a controlled relation list to the extractor (`relations=[...]`) to keep outputs consistent. The names must come from `RelationLiteral` in `kg/types.py`; the default is the whole list.
- Include the focus `character` name so the model grounds pronouns and implicit mentions.
- Treat `source_id` as mandatory in your prompting so you can trace every fact to text.

## Retrieval patterns
- **Postgres**:
  - Neighbors of a character: `SELECT e.relation, n.* FROM kg_edges e JOIN kg_nodes n ON e.to_id = n.id JOIN kg_nodes c ON e.from_id = c.id WHERE c.name ILIKE 'Character%'`.
  - Evidence: fetch `source_segments` by `source_ids` attached to nodes/edges.
- **Neo4j/Memgraph**:
  - `MATCH (c:Character {name:$name})-[r]->(n) RETURN r,n LIMIT 100`
  - Evidence via `MATCH (w:Work)-[:HAS_SEGMENT]->(s:SourceSegment) WHERE s.id IN $source_ids RETURN s`.

Use these results to build LLM context: node summaries + labeled edges + supporting segment text.

The FastAPI service in `app/main.py` wraps these patterns: `POST /query` returns a focus node with its edges (one hop), and `POST /llm-walk` starts from the same place and lets the LLM choose which nodes to expand next, fetching their edges and neighbours at each step. Both are described in the README.

## Operational notes
- Postgres: set `DATABASE_URL`, run through `create_engine_and_session()`. Add `pgvector` columns if you want embeddings later.
- Neo4j/Memgraph: supply a `neo4j.Driver` to `CypherGraphBuilder`. Relation names are uppercased/sanitized; node labels derive from `type`.
- Indexes/constraints to add (recommended):
  - Postgres: unique `(type, lower(name))` partial index; GIN on `source_ids` if frequently filtered.
  - Neo4j/Memgraph: `CREATE CONSTRAINT FOR (n:Character) REQUIRE n.name IS UNIQUE;` and similar for other frequent labels; `CREATE CONSTRAINT FOR (s:SourceSegment) REQUIRE s.id IS UNIQUE;`.

## Extending
- Deduping: embed `(name + summary)` and run similarity before creating nodes.
- Enrichment: aggregate `SourceSegment` content per node and resummarize into `summary`.
- Human review: expose candidates before commit, or flag low-confidence edges for approval.
