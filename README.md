# Character Knowledge Graph Toolkit (Python)

Concrete, runnable pieces to build a character-focused knowledge graph in Postgres and a small ingestion pipeline that turns documents into nodes/edges.

## What's inside
- `kg/schema.py` – SQLAlchemy models for `kg_nodes`, `kg_edges`, `source_segments` (Postgres-ready, UUID keys, JSONB/ARRAY columns).
- `kg/pipeline.py` – segmentation helper, extractor protocol, LLM + mock extractors, and a `GraphBuilder` that persists nodes/edges/segments.
- `kg/types.py` – Pydantic models for extraction payloads.
- `requirements.txt` – core dependencies (SQLAlchemy, pydantic, psycopg2, openai).
- Neo4j/Memgraph push helper: `kg/neo4j_memgraph.py` (`CypherGraphBuilder`).
- Deep dive: `docs/PIPELINE.md` (full flow, LLM contract, and retrieval patterns).

## Quickstart
1. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Set `DATABASE_URL` (e.g., `postgresql+psycopg2://user:pass@localhost:5432/kg`).
3. Run a small ingestion with the mock extractor:
   ```python
   from kg.schema import create_engine_and_session
   from kg.pipeline import GraphBuilder, MockExtractor

   engine, SessionLocal = create_engine_and_session()
   session = SessionLocal()

   text = "He wrote letters about duty.\n\nLater he opposed tyranny."
   builder = GraphBuilder(session=session, extractor=MockExtractor())
   builder.build_from_text(text, work_name="Collected Letters", character="Marcus")
   ```
   This creates `Work` and `Character` nodes, stores segments, and links them.

## Using an LLM extractor
`kg/pipeline.LLMExtractor` expects an OpenAI-compatible client (`openai.OpenAI()`) and a model that supports JSON mode.
```python
from openai import OpenAI
from kg.pipeline import GraphBuilder, LLMExtractor
from kg.schema import create_engine_and_session

client = OpenAI()  # requires OPENAI_API_KEY
relations = ["RELATES_TO","BELIEVES_IN","OPPOSES","INFLUENCED_BY","REFERENCES","CONTRADICTS"]
extractor = LLMExtractor(client=client, model="gpt-4o-mini", relations=relations)

engine, SessionLocal = create_engine_and_session()
session = SessionLocal()
builder = GraphBuilder(session, extractor)
builder.build_from_text(open("doc.txt").read(), work_name="Book I", character="Character Name")
```

LLM response format expected:
```json
{
  "nodes": [
    {"name":"Character Name","type":"Character","summary":"...","alias_names":["N/A"],"meta":{},"source_id":"book-i-0-0"}
  ],
  "edges": [
    {"from_name":"Character Name","to_name":"Book I","relation":"REFERENCES","description":"Discusses work","source_id":"book-i-0-0","confidence":0.9}
  ]
}
```

## Using Neo4j or Memgraph
- Install the driver: `pip install neo4j`.
- Start Neo4j (default 7687) or Memgraph (`mgconsole` → `CREATE USER neo4j IDENTIFIED BY 'pass'` if needed).
- Example:
```python
from neo4j import GraphDatabase
from kg.neo4j_memgraph import CypherGraphBuilder
from kg.pipeline import LLMExtractor, MockExtractor

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "pass"))
extractor = MockExtractor()  # or LLMExtractor(...)
builder = CypherGraphBuilder(driver, extractor)

text = "He wrote letters about duty.\n\nLater he opposed tyranny."
builder.build_from_text(text, work_name="Collected Letters", character="Marcus")
```
- What it writes:
  - Nodes labeled by `type` (Character, Work, Event, etc.) with properties `name`, `alias_names`, `summary`, `meta`, `source_ids`.
  - `SourceSegment` nodes with `id`, `content`, `location`, `meta`.
  - Relationships:
    - `(:Work)-[:HAS_SEGMENT]->(:SourceSegment)`
    - Extracted edges with relation type uppercased (e.g., `OPPOSES`, `BELIEVES_IN`, `REFERENCES`) between matched node names.
- Relation naming is sanitized to uppercase snake-case; MEMGRAPH accepts the same Cypher.

## Pipeline flow
- **Segment**: `segment_text` splits on paragraph gaps and limits word count, assigns stable `source_segments.id`.
- **Extract**: any `BaseExtractor` (LLM or mock) returns `NodeCandidate` and `EdgeCandidate` lists.
- **Persist**: `GraphBuilder` upserts nodes (merging aliases/meta/summary), stores segments, and dedupes edges by `(from_id, to_id, relation)`.

## Schema recap (matches Postgres DDL)
- `kg_nodes`: `id UUID PK`, `type`, `name`, `alias_names TEXT[]`, `summary`, `meta JSONB`, `source_ids TEXT[]`, timestamps.
- `kg_edges`: `id bigserial PK`, `from_id/to_id UUID FK`, `relation`, `weight`, `description`, `source_ids TEXT[]`, `created_at`.
- `source_segments`: `id TEXT PK`, `work_id UUID FK -> kg_nodes`, `location`, `content`, `meta JSONB`, `created_at`.

## Adapting to your chatbot
- Run extraction when a document uploads; call `GraphBuilder.build_from_text`.
- Keep `source_id` on every node/edge so you can trace facts back to text.
- At answer time, traverse nodes/edges for context + retrieve supporting `source_segments` (or embed them with pgvector later).

## Need more detail?
See `docs/PIPELINE.md` for step-by-step architecture, the LLM JSON contract, and example SQL/Cypher queries.

## Deploying to Render
- A `render.yaml` is included. It provisions:
  - A Postgres instance (`character-kg-db`) and a web service running FastAPI (`uvicorn app.main:app`).
  - `DATABASE_URL` is wired from the managed Postgres. `LLM_MODEL` defaults to `gpt-4o-mini`. Set `OPENAI_API_KEY` in the Render dashboard if you want LLM-backed extraction/answers; leave unset to use the mock extractor.
- Deploy steps:
  1) Push to your repo with `render.yaml`.
  2) Create a new “Blueprint” on Render pointing to the repo; Render builds with `pip install -r requirements.txt` and starts with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
  3) After deploy, call `POST /ingest-book` (multipart file upload) and `POST /query` with `prompt` (+ optional `character`). Healthcheck: `GET /health`.
