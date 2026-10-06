# Persona Memory Graph

Part of **Persona**, a memory system for AI characters. This repository is its graph memory: it turns a character's documents into a typed knowledge graph of facts, episodes and principles, stores the graph in PostgreSQL with an optional projection into Neo4j, and serves graph-traversal retrieval through a FastAPI service. The other two parts:

- [persona-chatbot](https://github.com/tinasheosewe/persona-chatbot): the chat backend (RAG, hierarchical memory, FastAPI). It calls this service's `POST /query` for graph facts when its `KG_SERVICE_URL` points here.
- [persona-ui](https://github.com/tinasheosewe/persona-ui): Next.js front end for the chat backend.

## What it does

**Typed graph models** (`kg/types.py`, `kg/schema.py`)
- Extraction results are Pydantic models. `NodeCandidate.type` must be one of 11 node types and `EdgeCandidate.relation` one of 105 relation names. Both are `Literal` types, so the models reject any other value. The node types are `Character`, `Person`, `Organization`, `Event`, `Concept`, `Work`, `Place`, `Period`, `SourceSegment`, `Episode` and `Principle`.
- `LLMExtractor` normalises the model's output before building those models: an unknown node type is an error, and an unknown relation becomes `RELATES_TO` with the original name kept in `meta.original_relation`.
- The same two vocabularies are sent to the LLM as a JSON schema (`response_format`), with separate node schemas for episodes, principles and everything else.
- Stored rows are SQLAlchemy models: `KGBasicNode`, `KGEdge` and `SourceSegment`. Their `type` and `relation` columns are plain text: the vocabularies are enforced when an extraction result is parsed, not by database constraints.

**Episodes and principles** (`kg/pipeline.py`)
- `Episode`: one situation and how the character handled it. Fields: `context`, `tension`, `response`, `rationale`, and optionally `outcome`, `confidence`, `canon_status`.
- `Principle`: a rule of behaviour drawn from episodes. Fields: `scope`, `support` (the episodes or source segments behind it), `exceptions`, `confidence`.
- These fields are requested through the JSON schema, which marks `context` and `response` as required for an episode and `scope` for a principle. They are stored in the node's JSON `meta` column; the Pydantic model accepts any `meta` object.
- Relations such as `EVIDENCED_BY`, `DERIVED_FROM`, `SUPPORTS`, `APPLIES_TO` and `EXCEPTION_OF` tie principles to episodes and episodes to source text.

**Extraction** (`kg/pipeline.py`)
- Sentence-aware segmentation with stable segment ids, so every node and edge can point back to the text it came from.
- `LLMExtractor` (OpenAI-compatible client, JSON-schema structured output, batched in parallel for long documents) or `MockExtractor` for runs without an LLM.

**PostgreSQL with a Neo4j projection** (`kg/schema.py`, `kg/neo4j_memgraph.py`)
- PostgreSQL is the primary store: `kg_nodes` (including `meta`), `kg_edges` and `source_segments`.
- The same extraction can be projected into Neo4j or Memgraph with Cypher `MERGE`: node labels come from the node types and relationship types from the relations. The projection carries names, aliases, summaries, edge descriptions, source ids and the source segments. The API does this on every ingestion when the `NEO4J_*` variables are set.
- A `character_key` on every node and edge lets several characters share one database.

**Graph-traversal retrieval** (`app/main.py`)
- `POST /query` is the one-hop lookup. It finds the focus character's node(s), returns them with every edge that touches them (both ends given by name), loads the source segments those facts cite, and, when an LLM key is set, answers the prompt from that context.
- `POST /llm-walk` walks outward from there. It starts with the focus node, its edges and the nodes those edges lead to. At each step the LLM names up to `max_expansions` nodes to expand; their edges and the nodes at the far end are fetched and merged into the context. The loop ends after `max_steps` steps or when a step adds nothing, so with the default of two steps it can reach nodes three hops from the focus.
- Both endpoints read the graph from Neo4j when the `NEO4J_*` variables are set and from PostgreSQL otherwise.

### Current limits
- The walk is driven by the LLM. Without `OPENAI_API_KEY`, `/llm-walk` returns its starting context and takes no steps.
- An expansion brings in every edge of the chosen nodes. Apart from `max_steps` and `max_expansions` nothing ranks or caps what is added, so the context grows with the degree of the nodes.
- Nodes are found by name. The focus is matched by substring (on PostgreSQL the first node whose name contains it, on Neo4j every such node, up to 200), and the walk expands nodes by exact name. There is no embedding search over the graph.
- Episode and principle fields (`meta`) are stored in PostgreSQL only. They are not copied to Neo4j and are not part of the query responses, which carry node type, name, summary, aliases and source ids.
- With Neo4j configured, both endpoints read only from Neo4j. Documents ingested before it was configured are not there until they are ingested again, and on that path `/llm-walk` does not load source segments for its answer.
- When `character` is omitted from a query, the first capitalised word of the prompt is used as the focus, so pass `character` explicitly.

## Character Knowledge Graph Toolkit (Python)

Concrete, runnable pieces to build a character-focused knowledge graph in Postgres and a small ingestion pipeline that turns documents into nodes/edges, with a FastAPI service on top.

## What's inside
- `kg/schema.py` – SQLAlchemy models for `kg_nodes`, `kg_edges`, `source_segments` (Postgres-ready, UUID keys, JSONB/ARRAY columns).
- `kg/pipeline.py` – segmentation helper, extractor protocol, LLM + mock extractors, and a `GraphBuilder` that persists nodes/edges/segments.
- `kg/types.py` – Pydantic models for extraction payloads, plus the allowed node types and the relation vocabulary.
- Episodic/principle support: `Episode` and `Principle` node types are allowed; meta can capture context/tension/response/rationale (episodes) and scope/support/exceptions (principles).
- Neo4j/Memgraph push helper: `kg/neo4j_memgraph.py` (`CypherGraphBuilder`).
- `app/main.py` – FastAPI service: `POST /ingest-book`, `POST /query`, `POST /llm-walk`, `GET /health`.
- `scripts/` – demos and helpers (see `scripts/README.md`).
- `tests/` – unit tests for segmentation, the extractors, the Cypher builder and the retrieval endpoints (fakes, no database), plus an integration test that runs the API against real databases when asked to (see [Tests](#tests)).
- `requirements.txt` – dependencies (SQLAlchemy, pydantic, psycopg2, openai, neo4j, pypdf, FastAPI, uvicorn).
- Deep dive: `docs/PIPELINE.md` (full flow, LLM contract, and retrieval patterns).

## Try it without a database
```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/run_demo_light.py --text test_files/demo_story.txt --character "Aurelia Maren"
```
This segments the sample story (`test_files/demo_story.txt`, a short fictional text) and prints the segments plus the nodes and edges from the mock extractor. Add `--use-llm` with `OPENAI_API_KEY` set for LLM extraction. The Python scripts import the `kg` package, so run them with the repository root on `PYTHONPATH`; the shell wrappers in `scripts/` set it for you.

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
`kg/pipeline.LLMExtractor` expects an OpenAI-compatible client (`openai.OpenAI()`) and a model that supports JSON-schema structured output (`response_format`).
```python
from openai import OpenAI
from kg.pipeline import GraphBuilder, LLMExtractor
from kg.schema import create_engine_and_session

client = OpenAI()  # requires OPENAI_API_KEY
relations = ["RELATES_TO","BELIEVES","OPPOSES","INFLUENCES","REFERENCES","CONTRADICTS"]
extractor = LLMExtractor(client=client, model="gpt-4o-mini", relations=relations)

engine, SessionLocal = create_engine_and_session()
session = SessionLocal()
builder = GraphBuilder(session, extractor)
builder.build_from_text(open("doc.txt").read(), work_name="Book I", character="Character Name")
```
`relations` is optional. The names must come from the vocabulary in `kg/types.py` (`RelationLiteral`); when it is omitted the whole vocabulary is offered to the model.

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
- The `neo4j` driver is part of `requirements.txt`.
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
  - Nodes labeled by `type` (Character, Work, Event, etc.) with properties `name`, `character_key`, `type`, `alias_names`, `summary`, `source_ids`.
  - `SourceSegment` nodes with `id`, `character_key`, `content`, `location`.
  - Relationships:
    - `(:Work)-[:HAS_SEGMENT]->(:SourceSegment)`
    - Extracted edges with relation type uppercased (e.g., `OPPOSES`, `BELIEVES`, `REFERENCES`) between matched node names.
- Relation naming is sanitized to uppercase snake-case; MEMGRAPH accepts the same Cypher.

## Pipeline flow
- **Segment**: `segment_text` splits on paragraph gaps, packs whole sentences up to a word limit, and assigns stable `source_segments.id` values.
- **Extract**: any `BaseExtractor` (LLM or mock) returns `NodeCandidate` and `EdgeCandidate` lists.
- **Persist**: `GraphBuilder` upserts nodes (merging aliases/meta/summary), stores segments, and dedupes edges by `(from_id, to_id, relation)`.

## Schema recap (matches Postgres DDL)
- `kg_nodes`: `id UUID PK`, `character_key`, `type`, `name`, `alias_names TEXT[]`, `summary`, `meta JSONB`, `source_ids TEXT[]`, timestamps.
- `kg_edges`: `id bigserial PK`, `character_key`, `from_id/to_id UUID FK`, `relation`, `weight`, `description`, `source_ids TEXT[]`, `meta JSONB`, `created_at`.
- `source_segments`: `id TEXT PK`, `work_id UUID FK -> kg_nodes`, `location`, `content`, `meta JSONB`, `created_at`.

`character_key` (default `default`) partitions the graph so several characters can share one database.

## Running the API
The service needs a reachable Postgres for `/query` and `/llm-walk`. The compose file can provide one:
```bash
cp .env.example .env
docker compose up -d db
export DATABASE_URL=postgresql+psycopg2://kg:kgpass@localhost:5432/kg
uvicorn app.main:app --port 8001
```
`scripts/run_api.sh` loads `.env` and starts uvicorn on `PORT` (`8001` when unset). For a run outside compose, change the host in that file's `DATABASE_URL` from `db` to `localhost`.

| Endpoint | What it does |
| --- | --- |
| `POST /ingest-book` | Multipart upload (`file`, text or PDF) with query parameters `character`, `character_key` and `work_name`. Extracts with the LLM when `OPENAI_API_KEY` is set, otherwise with the mock extractor. Writes to Postgres and, when the `NEO4J_*` variables are set, to Neo4j |
| `POST /query` | JSON body `{"prompt": "...", "character": "...", "character_key": "default"}`. Returns the focus `nodes` and the `edges` that touch them (`from` and `to` are node names), plus an LLM `answer` when `OPENAI_API_KEY` is set (otherwise `null`) |
| `POST /llm-walk` | Same body plus `max_steps` (default 2) and `max_expansions` (default 3). Starts from the focus node, its edges and its neighbours; at each step the LLM picks nodes to expand, and their edges and neighbours are added. The response lists the steps (`expand_nodes`, `added_nodes`, `added_edges`), the collected nodes and edges, and the answer. Without `OPENAI_API_KEY` no expansion happens and the starting context is returned |
| `GET /health` | Liveness check |

With the API running as above:
```bash
# Ingest the sample story for one character
curl -X POST "http://localhost:8001/ingest-book?character=Aurelia%20Maren&work_name=Demo%20Story" \
  -F "file=@test_files/demo_story.txt"

# One-hop lookup
curl -X POST http://localhost:8001/query -H "Content-Type: application/json" \
  -d '{"prompt": "What did Aurelia Maren write?", "character": "Aurelia Maren"}'

# Walk outward from the same starting point
curl -X POST http://localhost:8001/llm-walk -H "Content-Type: application/json" \
  -d '{"prompt": "Who influenced Aurelia Maren?", "character": "Aurelia Maren", "max_steps": 2}'
```
Without `OPENAI_API_KEY` the mock extractor only links the focus character to the work with a `REFERENCES` edge, so `/query` returns that edge with a `null` answer and `/llm-walk` returns the same two nodes and takes no steps. With the key set, ingestion uses the LLM extractor, `/query` adds an answer and `/llm-walk` expands the context step by step.

Environment variables: `DATABASE_URL`, `OPENAI_API_KEY`, `LLM_MODEL` (default `gpt-4o-mini`), `LLM_WALK_MODEL` (model for the expansion step, defaults to `LLM_MODEL`), `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`, and `PORT` for the Docker image and `scripts/run_api.sh`. See `.env.example`.

## Adapting to your chatbot
- Run extraction when a document uploads; call `GraphBuilder.build_from_text`.
- Keep `source_id` on every node/edge so you can trace facts back to text.
- At answer time, traverse nodes/edges for context + retrieve supporting `source_segments` (or embed them with pgvector later).

With [persona-chatbot](https://github.com/tinasheosewe/persona-chatbot): set `KG_SERVICE_URL` in its `.env` to this API's base URL (for example `http://localhost:8001`; both projects default to port 8000, so one of them needs another port). The backend posts `{"prompt", "character", "character_key"}` to `/query` and turns the returned `edges` (or `nodes`, when there are no edges) into graph facts for the prompt. It does not call `/llm-walk`. Its server uses each character's slug as `character_key`, so ingest that character's documents here with the same key.

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest
```
`python -m unittest discover -v` runs the same tests without pytest.

- `tests/test_pipeline.py` and `tests/test_cypher_builder.py` cover segmentation, the extractors and the Cypher builder with fakes.
- `tests/test_api_retrieval.py` covers `/query` and `/llm-walk` on both read paths. It runs them against an in-memory session and a fake Neo4j driver, with the walk's LLM step replaced by a scripted list of node names.
- `tests/test_integration.py` runs the API against real databases: it ingests the sample story (text and PDF) with the mock extractor, queries it, and walks the graph past the first hop. It is skipped unless `TEST_DATABASE_URL` is set:
  ```bash
  cp .env.example .env   # once; the compose file reads it
  docker compose up -d db
  TEST_DATABASE_URL=postgresql+psycopg2://kg:kgpass@localhost:5432/kg python -m pytest tests/test_integration.py
  ```
  Set `TEST_NEO4J_URI`, `TEST_NEO4J_USER` and `TEST_NEO4J_PASSWORD` as well to have the same flow write to Neo4j and read back from it. The test writes under its own `character_key` and deletes its rows afterwards.

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on Python 3.11 on every push to `main` and on pull requests. One job runs the unit tests and the two no-database demo commands (`scripts/run_demo_light.py` on the sample text and on the sample PDF). A second job runs the integration test against a PostgreSQL service container, and a third runs it with PostgreSQL and a Neo4j service container.

## Need more detail?
See `docs/PIPELINE.md` for step-by-step architecture, the LLM JSON contract, and example SQL/Cypher queries.

## Docker
- From the repository root, build the image: `docker build -t character-kg .`.
- Run against an existing Postgres: `docker run --env-file .env -e DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db -p 8000:8000 character-kg`.
- Bundle Postgres locally: `cp .env.example .env && docker compose up --build` (exposes API on `8000` and Postgres on `5432`).
- Optional Neo4j profile: `docker compose --profile neo4j up --build` then set `NEO4J_URI=neo4j://neo4j:7687`, `NEO4J_USER=neo4j`, `NEO4J_PASSWORD=kgneo4jpass` (credentials match the compose profile).
- Healthcheck once running: `GET http://localhost:8000/health`.
- The Postgres and Neo4j passwords in `docker-compose.yml` and `.env.example` are local development defaults.

## Deploying to Render
- A `render.yaml` is included. It provisions:
  - A Postgres instance (`character-kg-db`) and a web service running FastAPI (`uvicorn app.main:app`).
  - `DATABASE_URL` is wired from the managed Postgres. `LLM_MODEL` defaults to `gpt-4o-mini`. Set `OPENAI_API_KEY` in the Render dashboard if you want LLM-backed extraction/answers; leave unset to use the mock extractor.
- Deploy steps:
  1) Push to your repo with `render.yaml`.
  2) Create a new “Blueprint” on Render pointing to the repo; Render builds with `pip install -r requirements.txt` and starts with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
  3) After deploy, call `POST /ingest-book` (multipart file upload) and `POST /query` with `prompt` (+ optional `character`). Healthcheck: `GET /health`.
