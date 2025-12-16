"""
FastAPI service for ingesting books into the graph and querying with LLM-backed traversal.

Endpoints:
- POST /ingest-book: upload a text/PDF file, extract nodes/edges, persist to DB.
- POST /query: provide a prompt (+ optional character); fetch related nodes/edges and ask the LLM to answer.

Environment:
- DATABASE_URL (required)
- OPENAI_API_KEY (optional; when absent, ingestion/query falls back to mock extraction or returns context only)
- LLM_MODEL (optional; default gpt-4o-mini)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Set

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from kg.pipeline import GraphBuilder, LLMExtractor, MockExtractor, segment_text
from kg.schema import KGBasicNode, KGEdge, create_engine_and_session
from kg.neo4j_memgraph import CypherGraphBuilder

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

app = FastAPI(title="Character KG API", version="0.1.0")


# --- DB session dependency ----------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
ENGINE = None
SessionLocal = None
if DATABASE_URL:
    ENGINE, SessionLocal = create_engine_and_session(DATABASE_URL)


def get_session():
    if SessionLocal is None:
        yield None
        return
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# --- Models -------------------------------------------------------------------
class IngestResponse(BaseModel):
    work_name: str
    character: Optional[str]
    nodes: int
    edges: int
    segments: int
    published_to_neo4j: bool = False


class QueryRequest(BaseModel):
    prompt: str
    character: Optional[str] = None
    character_key: str = "default"


class QueryResponse(BaseModel):
    answer: Optional[str]
    nodes: List[dict]
    edges: List[dict]
    prompt_used: str


class LLMWalkRequest(BaseModel):
    prompt: str
    character: Optional[str] = None
    character_key: str = "default"
    max_steps: int = 2
    max_expansions: int = 3


class LLMWalkStep(BaseModel):
    expand_nodes: List[str]
    added_nodes: int
    added_edges: int


class LLMWalkResponse(BaseModel):
    prompt_used: str
    steps: List[LLMWalkStep]
    nodes: List[dict]
    edges: List[dict]
    answer: Optional[str]


# --- Helpers ------------------------------------------------------------------
def choose_extractor():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        return LLMExtractor(client=client, model=model)
    return MockExtractor()


neo4j_builder: Optional[CypherGraphBuilder] = None
neo4j_driver = None


def get_cypher_builder() -> Optional[CypherGraphBuilder]:
    global neo4j_builder, neo4j_driver
    if neo4j_builder:
        return neo4j_builder
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not password:
        return None
    try:
        from neo4j import GraphDatabase
    except Exception:
        return None
    neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    neo4j_builder = CypherGraphBuilder(driver=neo4j_driver, extractor=MockExtractor())
    return neo4j_builder


def read_upload(file: UploadFile) -> str:
    content = file.file.read()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix == ".pdf":
        if PdfReader is None:
            raise HTTPException(status_code=400, detail="pypdf not installed; cannot read PDF")
        reader = PdfReader(file.file)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    # assume text-like
    return content.decode("utf-8", errors="ignore")


def resolve_character(prompt: str, character: Optional[str]) -> Optional[str]:
    if character:
        return character
    # naive: use first capitalized token as potential character
    for token in prompt.split():
        if token.istitle():
            return token
    return None


def fetch_context(session: Session, character: Optional[str], prompt: str, character_key: str):
    nodes = []
    edges = []
    if character:
        node = session.scalar(
            select(KGBasicNode).where(
                KGBasicNode.name.ilike(f"%{character}%"),
                KGBasicNode.character_key == character_key,
            )
        )
        if node:
            nodes = [node]
            edges = session.scalars(
                select(KGEdge).where(
                    KGEdge.character_key == character_key,
                    or_(KGEdge.from_id == node.id, KGEdge.to_id == node.id),
                )
            ).all()
    if not nodes:
        # fallback: fuzzy match on prompt tokens
        key = prompt.split()[0] if prompt else ""
        if key:
            nodes = session.scalars(
                select(KGBasicNode).where(
                    KGBasicNode.name.ilike(f"%{key}%"),
                    KGBasicNode.character_key == character_key,
                )
            ).all()
            if nodes:
                ids = [n.id for n in nodes]
                edges = session.scalars(
                    select(KGEdge).where(
                        KGEdge.character_key == character_key,
                        or_(KGEdge.from_id.in_(ids), KGEdge.to_id.in_(ids)),
                    )
                ).all()
    return nodes, edges


def fetch_context_neo4j(character: Optional[str], prompt: str, character_key: str):
    if not neo4j_driver:
        return [], []
    name_filter = character or resolve_character(prompt, None)
    node_query = """
    MATCH (n {character_key: $ck})
    WHERE $name IS NULL OR n.name CONTAINS $name
    RETURN n LIMIT 200
    """
    edge_query = """
    MATCH (a {character_key: $ck})-[r]->(b {character_key: $ck})
    WHERE ($name IS NULL OR a.name CONTAINS $name OR b.name CONTAINS $name)
    RETURN a.name AS from_name, b.name AS to_name, type(r) AS relation, r.description AS description, r.meta AS meta, r.source_ids AS source_ids LIMIT 300
    """
    with neo4j_driver.session() as session:
        nodes = [
            {
                "id": str(record["n"].id),
                "type": record["n"].get("type"),
                "name": record["n"].get("name"),
                "summary": record["n"].get("summary"),
                "aliases": record["n"].get("alias_names", []),
                "character_key": record["n"].get("character_key"),
            }
            for record in session.run(node_query, ck=character_key, name=name_filter)
        ]
        edges = [
            {
                "from": record["from_name"],
                "to": record["to_name"],
                "relation": record["relation"],
                "description": record["description"],
                "meta": record["meta"],
                "source_ids": record["source_ids"],
                "character_key": character_key,
            }
            for record in session.run(edge_query, ck=character_key, name=name_filter)
        ]
    return nodes, edges


def build_llm_answer_from_dicts(prompt: str, nodes: List[dict], edges: List[dict]) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    node_lines = [
        f"[{n.get('type')}] {n.get('name')}: {n.get('summary') or ''} (aliases={n.get('aliases') or []})"
        for n in nodes
    ]
    edge_lines = [
        f"{e.get('from')} -[{e.get('relation')}]-> {e.get('to')} desc={e.get('description') or ''} meta={e.get('meta') or {}}"
        for e in edges
    ]
    context = "Nodes:\n" + "\n".join(node_lines) + "\nEdges:\n" + "\n".join(edge_lines)
    message = (
        "You are a graph-aware assistant. Use only the provided nodes/edges to answer concisely.\n"
        f"Question: {prompt}\n\nContext:\n{context}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message.content


def choose_walk_targets(prompt: str, nodes: List[dict], edges: List[dict], max_expansions: int) -> List[str]:
    """
    Ask the LLM which nodes to expand next based on current context.
    Falls back to an empty list if no API key is present.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    node_lines = [
        f"[{n.get('type')}] {n.get('name')}: {n.get('summary') or ''}"
        for n in nodes
    ]
    edge_lines = [
        f"{e.get('from')} -[{e.get('relation')}]-> {e.get('to')}"
        for e in edges
    ]
    content = (
        "You are deciding which nodes to expand next in a knowledge graph to answer a query.\n"
        f"Query: {prompt}\n"
        "Current nodes:\n" + "\n".join(node_lines) + "\n"
        "Current edges:\n" + "\n".join(edge_lines) + "\n"
        f"Return a JSON object with key 'expand' listing up to {max_expansions} node names to explore for more neighbors.\n"
        "If no expansion is needed, return an empty list."
    )
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(res.choices[0].message.content)
        expand = data.get("expand") or []
        if isinstance(expand, list):
            return [str(x) for x in expand][:max_expansions]
    except Exception:
        return []
    return []


# --- Routes -------------------------------------------------------------------
@app.post("/ingest-book", response_model=IngestResponse)
def ingest_book(
    character: Optional[str] = None,
    character_key: str = "default",
    work_name: Optional[str] = None,
    file: UploadFile = File(...),
    session: Optional[Session] = Depends(get_session),
):
    text = read_upload(file)
    work_label = work_name or file.filename or "Uploaded Work"
    extractor = choose_extractor()
    character_key = character_key or "default"
    cypher_builder = get_cypher_builder()
    published = False
    if session:
        builder = GraphBuilder(session=session, extractor=extractor)
        extraction, segments = builder.build_from_text(
            text,
            work_name=work_label,
            character=character,
            return_segments=True,
            character_key=character_key,
        )
        if cypher_builder:
            cypher_builder.project(
                segments=segments,
                extraction=extraction,
                work_name=work_label,
                work_meta={"character_key": character_key},
                character_key=character_key,
            )
            published = True
    else:
        # No DB; run extraction and publish to Neo4j only
        segments = segment_text(text, work_name=work_label)
        extraction = extractor.extract(segments, character=character)
        for n in extraction.nodes:
            if not n.character_key:
                n.character_key = character_key
        for e in extraction.edges:
            if not e.character_key:
                e.character_key = character_key
        if cypher_builder:
            cypher_builder.project(
                segments=segments,
                extraction=extraction,
                work_name=work_label,
                work_meta={"character_key": character_key},
                character_key=character_key,
            )
            published = True
    return IngestResponse(
        work_name=work_label,
        character=character,
        nodes=len(extraction.nodes),
        edges=len(extraction.edges),
        segments=len(segments),
        published_to_neo4j=published,
    )


@app.post("/query", response_model=QueryResponse)
def query_graph(body: QueryRequest, session: Session = Depends(get_session)):
    focus_character = resolve_character(body.prompt, body.character)
    nodes_out: List[dict] = []
    edges_out: List[dict] = []
    if neo4j_driver:
        nodes_out, edges_out = fetch_context_neo4j(focus_character, body.prompt, character_key=body.character_key)
    elif session:
        nodes, edges = fetch_context(session, focus_character, body.prompt, character_key=body.character_key)
        nodes_out = [
            {
                "id": str(n.id),
                "type": n.type,
                "name": n.name,
                "summary": n.summary,
                "aliases": n.alias_names,
                "character_key": n.character_key,
            }
            for n in nodes
        ]
        id_to_name = {n["id"]: n["name"] for n in nodes_out}
        edges_out = [
            {
                "id": e.id,
                "from": id_to_name.get(str(e.from_id), str(e.from_id)),
                "to": id_to_name.get(str(e.to_id), str(e.to_id)),
                "relation": e.relation,
                "description": e.description,
                "meta": e.meta,
            }
            for e in edges
        ]
    else:
        nodes_out, edges_out = [], []
    answer = build_llm_answer_from_dicts(body.prompt, nodes_out, edges_out)
    return QueryResponse(answer=answer, nodes=nodes_out, edges=edges_out, prompt_used=body.prompt)


def fetch_neighbors(session: Session, node_ids: Set[str], character_key: str):
    if not node_ids:
        return [], []
    nodes = session.scalars(
        select(KGBasicNode).where(
            KGBasicNode.character_key == character_key,
            KGBasicNode.id.in_(list(node_ids)),
        )
    ).all()
    edges = session.scalars(
        select(KGEdge).where(
            KGEdge.character_key == character_key,
            or_(KGEdge.from_id.in_(list(node_ids)), KGEdge.to_id.in_(list(node_ids))),
        )
    ).all()
    return nodes, edges


@app.post("/llm-walk", response_model=LLMWalkResponse)
def llm_walk(body: LLMWalkRequest, session: Session = Depends(get_session)):
    if session is None:
        raise HTTPException(status_code=400, detail="Database session required for graph walk")

    focus_character = resolve_character(body.prompt, body.character)
    nodes, edges = fetch_context(session, focus_character, body.prompt, character_key=body.character_key)
    steps: List[LLMWalkStep] = []

    # Serialize initial context
    nodes_out = [
        {
            "id": str(n.id),
            "type": n.type,
            "name": n.name,
            "summary": n.summary,
            "aliases": n.alias_names,
            "character_key": n.character_key,
        }
        for n in nodes
    ]
    id_to_name = {n["id"]: n["name"] for n in nodes_out}
    edges_out = [
        {
            "id": e.id,
            "from": id_to_name.get(str(e.from_id), str(e.from_id)),
            "to": id_to_name.get(str(e.to_id), str(e.to_id)),
            "relation": e.relation,
            "description": e.description,
            "meta": e.meta,
        }
        for e in edges
    ]

    seen_node_ids: Set[str] = set(id_to_name.keys())
    seen_edge_keys: Set[tuple] = set((edge["from"], edge["to"], edge["relation"]) for edge in edges_out)

    for _ in range(max(0, body.max_steps)):
        expand_names = choose_walk_targets(body.prompt, nodes_out, edges_out, body.max_expansions)
        if not expand_names:
            break
        # Map requested names to known node ids; if not found, skip
        target_ids = {nid for nid, name in id_to_name.items() if name in expand_names}
        if not target_ids:
            break
        new_nodes, new_edges = fetch_neighbors(session, target_ids, character_key=body.character_key)

        added_nodes = 0
        for n in new_nodes:
            nid = str(n.id)
            if nid in seen_node_ids:
                continue
            nodes_out.append(
                {
                    "id": nid,
                    "type": n.type,
                    "name": n.name,
                    "summary": n.summary,
                    "aliases": n.alias_names,
                    "character_key": n.character_key,
                }
            )
            id_to_name[nid] = n.name
            seen_node_ids.add(nid)
            added_nodes += 1

        added_edges = 0
        for e in new_edges:
            from_name = id_to_name.get(str(e.from_id), str(e.from_id))
            to_name = id_to_name.get(str(e.to_id), str(e.to_id))
            key = (from_name, to_name, e.relation)
            if key in seen_edge_keys:
                continue
            edges_out.append(
                {
                    "id": e.id,
                    "from": from_name,
                    "to": to_name,
                    "relation": e.relation,
                    "description": e.description,
                    "meta": e.meta,
                }
            )
            seen_edge_keys.add(key)
            added_edges += 1

        steps.append(LLMWalkStep(expand_nodes=list(expand_names), added_nodes=added_nodes, added_edges=added_edges))
        if added_nodes == 0 and added_edges == 0:
            break

    answer = build_llm_answer_from_dicts(body.prompt, nodes_out, edges_out)
    return LLMWalkResponse(
        prompt_used=body.prompt,
        steps=steps,
        nodes=nodes_out,
        edges=edges_out,
        answer=answer,
    )


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
