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

import os
from pathlib import Path
from typing import List, Optional

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
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required for the API")
ENGINE, SessionLocal = create_engine_and_session(DATABASE_URL)


def get_session():
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


def get_cypher_builder() -> Optional[CypherGraphBuilder]:
    global neo4j_builder
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
    driver = GraphDatabase.driver(uri, auth=(user, password))
    neo4j_builder = CypherGraphBuilder(driver=driver, extractor=MockExtractor())
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


def build_llm_answer(prompt: str, nodes: List[KGBasicNode], edges: List[KGEdge]) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    node_lines = [
        f"[{n.type}] {n.name}: {n.summary or ''} (aliases={n.alias_names or []})" for n in nodes
    ]
    edge_lines = []
    id_to_name = {n.id: n.name for n in nodes}
    for e in edges:
        edge_lines.append(
            f"{id_to_name.get(e.from_id, e.from_id)} -[{e.relation}]-> {id_to_name.get(e.to_id, e.to_id)} "
            f"desc={e.description or ''} meta={e.meta or {}}"
        )
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


# --- Routes -------------------------------------------------------------------
@app.post("/ingest-book", response_model=IngestResponse)
def ingest_book(
    character: Optional[str] = None,
    character_key: str = "default",
    work_name: Optional[str] = None,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    text = read_upload(file)
    work_label = work_name or file.filename or "Uploaded Work"
    extractor = choose_extractor()
    character_key = character_key or "default"
    cypher_builder = get_cypher_builder()
    builder = GraphBuilder(session=session, extractor=extractor)
    extraction, segments = builder.build_from_text(
        text,
        work_name=work_label,
        character=character,
        return_segments=True,
        character_key=character_key,
    )
    published = False
    if cypher_builder:
        cypher_builder.project(segments=segments, extraction=extraction, work_name=work_label, work_meta={})
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
    nodes, edges = fetch_context(session, focus_character, body.prompt, character_key=body.character_key)
    answer = build_llm_answer(body.prompt, nodes, edges)
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
    return QueryResponse(answer=answer, nodes=nodes_out, edges=edges_out, prompt_used=body.prompt)


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
