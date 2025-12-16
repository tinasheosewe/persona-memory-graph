import re
from typing import Iterable, Optional

from neo4j import Driver

from .pipeline import BaseExtractor, GraphExtractionResult, segment_text
from .types import EdgeCandidate, NodeCandidate, Segment
from .pipeline import LLMExtractor  # for type checking existing_context


def _normalize_label(label: str) -> str:
    """Keep alphanumerics/underscore for labels."""
    return re.sub(r"[^A-Za-z0-9_]", "", label) or "Node"


def _normalize_rel(relation: str) -> str:
    """Upper snake-case relation for Cypher."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", relation).strip("_")
    return safe.upper() or "RELATES_TO"


class CypherGraphBuilder:
    """
    Pushes extracted nodes/edges/segments into Neo4j or Memgraph.

    Uses labels matching node `type` (Character, Person, etc.) and a `HAS_SEGMENT`
    edge from Work -> SourceSegment.
    """

    def __init__(self, driver: Driver, extractor: BaseExtractor):
        self.driver = driver
        self.extractor = extractor

    def project(
        self,
        segments: Iterable[Segment],
        extraction: GraphExtractionResult,
        work_name: str,
        work_meta: Optional[dict] = None,
        character_key: str = "default",
    ) -> GraphExtractionResult:
        """
        Store provided segments and upsert extracted nodes/edges.

        Use this when you already ran extraction (e.g., via GraphBuilder) and want
        to publish to Neo4j/Memgraph without a second LLM call.
        """
        self._store_segments(work_name, work_meta, segments, character_key)
        for n in extraction.nodes:
            if not getattr(n, "character_key", None):
                n.character_key = character_key
        for e in extraction.edges:
            if not getattr(e, "character_key", None):
                e.character_key = character_key
        self._merge_nodes(extraction.nodes, character_key)
        self._merge_edges(extraction.edges, character_key)
        return extraction

    def build_from_text(
        self,
        text: str,
        work_name: str,
        work_meta: Optional[dict] = None,
        character: Optional[str] = None,
        location_prefix: Optional[str] = None,
        character_key: str = "default",
    ) -> GraphExtractionResult:
        segments = segment_text(
            text=text,
            work_name=work_name,
            location_prefix=location_prefix,
        )
        self._store_segments(work_name, work_meta, segments, character_key)
        existing_context = None
        if isinstance(self.extractor, LLMExtractor):
            existing_context = self._build_existing_context(character_key, focus_name=character)
        extraction = self.extractor.extract(segments, character=character, existing_context=existing_context)
        for n in extraction.nodes:
            if not getattr(n, "meta", None):
                n.meta = {}
            if not n.meta:
                n.meta = {}
        for e in extraction.edges:
            pass
        for n in extraction.nodes:
            if not getattr(n, "character_key", None):
                n.character_key = character_key
        for e in extraction.edges:
            if not getattr(e, "character_key", None):
                e.character_key = character_key
        self._merge_nodes(extraction.nodes, character_key)
        self._merge_edges(extraction.edges, character_key)
        return extraction

    # --- internal helpers -------------------------------------------------

    def _store_segments(
        self, work_name: str, work_meta: Optional[dict], segments: Iterable[Segment], character_key: str
    ) -> None:
        with self.driver.session() as session:
            session.execute_write(
                self._merge_work,
                work_name,
                work_meta or {},
                character_key,
            )
            for seg in segments:
                session.execute_write(
                    self._merge_segment,
                    work_name,
                    seg,
                    character_key,
                )

    @staticmethod
    def _merge_work(tx, work_name: str, work_meta: dict, character_key: str):
        tx.run(
            """
            MERGE (w:Work {name: $name, character_key: $character_key})
            ON CREATE SET w.type='Work'
            """,
            name=work_name,
            character_key=character_key,
        )

    @staticmethod
    def _merge_segment(tx, work_name: str, seg: Segment, character_key: str):
        tx.run(
            """
            MERGE (w:Work {name: $work_name, character_key: $character_key})
            ON CREATE SET w.type='Work'
            MERGE (s:SourceSegment {id: $id, character_key: $character_key})
            ON CREATE SET s.content=$content, s.location=$location
            MERGE (w)-[:HAS_SEGMENT {character_key: $character_key}]->(s)
            """,
            work_name=work_name,
            id=seg.id,
            content=seg.content,
            location=seg.location,
            character_key=character_key,
        )

    def _merge_nodes(self, nodes: Iterable[NodeCandidate], character_key: str) -> None:
        with self.driver.session() as session:
            for node in nodes:
                label = _normalize_label(node.type)
                session.execute_write(
                    self._merge_node_tx,
                    label,
                    node,
                    character_key,
                )

    @staticmethod
    def _merge_node_tx(tx, label: str, node: NodeCandidate, character_key: str):
        tx.run(
            f"""
            MERGE (n:{label} {{name: $name, character_key: $character_key}})
            ON CREATE SET n.type=$type, n.alias_names=$alias_names, n.summary=$summary, n.source_ids=CASE WHEN $source_id IS NULL THEN [] ELSE [$source_id] END
            ON MATCH SET
                n.alias_names = coalesce(n.alias_names, []) + $alias_names,
                n.summary = coalesce(n.summary, $summary),
                n.source_ids = coalesce(n.source_ids, []) + CASE WHEN $source_id IS NULL THEN [] ELSE [$source_id] END,
                n.character_key = $character_key
            """,
            name=node.name,
            type=node.type,
            alias_names=node.alias_names,
            summary=node.summary,
            source_id=node.source_id,
            character_key=character_key,
        )

    def _merge_edges(self, edges: Iterable[EdgeCandidate], character_key: str) -> None:
        with self.driver.session() as session:
            for edge in edges:
                rel_type = _normalize_rel(edge.relation)
                session.execute_write(
                    self._merge_edge_tx,
                    rel_type,
                    edge,
                    character_key,
                )

    def _build_existing_context(self, character_key: str, focus_name: Optional[str]) -> Optional[str]:
        """
        Build a compact textual summary of existing graph facts to discourage duplicates.
        """
        try:
            with self.driver.session() as session:
                node_query = """
                MATCH (n {character_key: $ck})
                WHERE $name IS NULL OR n.name CONTAINS $name
                RETURN n.name AS name, n.type AS type, n.alias_names AS aliases, n.summary AS summary
                LIMIT 30
                """
                edge_query = """
                MATCH (a {character_key: $ck})-[r]->(b {character_key: $ck})
                WHERE ($name IS NULL OR a.name CONTAINS $name OR b.name CONTAINS $name)
                RETURN a.name AS from_name, b.name AS to_name, type(r) AS relation
                LIMIT 50
                """
                nodes = session.run(node_query, ck=character_key, name=focus_name).data()
                edges = session.run(edge_query, ck=character_key, name=focus_name).data()
        except Exception:
            return None

        node_lines = [
            f"[{n.get('type')}] {n.get('name')} aliases={n.get('aliases') or []} summary={n.get('summary') or ''}"
            for n in nodes
        ]
        edge_lines = [
            f"{e.get('from_name')} -[{e.get('relation')}]-> {e.get('to_name')}"
            for e in edges
        ]
        if not node_lines and not edge_lines:
            return None
        return "\n".join(node_lines + edge_lines)

    @staticmethod
    def _merge_edge_tx(tx, rel_type: str, edge: EdgeCandidate, character_key: str):
        tx.run(
            f"""
            MATCH (a {{name: $from_name, character_key: $character_key}}), (b {{name: $to_name, character_key: $character_key}})
            MERGE (a)-[r:{rel_type} {{character_key: $character_key}}]->(b)
            ON CREATE SET r.weight=$weight, r.description=$description, r.source_ids = CASE WHEN $source_id IS NULL THEN [] ELSE [$source_id] END
            ON MATCH SET
                r.weight = coalesce(r.weight, $weight),
                r.description = coalesce(r.description, $description),
                r.source_ids = coalesce(r.source_ids, []) + CASE WHEN $source_id IS NULL THEN [] ELSE [$source_id] END,
                r.character_key = $character_key
            """,
            from_name=edge.from_name,
            to_name=edge.to_name,
            weight=edge.confidence,
            description=edge.description,
            source_id=edge.source_id,
            character_key=character_key,
        )
