import re
from typing import Iterable, Optional

from neo4j import Driver

from .pipeline import BaseExtractor, GraphExtractionResult, segment_text
from .types import EdgeCandidate, NodeCandidate, Segment


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
    ) -> GraphExtractionResult:
        """
        Store provided segments and upsert extracted nodes/edges.

        Use this when you already ran extraction (e.g., via GraphBuilder) and want
        to publish to Neo4j/Memgraph without a second LLM call.
        """
        self._store_segments(work_name, work_meta, segments)
        self._merge_nodes(extraction.nodes)
        self._merge_edges(extraction.edges)
        return extraction

    def build_from_text(
        self,
        text: str,
        work_name: str,
        work_meta: Optional[dict] = None,
        character: Optional[str] = None,
        location_prefix: Optional[str] = None,
    ) -> GraphExtractionResult:
        segments = segment_text(
            text=text,
            work_name=work_name,
            location_prefix=location_prefix,
        )
        self._store_segments(work_name, work_meta, segments)
        extraction = self.extractor.extract(segments, character=character)
        self._merge_nodes(extraction.nodes)
        self._merge_edges(extraction.edges)
        return extraction

    # --- internal helpers -------------------------------------------------

    def _store_segments(
        self, work_name: str, work_meta: Optional[dict], segments: Iterable[Segment]
    ) -> None:
        with self.driver.session() as session:
            session.execute_write(
                self._merge_work,
                work_name,
                work_meta or {},
            )
            for seg in segments:
                session.execute_write(
                    self._merge_segment,
                    work_name,
                    seg,
                )

    @staticmethod
    def _merge_work(tx, work_name: str, work_meta: dict):
        tx.run(
            """
            MERGE (w:Work {name: $name})
            ON CREATE SET w.type='Work', w.meta=$meta
            """,
            name=work_name,
            meta=work_meta,
        )

    @staticmethod
    def _merge_segment(tx, work_name: str, seg: Segment):
        tx.run(
            """
            MERGE (w:Work {name: $work_name})
            ON CREATE SET w.type='Work'
            MERGE (s:SourceSegment {id: $id})
            ON CREATE SET s.content=$content, s.location=$location, s.meta=$meta
            MERGE (w)-[:HAS_SEGMENT]->(s)
            """,
            work_name=work_name,
            id=seg.id,
            content=seg.content,
            location=seg.location,
            meta=seg.meta,
        )

    def _merge_nodes(self, nodes: Iterable[NodeCandidate]) -> None:
        with self.driver.session() as session:
            for node in nodes:
                label = _normalize_label(node.type)
                session.execute_write(
                    self._merge_node_tx,
                    label,
                    node,
                )

    @staticmethod
    def _merge_node_tx(tx, label: str, node: NodeCandidate):
        tx.run(
            f"""
            MERGE (n:{label} {{name: $name}})
            ON CREATE SET n.type=$type, n.alias_names=$alias_names, n.summary=$summary, n.meta=$meta, n.source_ids=CASE WHEN $source_id IS NULL THEN [] ELSE [$source_id] END
            ON MATCH SET
                n.alias_names = coalesce(n.alias_names, []) + $alias_names,
                n.meta = coalesce(n.meta, {{}}) + $meta,
                n.summary = coalesce(n.summary, $summary),
                n.source_ids = coalesce(n.source_ids, []) + CASE WHEN $source_id IS NULL THEN [] ELSE [$source_id] END
            """,
            name=node.name,
            type=node.type,
            alias_names=node.alias_names,
            summary=node.summary,
            meta=node.meta,
            source_id=node.source_id,
        )

    def _merge_edges(self, edges: Iterable[EdgeCandidate]) -> None:
        with self.driver.session() as session:
            for edge in edges:
                rel_type = _normalize_rel(edge.relation)
                session.execute_write(
                    self._merge_edge_tx,
                    rel_type,
                    edge,
                )

    @staticmethod
    def _merge_edge_tx(tx, rel_type: str, edge: EdgeCandidate):
        tx.run(
            f"""
            MATCH (a {{name: $from_name}}), (b {{name: $to_name}})
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET r.weight=$weight, r.description=$description, r.source_ids = CASE WHEN $source_id IS NULL THEN [] ELSE [$source_id] END
            ON MATCH SET
                r.weight = coalesce(r.weight, $weight),
                r.description = coalesce(r.description, $description),
                r.source_ids = coalesce(r.source_ids, []) + CASE WHEN $source_id IS NULL THEN [] ELSE [$source_id] END
            """,
            from_name=edge.from_name,
            to_name=edge.to_name,
            weight=edge.confidence,
            description=edge.description,
            source_id=edge.source_id,
        )
