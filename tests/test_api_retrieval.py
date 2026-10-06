"""
Retrieval endpoints (/query and /llm-walk) over small in-memory graphs.

No database and no LLM: the PostgreSQL path runs against FakeSession, which evaluates the
SELECT statements app.main builds, and the Neo4j path runs against FakeNeo4jDriver. The
walk's LLM step (choose_walk_targets) is replaced by a scripted list of node names.
"""

import os
import re
import unittest
import uuid
from unittest import mock

from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import BooleanClauseList, Grouping

from kg.schema import KGBasicNode, KGEdge, SourceSegment

# app.main opens a database connection at import time when DATABASE_URL is set. These tests
# bring their own sessions, so hide the variable while the module loads.
with mock.patch.dict(os.environ):
    os.environ.pop("DATABASE_URL", None)
    import app.main as main

# The graph used by every test, as (from, relation, to). From the focus character the
# other nodes are one, two and three hops away:
#   Aurelia Maren -> Elias Nyberg <- Harbor Circle -> Porto
CHAIN = [
    ("Aurelia Maren", "PRAISES", "Elias Nyberg"),
    ("Harbor Circle", "INSPIRED_BY", "Elias Nyberg"),
    ("Harbor Circle", "RELATES_TO", "Porto"),
]
NODE_TYPES = {
    "Aurelia Maren": "Character",
    "Elias Nyberg": "Person",
    "Harbor Circle": "Organization",
    "Porto": "Place",
}
CHARACTER_KEY = "story"


def looks_like_uuid(value) -> bool:
    try:
        uuid.UUID(str(value))
    except ValueError:
        return False
    return True


def edge_triples(edges):
    return {(e["from"], e["relation"], e["to"]) for e in edges}


def node_names(nodes):
    return [n["name"] for n in nodes]


# --- PostgreSQL path: a session that answers SELECTs from lists ------------------


def _same(left, right) -> bool:
    # The database compares UUIDs by value, whether the parameter is a UUID or its string form.
    return str(left) == str(right)


def _matches(clause, row) -> bool:
    """Evaluate a WHERE clause built by app.main against one in-memory row."""
    if isinstance(clause, Grouping):
        return _matches(clause.element, row)
    if isinstance(clause, BooleanClauseList):
        results = [_matches(part, row) for part in clause.clauses]
        return any(results) if clause.operator is operators.or_ else all(results)
    value = getattr(row, clause.left.key)
    expected = clause.right.value
    if clause.operator is operators.eq:
        return _same(value, expected)
    if clause.operator is operators.in_op:
        return any(_same(value, item) for item in expected)
    if clause.operator is operators.ilike_op:
        pattern = ".*".join(re.escape(part) for part in expected.split("%"))
        return re.fullmatch(pattern, value or "", flags=re.IGNORECASE) is not None
    raise NotImplementedError(f"FakeSession cannot evaluate {clause.operator}")


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class FakeSession:
    """Stands in for a SQLAlchemy Session over kg_nodes, kg_edges and source_segments."""

    def __init__(self, nodes=(), edges=(), segments=()):
        self.rows = {KGBasicNode: list(nodes), KGEdge: list(edges), SourceSegment: list(segments)}

    def _select(self, statement):
        entity = statement.column_descriptions[0]["entity"]
        where = statement.whereclause
        return [row for row in self.rows[entity] if where is None or _matches(where, row)]

    def scalars(self, statement):
        return FakeResult(self._select(statement))

    def scalar(self, statement):
        rows = self._select(statement)
        return rows[0] if rows else None


def build_postgres_rows():
    nodes = {
        name: KGBasicNode(
            id=uuid.uuid4(),
            character_key=CHARACTER_KEY,
            type=node_type,
            name=name,
            alias_names=[],
            summary=f"{name} in the sample story",
            meta={},
            source_ids=["story-0-0"],
        )
        for name, node_type in NODE_TYPES.items()
    }
    edges = [
        KGEdge(
            id=index,
            character_key=CHARACTER_KEY,
            from_id=nodes[from_name].id,
            to_id=nodes[to_name].id,
            relation=relation,
            description=None,
            meta={},
            source_ids=["story-0-0"],
        )
        for index, (from_name, relation, to_name) in enumerate(CHAIN, start=1)
    ]
    # The same name under another character_key must never leak into the results.
    other_focus = KGBasicNode(
        id=uuid.uuid4(), character_key="other", type="Character", name="Aurelia Maren",
        alias_names=[], summary=None, meta={}, source_ids=[],
    )
    other_place = KGBasicNode(
        id=uuid.uuid4(), character_key="other", type="Place", name="Lisbon",
        alias_names=[], summary=None, meta={}, source_ids=[],
    )
    other_edge = KGEdge(
        id=99, character_key="other", from_id=other_focus.id, to_id=other_place.id,
        relation="RELATES_TO", description=None, meta={}, source_ids=[],
    )
    segments = [SourceSegment(id="story-0-0", location="p0-0", content="Aurelia Maren praised Elias Nyberg.", meta={})]
    return list(nodes.values()) + [other_focus, other_place], edges + [other_edge], segments


class TestPostgresReadPath(unittest.TestCase):
    def setUp(self):
        nodes, edges, segments = build_postgres_rows()
        self.session = FakeSession(nodes=nodes, edges=edges, segments=segments)
        # No LLM key and no Neo4j settings, whatever the developer's shell has.
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY" and not k.startswith("NEO4J_")}
        patches = [
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(main, "neo4j_driver", None),
            mock.patch.object(main, "neo4j_builder", None),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_query_reports_edge_endpoints_by_name(self):
        body = main.QueryRequest(prompt="Who taught Aurelia Maren?", character="Aurelia Maren", character_key=CHARACTER_KEY)
        response = main.query_graph(body, session=self.session)

        self.assertEqual(node_names(response.nodes), ["Aurelia Maren"])
        self.assertEqual(edge_triples(response.edges), {("Aurelia Maren", "PRAISES", "Elias Nyberg")})
        for edge in response.edges:
            self.assertFalse(looks_like_uuid(edge["from"]), edge)
            self.assertFalse(looks_like_uuid(edge["to"]), edge)
        self.assertIsNone(response.answer)

    def test_walk_starts_with_the_focus_node_and_its_neighbours(self):
        # Without an LLM key no expansion is chosen, so this is the starting context.
        body = main.LLMWalkRequest(prompt="Who taught Aurelia Maren?", character="Aurelia Maren", character_key=CHARACTER_KEY)
        response = main.llm_walk(body, session=self.session)

        self.assertEqual(response.steps, [])
        self.assertEqual(node_names(response.nodes), ["Aurelia Maren", "Elias Nyberg"])
        self.assertEqual(edge_triples(response.edges), {("Aurelia Maren", "PRAISES", "Elias Nyberg")})

    def test_walk_reaches_nodes_beyond_the_first_hop(self):
        body = main.LLMWalkRequest(
            prompt="Where did the people around Aurelia Maren meet?",
            character="Aurelia Maren",
            character_key=CHARACTER_KEY,
            max_steps=2,
        )
        targets = [["Elias Nyberg"], ["Harbor Circle"]]
        with mock.patch.object(main, "choose_walk_targets", side_effect=targets):
            response = main.llm_walk(body, session=self.session)

        self.assertEqual(
            [(step.expand_nodes, step.added_nodes, step.added_edges) for step in response.steps],
            [(["Elias Nyberg"], 1, 1), (["Harbor Circle"], 1, 1)],
        )
        self.assertEqual(
            node_names(response.nodes),
            ["Aurelia Maren", "Elias Nyberg", "Harbor Circle", "Porto"],
        )
        self.assertEqual(edge_triples(response.edges), set(CHAIN))
        for edge in response.edges:
            self.assertFalse(looks_like_uuid(edge["from"]), edge)
            self.assertFalse(looks_like_uuid(edge["to"]), edge)

    def test_walk_stops_after_max_steps(self):
        body = main.LLMWalkRequest(
            prompt="Where did the people around Aurelia Maren meet?",
            character="Aurelia Maren",
            character_key=CHARACTER_KEY,
            max_steps=1,
        )
        targets = [["Elias Nyberg"], ["Harbor Circle"]]
        with mock.patch.object(main, "choose_walk_targets", side_effect=targets) as choose:
            response = main.llm_walk(body, session=self.session)

        self.assertEqual(choose.call_count, 1)
        self.assertEqual(len(response.steps), 1)
        self.assertNotIn("Porto", node_names(response.nodes))

    def test_fetch_neighbors_returns_the_nodes_at_the_far_end(self):
        elias = next(n for n in self.session.rows[KGBasicNode] if n.name == "Elias Nyberg")
        nodes, edges = main.fetch_neighbors(self.session, {str(elias.id)}, character_key=CHARACTER_KEY)

        self.assertEqual({n.name for n in nodes}, {"Elias Nyberg", "Aurelia Maren", "Harbor Circle"})
        self.assertEqual({e.relation for e in edges}, {"PRAISES", "INSPIRED_BY"})


# --- Neo4j path: a driver that answers the read queries from lists ----------------


class FakeNeo4jNode(dict):
    """A property map with identifiers, like neo4j.graph.Node."""

    def __init__(self, number, labels, **properties):
        super().__init__(properties)
        self.id = number
        self.element_id = f"4:fake:{number}"
        self.labels = frozenset(labels)


class FakeNeo4jSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def run(self, query, **params):
        self.driver.queries.append((query, params))

        def in_partition(node) -> bool:
            return node.get("character_key") == params["ck"]

        def selected(node) -> bool:
            if "names" in params:
                return node.get("name") in params["names"]
            name = params.get("name")
            return name is None or name in (node.get("name") or "")

        if "RETURN n" in query:
            nodes = [n for n in self.driver.nodes if in_partition(n) and selected(n)]
            if "SourceSegment" in query:  # the query filters segment nodes out
                nodes = [n for n in nodes if "SourceSegment" not in n.labels]
            return [{"n": n} for n in nodes]

        records = []
        for start, relation, end, properties in self.driver.relationships:
            if not (in_partition(start) and in_partition(end)):
                continue
            if relation == "HAS_SEGMENT" and "HAS_SEGMENT" in query:  # the query filters these out
                continue
            if selected(start) or selected(end):
                records.append(
                    {
                        "from_name": start.get("name"),
                        "to_name": end.get("name"),
                        "relation": relation,
                        "description": properties.get("description"),
                        "meta": properties.get("meta"),
                        "source_ids": properties.get("source_ids"),
                    }
                )
        return records


class FakeNeo4jDriver:
    """Holds the sample graph as CypherGraphBuilder projects it, segments included."""

    def __init__(self):
        self.queries = []
        nodes = {
            name: FakeNeo4jNode(
                number,
                [node_type],
                name=name,
                type=node_type,
                character_key=CHARACTER_KEY,
                summary=f"{name} in the sample story",
                source_ids=["story-0-0"],
            )
            for number, (name, node_type) in enumerate(NODE_TYPES.items(), start=1)
        }
        work = FakeNeo4jNode(10, ["Work"], name="Demo Story", type="Work", character_key=CHARACTER_KEY)
        segment = FakeNeo4jNode(
            11, ["SourceSegment"], id="story-0-0", content="Aurelia Maren praised Elias Nyberg.",
            location="p0-0", character_key=CHARACTER_KEY,
        )
        self.nodes = list(nodes.values()) + [work, segment]
        self.relationships = [
            (nodes[from_name], relation, nodes[to_name], {"source_ids": ["story-0-0"]})
            for from_name, relation, to_name in CHAIN
        ]
        self.relationships.append((nodes["Aurelia Maren"], "REFERENCES", work, {"source_ids": ["story-0-0"]}))
        self.relationships.append((work, "HAS_SEGMENT", segment, {}))

    def session(self):
        return FakeNeo4jSession(self)


NEO4J_SETTINGS = {
    "NEO4J_URI": "bolt://neo4j.invalid:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "not-a-real-password",
}


class TestNeo4jReadPath(unittest.TestCase):
    def setUp(self):
        self.driver = FakeNeo4jDriver()
        # Postgres holds no graph here, so any node or edge in a response came from Neo4j.
        self.session = FakeSession(
            segments=[SourceSegment(id="story-0-0", location="p0-0", content="Aurelia Maren praised Elias Nyberg.", meta={})]
        )
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        env.update(NEO4J_SETTINGS)
        patches = [
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(main, "neo4j_driver", None),
            mock.patch.object(main, "neo4j_builder", None),
            mock.patch("neo4j.GraphDatabase.driver", return_value=self.driver),
        ]
        started = [patch.start() for patch in patches]
        for patch in patches:
            self.addCleanup(patch.stop)
        self.driver_factory = started[-1]

    def test_query_opens_the_driver_and_reads_from_neo4j(self):
        body = main.QueryRequest(prompt="Who taught Aurelia Maren?", character="Aurelia Maren", character_key=CHARACTER_KEY)
        response = main.query_graph(body, session=self.session)

        self.driver_factory.assert_called_once_with(
            NEO4J_SETTINGS["NEO4J_URI"],
            auth=(NEO4J_SETTINGS["NEO4J_USER"], NEO4J_SETTINGS["NEO4J_PASSWORD"]),
        )
        self.assertEqual(node_names(response.nodes), ["Aurelia Maren"])
        self.assertEqual(
            edge_triples(response.edges),
            {("Aurelia Maren", "PRAISES", "Elias Nyberg"), ("Aurelia Maren", "REFERENCES", "Demo Story")},
        )

    def test_walk_reaches_nodes_beyond_the_first_hop(self):
        body = main.LLMWalkRequest(
            prompt="Where did the people around Aurelia Maren meet?",
            character="Aurelia Maren",
            character_key=CHARACTER_KEY,
            max_steps=2,
        )
        targets = [["Elias Nyberg"], ["Harbor Circle"]]
        with mock.patch.object(main, "choose_walk_targets", side_effect=targets):
            response = main.llm_walk(body, session=self.session)

        self.assertEqual(
            [(step.expand_nodes, step.added_nodes, step.added_edges) for step in response.steps],
            [(["Elias Nyberg"], 1, 1), (["Harbor Circle"], 1, 1)],
        )
        names = node_names(response.nodes)
        self.assertEqual(sorted(names), ["Aurelia Maren", "Demo Story", "Elias Nyberg", "Harbor Circle", "Porto"])
        self.assertEqual(len(names), len({n["id"] for n in response.nodes}), "each node is listed once")
        self.assertEqual(
            edge_triples(response.edges),
            set(CHAIN) | {("Aurelia Maren", "REFERENCES", "Demo Story")},
        )

    def test_segment_nodes_and_edges_are_not_graph_facts(self):
        # Expanding a Work must not bring in its SourceSegment nodes or HAS_SEGMENT edges.
        body = main.LLMWalkRequest(prompt="What is Demo Story?", character="Demo Story", character_key=CHARACTER_KEY, max_steps=1)
        with mock.patch.object(main, "choose_walk_targets", side_effect=[["Demo Story"]]):
            response = main.llm_walk(body, session=self.session)

        self.assertEqual(sorted(node_names(response.nodes)), ["Aurelia Maren", "Demo Story"])
        self.assertEqual(edge_triples(response.edges), {("Aurelia Maren", "REFERENCES", "Demo Story")})

        # With no focus name the context query matches everything in the partition.
        nodes, edges = main.fetch_context_neo4j(None, "", character_key=CHARACTER_KEY)
        self.assertNotIn(None, node_names(nodes))
        self.assertNotIn("HAS_SEGMENT", {e["relation"] for e in edges})


if __name__ == "__main__":
    unittest.main()
