"""
The API flow against real databases: ingest the sample story over HTTP, query it, and
walk the graph past the first hop.

Skipped unless TEST_DATABASE_URL names a PostgreSQL database the test may write to:

    TEST_DATABASE_URL=postgresql+psycopg2://kg:kgpass@localhost:5432/kg python -m pytest tests/test_integration.py

(`docker compose up -d db` provides that database.) With TEST_NEO4J_URI, TEST_NEO4J_USER and
TEST_NEO4J_PASSWORD set as well, ingestion also writes to Neo4j and /query and /llm-walk read
from it, so the same assertions then cover the Neo4j path.

Extraction uses MockExtractor and no LLM is called: the walk's choice of nodes to expand is
scripted. The test writes under its own character_key and work names and deletes them at the end.
"""

import os
import unittest
import uuid
from pathlib import Path
from unittest import mock

from sqlalchemy import delete, select

from kg.schema import KGBasicNode, KGEdge, SourceSegment, create_engine_and_session

# app.main opens a database connection at import time when DATABASE_URL is set. This test
# connects to TEST_DATABASE_URL instead, so hide the variable while the module loads.
with mock.patch.dict(os.environ):
    os.environ.pop("DATABASE_URL", None)
    import app.main as main

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
TEST_NEO4J_URI = os.getenv("TEST_NEO4J_URI")
SAMPLE_TEXT = Path(__file__).resolve().parent.parent / "test_files" / "demo_story.txt"
SAMPLE_PDF = SAMPLE_TEXT.with_suffix(".pdf")


def looks_like_uuid(value) -> bool:
    try:
        uuid.UUID(str(value))
    except ValueError:
        return False
    return True


@unittest.skipUnless(TEST_DATABASE_URL, "set TEST_DATABASE_URL to run the integration test")
class TestIngestQueryWalk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        suffix = uuid.uuid4().hex[:8]
        cls.character_key = f"itest-{suffix}"
        cls.letters = f"demo-story-{suffix}"
        cls.memoir = f"demo-story-pdf-{suffix}"
        cls.use_neo4j = bool(TEST_NEO4J_URI)

        # The app reads its settings from the environment: no LLM key (mock extraction, no
        # answers), and Neo4j only when the TEST_NEO4J_* variables ask for it.
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY" and not k.startswith("NEO4J_")}
        if cls.use_neo4j:
            env["NEO4J_URI"] = TEST_NEO4J_URI
            env["NEO4J_USER"] = os.environ["TEST_NEO4J_USER"]
            env["NEO4J_PASSWORD"] = os.environ["TEST_NEO4J_PASSWORD"]

        cls.engine, cls.SessionLocal = create_engine_and_session(TEST_DATABASE_URL)
        patches = [
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(main, "SessionLocal", cls.SessionLocal),
            mock.patch.object(main, "neo4j_driver", None),
            mock.patch.object(main, "neo4j_builder", None),
        ]
        for patch in patches:
            patch.start()
            cls.addClassCleanup(patch.stop)
        cls.addClassCleanup(cls.delete_test_data)

        cls.client = TestClient(main.app)
        cls.ingests = [
            cls.ingest(SAMPLE_TEXT, work_name=cls.letters, character="Aurelia Maren"),
            cls.ingest(SAMPLE_TEXT, work_name=cls.letters, character="Elias Nyberg"),
            cls.ingest(SAMPLE_PDF, work_name=cls.memoir, character="Elias Nyberg"),
        ]

    @classmethod
    def ingest(cls, path: Path, work_name: str, character: str) -> dict:
        with path.open("rb") as handle:
            response = cls.client.post(
                "/ingest-book",
                params={"character": character, "character_key": cls.character_key, "work_name": work_name},
                files={"file": (path.name, handle)},
            )
        assert response.status_code == 200, response.text
        return response.json()

    @classmethod
    def delete_test_data(cls):
        with cls.SessionLocal() as session:
            node_ids = session.scalars(
                select(KGBasicNode.id).where(KGBasicNode.character_key == cls.character_key)
            ).all()
            if node_ids:
                session.execute(delete(SourceSegment).where(SourceSegment.work_id.in_(node_ids)))
            session.execute(delete(KGEdge).where(KGEdge.character_key == cls.character_key))
            session.execute(delete(KGBasicNode).where(KGBasicNode.character_key == cls.character_key))
            session.commit()
        cls.engine.dispose()
        if main.neo4j_driver is not None:
            with main.neo4j_driver.session() as session:
                session.run("MATCH (n {character_key: $ck}) DETACH DELETE n", ck=cls.character_key).consume()
            main.neo4j_driver.close()

    # The mock extractor links each focus character to the work it was ingested with:
    #   Aurelia Maren -> letters <- Elias Nyberg -> memoir
    # so from Aurelia Maren the memoir is three hops away.

    def test_ingest_reports_what_was_extracted(self):
        first, second, third = self.ingests
        # The sample text has four paragraphs: one segment, one Work candidate and one edge each.
        self.assertEqual(first["segments"], 4)
        self.assertEqual(first["nodes"], 5)
        self.assertEqual(first["edges"], 4)
        self.assertEqual(first["work_name"], self.letters)
        self.assertEqual(second["segments"], 4)
        self.assertGreaterEqual(third["segments"], 1)
        self.assertEqual(third["edges"], third["segments"])
        for result in self.ingests:
            self.assertEqual(result["published_to_neo4j"], self.use_neo4j)

    def test_query_returns_the_focus_node_and_named_edges(self):
        response = self.client.post(
            "/query",
            json={"prompt": "What did Aurelia Maren write?", "character": "Aurelia Maren", "character_key": self.character_key},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        self.assertEqual([n["name"] for n in body["nodes"]], ["Aurelia Maren"])
        self.assertEqual(body["nodes"][0]["type"], "Character")
        self.assertEqual(len(body["edges"]), 1)
        edge = body["edges"][0]
        self.assertEqual((edge["from"], edge["relation"], edge["to"]), ("Aurelia Maren", "REFERENCES", self.letters))
        # The edge cites the four segments of the text it came from.
        self.assertEqual(len(set(edge["source_ids"])), 4)
        self.assertIsNone(body["answer"])

    def test_walk_without_an_llm_returns_the_starting_context(self):
        response = self.client.post(
            "/llm-walk",
            json={"prompt": "What did Aurelia Maren write?", "character": "Aurelia Maren", "character_key": self.character_key},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        self.assertEqual(body["steps"], [])
        self.assertEqual({n["name"] for n in body["nodes"]}, {"Aurelia Maren", self.letters})
        self.assertEqual(len(body["edges"]), 1)

    def test_walk_goes_past_the_first_hop(self):
        targets = [[self.letters], ["Elias Nyberg"]]
        with mock.patch.object(main, "choose_walk_targets", side_effect=targets):
            response = self.client.post(
                "/llm-walk",
                json={
                    "prompt": "What else did the people around Aurelia Maren write?",
                    "character": "Aurelia Maren",
                    "character_key": self.character_key,
                    "max_steps": 2,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        self.assertEqual(
            body["steps"],
            [
                {"expand_nodes": [self.letters], "added_nodes": 1, "added_edges": 1},
                {"expand_nodes": ["Elias Nyberg"], "added_nodes": 1, "added_edges": 1},
            ],
        )
        self.assertEqual(
            sorted(n["name"] for n in body["nodes"]),
            sorted(["Aurelia Maren", "Elias Nyberg", self.letters, self.memoir]),
        )
        self.assertEqual(
            {(e["from"], e["relation"], e["to"]) for e in body["edges"]},
            {
                ("Aurelia Maren", "REFERENCES", self.letters),
                ("Elias Nyberg", "REFERENCES", self.letters),
                ("Elias Nyberg", "REFERENCES", self.memoir),
            },
        )
        for edge in body["edges"]:
            for end in (edge["from"], edge["to"]):
                self.assertTrue(end, edge)
                self.assertFalse(looks_like_uuid(end), edge)


if __name__ == "__main__":
    unittest.main()
