import unittest

from kg.neo4j_memgraph import CypherGraphBuilder
from kg.pipeline import MockExtractor


class FakeTx:
    def __init__(self, log):
        self.log = log

    def run(self, query, **params):
        self.log.append((query, params))


class RecordingSession:
    def __init__(self):
        self.tx_log = []

    def execute_write(self, func, *args, **kwargs):
        tx = FakeTx(self.tx_log)
        func(tx, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class RecordingDriver:
    def __init__(self):
        self.sessions = []

    def session(self):
        s = RecordingSession()
        self.sessions.append(s)
        return s


class TestCypherGraphBuilder(unittest.TestCase):
    def test_build_from_text_emits_cypher(self):
        driver = RecordingDriver()
        extractor = MockExtractor()
        builder = CypherGraphBuilder(driver=driver, extractor=extractor)

        text = "He wrote letters about duty."
        builder.build_from_text(text, work_name="Letters", character="Marcus")

        # Collect all executed queries
        queries = [q for session in driver.sessions for (q, _params) in session.tx_log]
        self.assertTrue(any("Work" in q for q in queries), "Should merge Work nodes")
        self.assertTrue(any("HAS_SEGMENT" in q for q in queries), "Should create HAS_SEGMENT relationships")
        self.assertTrue(any("MERGE (n:" in q for q in queries), "Should merge entity nodes")
        self.assertTrue(any("MERGE (a)-[r:" in q for q in queries), "Should merge relationships")


if __name__ == "__main__":
    unittest.main()
