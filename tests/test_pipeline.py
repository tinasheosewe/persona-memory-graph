import json
import unittest
from dataclasses import dataclass

from kg.pipeline import (
    GraphBuilder,
    LLMExtractor,
    MockExtractor,
    segment_text,
    slugify,
)
from kg.types import Segment


class TestPipeline(unittest.TestCase):
    def test_slugify_basic(self):
        self.assertEqual(slugify("Hello World!"), "hello-world")
        self.assertEqual(slugify("Already-clean"), "already-clean")

    def test_segment_text_respects_word_budget_sentence_safe(self):
        sentences = [
            "alpha beta gamma delta epsilon.",
            "zeta eta theta iota kappa.",
            "lambda mu nu xi omicron.",
            "pi rho sigma tau upsilon.",
            "phi chi psi omega omega.",
            "one two three four five.",
        ]
        text = " ".join(sentences)
        segments = segment_text(
            text,
            work_name="Test Work",
            max_words=12,
            overlap_sentences=1,
        )
        # Expect multiple chunks, none exceeding 12 words, and overlap present
        self.assertTrue(len(segments) >= 4)
        self.assertTrue(all(len(seg.content.split()) <= 12 for seg in segments))
        self.assertTrue(segments[0].id.startswith("test-work-0-0"))
        if len(segments) >= 2:
            first_last_sentence = segments[0].content.strip().split(".")[-2].strip() + "."
            self.assertIn(first_last_sentence, segments[1].content)

    def test_llm_extractor_parses_json(self):
        payload = {
            "nodes": [{"name": "Marcus", "type": "Character"}],
            "edges": [{"from_name": "Marcus", "to_name": "Book", "relation": "REFERENCES"}],
        }
        extractor = LLMExtractor(client=FakeClient(payload), model="dummy")
        segments = [Segment(id="s1", work_name="Book", content="hi")]
        result = extractor.extract(segments)
        self.assertEqual(result.nodes[0].name, "Marcus")
        self.assertEqual(result.edges[0].relation, "REFERENCES")

    def test_graph_builder_flow_records_calls(self):
        extractor = MockExtractor()
        builder = RecordingBuilder(extractor=extractor)
        text = "A short paragraph about duty."
        builder.build_from_text(text, work_name="Letters", character="Marcus")
        self.assertTrue(builder.saved_segments, "Segments should be stored")
        self.assertTrue(builder.saved_nodes, "Nodes should be upserted")
        self.assertTrue(builder.saved_edges, "Edges should be upserted")
        self.assertTrue(any(n.name == "Marcus" for n in builder.saved_nodes))
        self.assertTrue(all(e.relation for e in builder.saved_edges))


class FakeChoice:
    def __init__(self, content: str):
        self.message = type("M", (), {"content": content})


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.chat = type(
            "Chat",
            (),
            {
                "completions": type(
                    "Completions",
                    (),
                    {
                        "create": lambda *_args, **_kwargs: FakeResponse(
                            json.dumps(self.payload)
                        )
                    },
                )()
            },
        )()


@dataclass
class StubWork:
    id: str
    name: str


class RecordingBuilder(GraphBuilder):
    def __init__(self, extractor):
        self.saved_segments = []
        self.saved_nodes = []
        self.saved_edges = []
        super().__init__(session=type("S", (), {"commit": lambda self: None})(), extractor=extractor)

    def _ensure_work_node(self, work_name, work_meta):
        return StubWork(id="work-1", name=work_name)

    def _store_segments(self, segments, work_id):
        self.saved_segments.extend(segments)

    def _upsert_nodes(self, nodes):
        self.saved_nodes.extend(nodes)

    def _upsert_edges(self, edges):
        self.saved_edges.extend(edges)


if __name__ == "__main__":
    unittest.main()
