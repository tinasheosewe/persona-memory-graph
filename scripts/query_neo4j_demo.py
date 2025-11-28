"""
Query Neo4j for a character_key and ask the LLM to answer based on graph context.

Requires:
  - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
  - OPENAI_API_KEY (for LLM answer)

Usage:
  python scripts/query_neo4j_demo.py --character_key default --prompt "Who influenced Aurelia?"
"""

import argparse
import os

from neo4j import GraphDatabase
from openai import OpenAI


def fetch_context(driver, character_key, name_filter=None):
    node_query = """
    MATCH (n {character_key: $ck})
    WHERE $name IS NULL OR n.name CONTAINS $name
    RETURN n LIMIT 200
    """
    edge_query = """
    MATCH (a {character_key: $ck})-[r]->(b {character_key: $ck})
    WHERE ($name IS NULL OR a.name CONTAINS $name OR b.name CONTAINS $name)
    RETURN a.name AS from_name, b.name AS to_name, type(r) AS relation, r.description AS description, r.meta AS meta LIMIT 300
    """
    with driver.session() as session:
        nodes = [
            {
                "type": record["n"].get("type"),
                "name": record["n"].get("name"),
                "summary": record["n"].get("summary"),
                "aliases": record["n"].get("alias_names", []),
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
            }
            for record in session.run(edge_query, ck=character_key, name=name_filter)
        ]
    return nodes, edges


def ask_llm(prompt, nodes, edges):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
    )
    return res.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character_key", default="default")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--name", help="Optional name filter")
    args = parser.parse_args()

    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]

    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes, edges = fetch_context(driver, args.character_key, args.name)
    print(f"Fetched {len(nodes)} nodes, {len(edges)} edges")
    answer = ask_llm(args.prompt, nodes, edges)
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
