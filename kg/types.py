from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

NodeType = Literal[
    "Character",
    "Person",
    "Event",
    "Concept",
    "Work",
    "Place",
    "Period",
    "SourceSegment",
]


class Segment(BaseModel):
    id: str
    work_name: str
    location: Optional[str] = None
    content: str
    meta: Dict = Field(default_factory=dict)


class NodeCandidate(BaseModel):
    name: str
    type: NodeType
    summary: Optional[str] = None
    alias_names: List[str] = Field(default_factory=list)
    meta: Dict = Field(default_factory=dict)
    source_id: Optional[str] = None


class EdgeCandidate(BaseModel):
    from_name: str
    to_name: str
    relation: str
    description: Optional[str] = None
    source_id: Optional[str] = None
    confidence: Optional[float] = None
