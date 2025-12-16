from typing import Dict, List, Literal, Optional, get_args

from pydantic import BaseModel, Field

NodeType = Literal[
    "Character",
    "Person",
    "Event",
    "Concept",
    "Organization",
    "Work",
    "Place",
    "Period",
    "SourceSegment",
    "Episode",
    "Principle",
]

NODE_TYPE_VALUES: tuple[str, ...] = get_args(NodeType)

RelationLiteral = Literal[
    "RELATES_TO",
    "IS",
    "BECOMES",
    "REMAINS",
    "TRANSFORMS_INTO",
    "EMBODIES",
    "REPRESENTS",
    "SYMBOLIZES",
    "CONSTITUTES",
    "CONTAINS",
    "PART_OF",
    "PRECEDES",
    "FOLLOWS",
    "COINCIDES_WITH",
    "CAUSES",
    "ENABLES",
    "PREVENTS",
    "TRIGGERS",
    "RESULTS_IN",
    "INTERRUPTS",
    "ACCELERATES",
    "BELIEVES",
    "KNOWS",
    "ASSUMES",
    "DOUBTS",
    "QUESTIONS",
    "INTENDS",
    "DESIRES",
    "FEARS",
    "EXPECTS",
    "REGRETS",
    "CHOOSES",
    "DECIDES_AGAINST",
    "ACTS_ON",
    "REACTS_TO",
    "INITIATES",
    "ABANDONS",
    "PURSUES",
    "AVOIDS",
    "COMMITS_TO",
    "WITHDRAWS_FROM",
    "INFLUENCES",
    "SHAPES",
    "CONTROLS",
    "CONSTRAINS",
    "EMPOWERS",
    "UNDERMINES",
    "MANIPULATES",
    "RESISTS",
    "DOMINATES",
    "DEPENDS_ON",
    "SUPPORTS",
    "OPPOSES",
    "ALLIES_WITH",
    "BETRAYS",
    "TRUSTS",
    "DISTRUSTS",
    "OBEYS",
    "DEFIES",
    "LEADS",
    "STATES",
    "CLAIMS",
    "ARGUES",
    "DENIES",
    "ADMITS",
    "PROMISES",
    "WARNS",
    "CONFESSES",
    "IMPLIES",
    "CONCEALS",
    "VALUES",
    "DEVALUES",
    "PRAISES",
    "CRITICIZES",
    "APPROVES",
    "REJECTS",
    "JUSTIFIES",
    "CONDEMNS",
    "RATIONALIZES",
    "PRIORITIZES",
    "REFERENCES",
    "CITES",
    "DERIVES_FROM",
    "DERIVED_FROM",
    "CONTRADICTS",
    "CONFIRMS",
    "MISINTERPRETS",
    "CLARIFIES",
    "SUMMARIZES",
    "EXPLAINS",
    "QUESTIONS_VALIDITY_OF",
    "EVIDENCED_BY",
    "CREATES",
    "DESTROYS",
    "OWNS",
    "USES",
    "ABUSES",
    "PROTECTS",
    "SACRIFICES",
    "INHERITS",
    "TRANSMITS",
    "ATTRIBUTED_TO",
    "APPLIES_TO",
    "EXCEPTION_OF",
    "INSPIRED_BY",
]

RELATION_VALUES: tuple[str, ...] = get_args(RelationLiteral)


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
    character_key: Optional[str] = None


class EdgeCandidate(BaseModel):
    from_name: str
    to_name: str
    relation: RelationLiteral
    description: Optional[str] = None
    source_id: Optional[str] = None
    confidence: Optional[float] = None
    meta: Dict = Field(default_factory=dict)
    character_key: Optional[str] = None
