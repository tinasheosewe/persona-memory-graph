import os
import uuid
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class KGBasicNode(Base):
    __tablename__ = "kg_nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    character_key = Column(String, nullable=False, default="default")
    type = Column(
        String,
        nullable=False,
        doc="One of Character, Person, Event, Concept, Work, Place, Period, SourceSegment",
    )
    name = Column(String, nullable=False)
    alias_names = Column(ARRAY(String), default=list)
    summary = Column(Text)
    meta = Column(JSONB, default=dict)
    source_ids = Column(ARRAY(String), default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    edges_from = relationship(
        "KGEdge",
        foreign_keys="KGEdge.from_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    edges_to = relationship(
        "KGEdge",
        foreign_keys="KGEdge.to_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<KGBasicNode {self.type}:{self.name}>"


class KGEdge(Base):
    __tablename__ = "kg_edges"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    character_key = Column(String, nullable=False, default="default")
    from_id = Column(UUID(as_uuid=True), ForeignKey("kg_nodes.id", ondelete="CASCADE"))
    to_id = Column(UUID(as_uuid=True), ForeignKey("kg_nodes.id", ondelete="CASCADE"))
    relation = Column(String, nullable=False)
    weight = Column(Float)
    description = Column(Text)
    source_ids = Column(ARRAY(String), default=list)
    meta = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    from_node = relationship("KGBasicNode", foreign_keys=[from_id])
    to_node = relationship("KGBasicNode", foreign_keys=[to_id])

    def __repr__(self) -> str:
        return f"<KGEdge {self.from_id} -[{self.relation}]-> {self.to_id}>"


class SourceSegment(Base):
    __tablename__ = "source_segments"

    id = Column(String, primary_key=True)
    work_id = Column(UUID(as_uuid=True), ForeignKey("kg_nodes.id"))
    location = Column(String)
    content = Column(Text, nullable=False)
    meta = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    work = relationship("KGBasicNode", foreign_keys=[work_id])

    def __repr__(self) -> str:
        return f"<SourceSegment {self.id}>"


def create_engine_and_session(
    database_url: Optional[str] = None,
    echo: bool = False,
):
    """Create engine and session factory. Pass DATABASE_URL or use env."""
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set")

    engine = create_engine(url, echo=echo, future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return engine, SessionLocal
