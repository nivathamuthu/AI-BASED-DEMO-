"""What the extraction and GraphRAG prompts must return."""
from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["Aircraft", "Component", "Task", "Regulation", "Procedure", "Part", "Interval"]
RelationType = Literal["APPLIES_TO", "REQUIRES", "GOVERNED_BY", "REPLACES", "HAS_INTERVAL", "PART_OF"]

ENTITY_TYPES: tuple[str, ...] = EntityType.__args__
RELATION_TYPES: tuple[str, ...] = RelationType.__args__


class Entity(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    type: EntityType
    aliases: list[str] = Field(default_factory=list, max_length=10)


class Relation(BaseModel):
    source: str = Field(min_length=1, max_length=200)
    relation: RelationType
    target: str = Field(min_length=1, max_length=200)
    evidence_quote: str = Field(min_length=1, max_length=500)


class KGOutput(BaseModel):
    entities: list[Entity] = Field(default_factory=list, max_length=60)
    relations: list[Relation] = Field(default_factory=list, max_length=80)


class GraphAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=800)
    known: bool
    fact_indexes: list[int] = Field(default_factory=list)
    citation_indexes: list[int] = Field(default_factory=list)
