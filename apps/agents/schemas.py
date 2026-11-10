"""What each LLM node of the compliance graph must return."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Intent = Literal["lookup", "compliance_check", "conflict_check", "out_of_scope"]
ToolName = Literal["hybrid_search", "kg_query", "rule_lookup"]


class Classification(BaseModel):
    intent: Intent
    entities: list[str] = Field(default_factory=list, max_length=12)
    reason: str = Field(min_length=1, max_length=400)


class Plan(BaseModel):
    tools: list[ToolName] = Field(min_length=1, max_length=3)
    search_query: str = Field(min_length=3, max_length=300)
    reason: str = Field(min_length=1, max_length=400)

    @field_validator("tools")
    @classmethod
    def _search_first_and_unique(cls, tools: list[str]) -> list[str]:
        if len(set(tools)) != len(tools):
            raise ValueError("list each tool at most once")
        if "hybrid_search" not in tools:
            raise ValueError("hybrid_search is required: the answer may only cite retrieved chunks")
        return tools


class RuleAssessment(BaseModel):
    rule_key: str = Field(pattern=r"^R-\d{4}$")
    applies: Literal["yes", "no", "unclear"] = Field(description="Is the rule's condition met by the stated facts?")
    relevant: bool = Field(description="Does the rule's outcome bear on the answer, even when its condition is not met?")
    reason: str = Field(min_length=1, max_length=400)


class RuleCheck(BaseModel):
    facts: list[str] = Field(default_factory=list, max_length=10)
    assessments: list[RuleAssessment] = Field(default_factory=list, max_length=20)


class Claim(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    source: int = Field(ge=1)
    quote: str = Field(min_length=8, max_length=600)


class Draft(BaseModel):
    status: Literal["answered", "cannot_determine"]
    answer: str = Field(min_length=1, max_length=2000)
    claims: list[Claim] = Field(default_factory=list, max_length=8)
    conflict_note: str = Field(default="", max_length=800)

    @model_validator(mode="after")
    def _answered_needs_claims(self):
        if self.status == "answered" and not self.claims:
            raise ValueError("an answered draft needs at least one claim with a source and a verbatim quote")
        return self


class ClaimVerdict(BaseModel):
    claim: int = Field(ge=1)
    supported: bool
    reason: str = Field(min_length=1, max_length=400)


class Verification(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list, max_length=8)
    refined_query: str = Field(default="", max_length=300)
