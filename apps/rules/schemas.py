"""What the rule extraction and conflict judge prompts must return."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ExtractedRule(BaseModel):
    condition: str = Field(min_length=1, max_length=500)
    action: str = Field(min_length=1, max_length=500)
    exception: str = Field(default="", max_length=500)
    source_quote: str = Field(min_length=1, max_length=800)
    confidence: float = Field(ge=0, le=1)


class RuleSet(BaseModel):
    rules: list[ExtractedRule] = Field(default_factory=list, max_length=25)


class ConflictJudgment(BaseModel):
    conflict: bool
    kind: Literal["threshold", "action", "none"]
    severity: Literal["low", "medium", "high"]
    reason: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def kind_matches_verdict(self):
        if self.conflict == (self.kind == "none"):
            raise ValueError("kind must be 'none' exactly when conflict is false")
        return self
