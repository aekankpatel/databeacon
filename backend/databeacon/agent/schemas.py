"""Structured-output schemas used by the agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    id: str = Field(description="Short id such as H1")
    statement: str = Field(description="Falsifiable hypothesis")
    approach: str = Field(default="", description="Which tools / queries will test it")


class Plan(BaseModel):
    objective: str = Field(description="Restated analytical objective")
    key_metric: str = Field(default="", description="Primary metric column or expression")
    time_scope: str = Field(default="", description="Relevant periods with concrete ISO dates, if any")
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list, description="Ordered investigation steps")


class Reflection(BaseModel):
    sufficient: bool = Field(description="True if the evidence answers the question and tests the main hypotheses")
    confidence: float = Field(default=0.5, ge=0, le=1)
    hypothesis_status: dict[str, str] = Field(default_factory=dict,
                                              description="hypothesis id -> supported|rejected|inconclusive|untested")
    gaps: list[str] = Field(default_factory=list, description="What is still unknown")
    next_actions: list[str] = Field(default_factory=list, description="Concrete next analyses if not sufficient")


class Finding(BaseModel):
    title: str
    statement: str = Field(description="One or two sentences with specific numbers copied from the evidence")
    evidence_ids: list[str] = Field(default_factory=list, description="Evidence ids (E1, E2, ...) supporting it")
    confidence: str = Field(default="medium", description="high|medium|low")


class Synthesis(BaseModel):
    title: str = Field(default="", description="Short report title")
    summary: str = Field(description="Direct answer to the question in 3-5 sentences")
    findings: list[Finding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
