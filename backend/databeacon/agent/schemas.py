"""Structured-output schemas used by the agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    id: str = Field(description="Short id such as H1")
    statement: str = Field(description="Falsifiable hypothesis")
    approach: str = Field(default="", description="Which tools / queries will test it")


class Plan(BaseModel):
    objective: str = Field(description="Restated analytical objective")
    question_type: str = Field(default="other", description="change_attribution | group_comparison | ranking | "
                                                            "trend | data_quality | relationship | other")
    metric_column: str = Field(default="", description="The column the question's metric is computed from")
    aggregation: str = Field(default="", description="sum | avg | count | rate | median | ... applied to metric_column")
    group_by: list[str] = Field(default_factory=list, description="Columns the question asks to break results down by")
    answer_target: str = Field(default="", description="The single thing the answer must identify, e.g. "
                                                       "'customer_segment with the largest revenue decline'")
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


class Answer(BaseModel):
    primary_entity: str | None = Field(default=None, description="The main conclusion value, e.g. 'Enterprise', "
                                                                 "'West', a month like '2025-10', or a column name")
    dimension: str | None = Field(default=None, description="Column the primary_entity is a value of (None if n/a)")
    metric_column: str | None = Field(default=None, description="Column the conclusion is measured on")
    aggregation: str | None = None
    direction: str | None = Field(default=None, description="largest_decline | largest_increase | highest | lowest | "
                                                            "change_point | other")
    current_period: list[str] | None = Field(default=None, description="[start, end_exclusive] ISO dates analysed")
    previous_period: list[str] | None = Field(default=None, description="[start, end_exclusive] comparison period")
    key_value: float | None = Field(default=None, description="The headline number behind the conclusion")
    evidence_ids: list[str] = Field(default_factory=list, description="Evidence ids that establish the conclusion")


class Synthesis(BaseModel):
    title: str = Field(default="", description="Short report title")
    summary: str = Field(description="Direct answer to the question in 3-5 sentences")
    answer: Answer = Field(default_factory=Answer, description="Machine-checkable form of the main conclusion")
    findings: list[Finding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
