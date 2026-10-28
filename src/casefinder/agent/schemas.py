"""Structured outputs the model must return, and the agent's final result."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Sentence(BaseModel):
    section: Literal["findings", "impression"]
    text: str = Field(description="One sentence of the radiology report.")
    sources: list[str] = Field(
        description='Evidence for this sentence: "image" if it is visible in the uploaded X-ray, '
        'and/or "case:<uid>" for each similar case whose report supports it.'
    )


class Report(BaseModel):
    sentences: list[Sentence]


class Verdict(BaseModel):
    index: int = Field(description="0-based index of the sentence being judged.")
    supported: bool
    reason: str = Field(description="One short sentence explaining the verdict.")


class Verification(BaseModel):
    verdicts: list[Verdict]


class CheckItem(BaseModel):
    finding: str = Field(description="The checklist finding, exactly as named in the checklist.")
    present: Literal["yes", "no", "uncertain"] = Field(description="Whether THIS image shows the finding.")
    in_draft: bool = Field(description="True if the draft already states this finding correctly.")
    evidence: str = Field(description="One short phrase: what/where in the image, or why it is absent.")


class LookAgain(BaseModel):
    items: list[CheckItem]


class CheckedSentence(Sentence):
    supported: bool
    reason: str


class ReportResult(BaseModel):
    """What the agent returns. `report` holds only sentences that passed verification."""

    report: list[CheckedSentence]
    removed: list[CheckedSentence]
    cases: list[dict]
    revisions: int
    first_pass_supported: float = Field(description="Share of sentences supported in the first draft.")
    first_draft: list[Sentence] = Field(default_factory=list,
                                        description="The draft before look-again, verification and revision.")
    missed_findings: list[CheckItem] = Field(default_factory=list,
                                             description="Findings the look-again step found missing from the draft.")
    prompt_versions: dict[str, str]
    model: str
