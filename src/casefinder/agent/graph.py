"""LangGraph workflow: retrieve -> draft -> look_again -> [amend] -> check -> (revise -> check)* -> finalize.

`look_again` is a second reader: it goes through a checklist (always-check findings plus the findings the
similar cases had) and reports findings the draft missed; `amend` adds them. It targets omissions, which the
sentence-level verifier cannot see. Disable with FINDXRAY_LOOK_AGAIN=0.

`check` combines a deterministic citation check (every source must be "image" or a retrieved
case) with the model's sentence-by-sentence verification. A sentence passes only if both agree.
"""

from __future__ import annotations

import os
import re
from typing import TypedDict

import numpy as np
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from casefinder.agent.model import ReportModel, XRayImage
from casefinder.agent.schemas import CheckedSentence, CheckItem, Report, ReportResult, Verdict
from casefinder.search import CaseLibrary

MAX_REVISIONS = 2
_CASE_SOURCE = re.compile(r"^case:(\d+)$")


class State(TypedDict, total=False):
    # The image itself is not part of the state (it is passed to the nodes by closure), so the
    # graph's traced inputs/outputs stay small; the model calls still receive it.
    embedding: list[float]
    k: int
    cases: list[dict]
    report: Report
    first_draft: Report
    missed: list[CheckItem]
    verdicts: list[Verdict]
    revisions: int
    first_pass_supported: float


def citation_problems(report: Report, case_uids: set[int]) -> dict[int, str]:
    """Sentence index -> problem, for sources that are missing or point outside the retrieved cases."""
    problems = {}
    for i, s in enumerate(report.sentences):
        if not s.sources:
            problems[i] = "no sources cited"
            continue
        for src in s.sources:
            src = src.strip().lower()
            m = _CASE_SOURCE.match(src)
            if src != "image" and not (m and int(m.group(1)) in case_uids):
                problems[i] = f"cites '{src}', which is not the image or a retrieved case"
                break
    return problems


def merge_verdicts(report: Report, model_verdicts: list[Verdict], problems: dict[int, str]) -> list[Verdict]:
    by_index = {v.index: v for v in model_verdicts if 0 <= v.index < len(report.sentences)}
    merged = []
    for i in range(len(report.sentences)):
        if i in problems:
            merged.append(Verdict(index=i, supported=False, reason=problems[i]))
        elif i not in by_index:
            merged.append(Verdict(index=i, supported=False, reason="not judged by the verifier"))
        else:
            merged.append(by_index[i])
    return merged


def look_again_enabled() -> bool:
    return os.environ.get("FINDXRAY_LOOK_AGAIN", "1") not in ("0", "false", "False", "")


def build_graph(library: CaseLibrary, model: ReportModel, image: XRayImage):
    def retrieve(state: State) -> State:
        hits = library.search(np.asarray(state["embedding"], dtype=np.float32), k=state.get("k", 5))
        return {"cases": [{**h.case, "similarity": h.similarity, "rank": h.rank} for h in hits]}

    def draft(state: State, config: RunnableConfig) -> State:
        report = model.draft(image, state["cases"], config)
        return {"report": report, "first_draft": report, "revisions": 0}

    def look_again(state: State, config: RunnableConfig) -> State:
        if not state["report"].sentences:
            return {"missed": []}
        items = model.look_again(image, state["cases"], state["report"], config).items
        return {"missed": [i for i in items if i.present in ("yes", "uncertain") and not i.in_draft]}

    def amend(state: State, config: RunnableConfig) -> State:
        return {"report": model.amend(image, state["cases"], state["report"], state["missed"], config)}

    def after_look_again(state: State) -> str:
        return "amend" if state.get("missed") else "check"

    def check(state: State, config: RunnableConfig) -> State:
        report = state["report"]
        if not report.sentences:
            return {"verdicts": [], **({} if "first_pass_supported" in state else {"first_pass_supported": 0.0})}
        problems = citation_problems(report, {c["uid"] for c in state["cases"]})
        verdicts = merge_verdicts(report, model.verify(image, state["cases"], report, config).verdicts,
                                  problems)
        update: State = {"verdicts": verdicts}
        if "first_pass_supported" not in state:
            update["first_pass_supported"] = sum(v.supported for v in verdicts) / len(verdicts)
        return update

    def revise(state: State, config: RunnableConfig) -> State:
        new = model.revise(image, state["cases"], state["report"], state["verdicts"], config)
        return {"report": new, "revisions": state["revisions"] + 1}

    def route(state: State) -> str:
        if all(v.supported for v in state["verdicts"]) or state["revisions"] >= MAX_REVISIONS:
            return END
        return "revise"

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("draft", draft)
    g.add_node("check", check)
    g.add_node("revise", revise)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "draft")
    if look_again_enabled():
        g.add_node("look_again", look_again)
        g.add_node("amend", amend)
        g.add_edge("draft", "look_again")
        g.add_conditional_edges("look_again", after_look_again, {"amend": "amend", "check": "check"})
        g.add_edge("amend", "check")
    else:
        g.add_edge("draft", "check")
    g.add_conditional_edges("check", route, {"revise": "revise", END: END})
    g.add_edge("revise", "check")
    return g.compile()


def to_result(state: State, model: ReportModel) -> ReportResult:
    checked = [CheckedSentence(**s.model_dump(), supported=v.supported, reason=v.reason)
               for s, v in zip(state["report"].sentences, state["verdicts"])]
    return ReportResult(
        report=[c for c in checked if c.supported],
        removed=[c for c in checked if not c.supported],
        cases=state["cases"],
        revisions=state["revisions"],
        first_pass_supported=state.get("first_pass_supported", 0.0),
        first_draft=list(state["first_draft"].sentences),
        missed_findings=list(state.get("missed", [])),
        prompt_versions=dict(model.prompt_versions),
        model=model.name,
    )


def stream_report(library: CaseLibrary, model: ReportModel, image: XRayImage, embedding, k: int = 5,
                  callbacks: list | None = None):
    """Run the agent, yielding each finished step's name ("retrieve", "draft", ...) and finally the ReportResult."""
    graph = build_graph(library, model, image)
    config: RunnableConfig = {"callbacks": callbacks or [], "run_name": "findxray-report",
                              "recursion_limit": 20}
    state: State = {"embedding": list(map(float, embedding)), "k": k}
    for chunk in graph.stream(dict(state), config=config, stream_mode="updates"):
        for node, update in chunk.items():
            state.update(update or {})
            yield node
    yield to_result(state, model)


def run_report(library: CaseLibrary, model: ReportModel, image: XRayImage, embedding, k: int = 5,
               callbacks: list | None = None) -> ReportResult:
    *_, result = stream_report(library, model, image, embedding, k, callbacks)
    return result
