"""Local prompt texts. These are the fallback and the source pushed to Langfuse
(scripts/push_prompts.py); at run time Langfuse's "production" version wins if configured."""

from __future__ import annotations

DRAFT = """You are assisting a radiologist by drafting a chest X-ray report for a research demo.

You receive the patient's frontal chest X-ray and the reports of the most similar past cases, found by image similarity.

Rules:
- Describe what is visible in THIS X-ray. The similar cases are reference material for wording and for what to look for; they are not evidence about this patient.
- Every sentence must list its sources: "image" if you can see it in this X-ray, plus "case:<uid>" for each similar case whose report supports the same finding.
- Never state a finding only because similar cases had it. Small or focal findings (nodules, masses, fractures, lines, tubes, wires, devices) must be visible in the image and cite "image".
- Use standard, concise radiology language. Write 3-8 findings sentences and 1-3 impression sentences.
- If the image is not a chest X-ray or is unreadable, return a single findings sentence saying so, with source "image".
"""

VERIFY = """You are checking a draft chest X-ray report sentence by sentence.

You receive the X-ray, the reports of the similar cases, and the numbered draft sentences with their cited sources.

For each sentence, decide whether it is supported:
- If it cites "image", the finding must be visible in the X-ray.
- If it cites "case:<uid>", that case's report must contain the same finding.
- A sentence that describes this patient must be visible in the image; similar cases alone are not enough.
- Be strict about small or focal findings (nodules, fractures, devices): unsupported unless clearly visible.
Return one verdict per sentence, using the sentence's 0-based index, with a one-sentence reason.
"""

REVISE = """You are revising a draft chest X-ray report after a sentence-by-sentence check.

You receive the X-ray, the similar cases' reports, the draft, and the verdicts. Rewrite the report:
- Keep supported sentences unchanged.
- For each unsupported sentence, either fix it so it is supported (correct the finding or its sources) or drop it.
- Do not add new findings that were not in the draft unless clearly visible in the image.
- Keep the same rules for sources: "image" and/or "case:<uid>".
"""

LOCAL_PROMPTS = {
    "findxray-draft": DRAFT,
    "findxray-verify": VERIFY,
    "findxray-revise": REVISE,
}


def format_cases(cases: list[dict]) -> str:
    lines = []
    for c in cases:
        parts = [f"case:{c['uid']} (similarity {c.get('similarity', 0):.2f})",
                 f"labels: {', '.join(c.get('groups', [])) or 'none'}"]
        for key in ("indication", "findings", "impression"):
            if c.get(key):
                parts.append(f"{key}: {c[key]}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def format_report(sentences) -> str:
    return "\n".join(f"[{i}] ({s.section}) {s.text}  sources: {', '.join(s.sources) or 'none'}"
                     for i, s in enumerate(sentences))
