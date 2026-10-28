"""Local prompt texts. These are the fallback and the source pushed to Langfuse
(scripts/push_prompts.py); at run time Langfuse's "production" version wins if configured."""

from __future__ import annotations

DRAFT = """You are assisting a radiologist by drafting a chest X-ray report for a research demo.

You receive the patient's frontal chest X-ray, the reports of the most similar past cases (found by image
similarity), and a checklist of findings to look for.

How to read the X-ray:
- Review it systematically: lines/tubes/devices, heart size, mediastinum and hila, lungs (opacities,
  nodules, volumes, hyperinflation), pleura (effusion; pneumothorax, especially at the apices),
  diaphragm, bones and soft tissues.
- Go through the checklist one by one and decide for THIS image whether each finding is present.
  The checklist includes findings the similar cases had; they often recur in look-alike images.
- Report abnormalities you see, including subtle or uncertain ones (say "possible" or "may represent").
  Do not write a normal report unless you have checked each item and found nothing.

Writing rules:
- Every sentence lists its sources: "image" for what you see in this X-ray, plus "case:<uid>" ONLY when
  that similar case's report contains the same abnormal finding. Never cite a case for a normal or
  negative statement (e.g. "the lungs are clear"); those cite "image" only.
- Never state a finding only because similar cases had it; it must be visible in this image.
- Use standard, concise radiology language: 3-8 findings sentences and 1-3 impression sentences.
- If the image is not a chest X-ray or is unreadable, return a single findings sentence saying so, with source "image".
"""

VERIFY = """You are checking a draft chest X-ray report sentence by sentence.

You receive the X-ray, the reports of the similar cases, and the numbered draft sentences with their cited sources.

For each sentence, decide whether it is supported:
- If it cites "image", the finding must be visible in the X-ray.
- If it cites "case:<uid>", that case's report must contain the same abnormal finding. A case cited for a
  normal or negative statement is an invalid citation: mark the sentence unsupported and say so.
- A sentence that describes this patient must be visible in the image; similar cases alone are not enough.
- Be strict about small or focal findings (nodules, fractures, devices): unsupported unless clearly visible.
Return one verdict per sentence, using the sentence's 0-based index, with a one-sentence reason.
"""

REVISE = """You are revising a draft chest X-ray report after a sentence-by-sentence check.

You receive the X-ray, the similar cases' reports, the draft, and the verdicts. Rewrite the report:
- Keep supported sentences unchanged.
- For each unsupported sentence, either fix it so it is supported (correct the finding or its sources) or drop it.
- Do not add new findings that were not in the draft unless clearly visible in the image.
- Sources: "image", plus "case:<uid>" only for cases whose report has the same abnormal finding.
"""

LOOK_AGAIN = """You are a second reader double-checking a draft chest X-ray report for MISSED findings.

You receive the X-ray, the draft report, and a checklist of findings. For EACH checklist item, look at the
image again and answer:
- present: "yes", "no" or "uncertain" for THIS image;
- in_draft: true if the draft already states this finding correctly (present findings mentioned as present,
  absent ones not claimed), false if the draft omits it or contradicts the image;
- evidence: one short phrase describing where/what you see (or why it is absent).
Be especially careful with subtle findings: small effusions (blunted costophrenic angles), pneumothorax
(visceral pleural line at the apex), mild cardiomegaly, and faint opacities.
"""

AMEND = """You are amending a draft chest X-ray report because a second reader found findings the draft missed.

You receive the X-ray, the similar cases' reports, the draft, and the missed findings with evidence.
- Add a findings sentence for each missed finding (use "possible" for uncertain ones) and update the
  impression if the finding is clinically relevant.
- Fix or remove any sentence that contradicts the missed findings (e.g. "no pleural effusion" when one is present).
- Keep all other sentences unchanged.
- Sources: "image", plus "case:<uid>" only for cases whose report has the same abnormal finding.
"""

# Findings always on the checklist (common and/or clinically important); similar cases add more.
ALWAYS_CHECK = ["Pneumothorax", "Pleural Effusion", "Cardiomegaly", "Lung Opacity", "Edema / Congestion",
                "Atelectasis", "Nodule / Mass", "Fracture", "Devices / Hardware"]
_NOT_FINDINGS = {"No Finding", "Other Abnormal"}


def checklist(cases: list[dict]) -> list[tuple[str, list[int]]]:
    """Findings to check: the always-check list plus every finding group the similar cases had,
    each with the uids of the cases that had it (those are the only valid case citations for it)."""
    seen: dict[str, list[int]] = {f: [] for f in ALWAYS_CHECK}
    for c in cases:
        for g in c.get("groups", []):
            if g not in _NOT_FINDINGS:
                seen.setdefault(g, []).append(c["uid"])
    return list(seen.items())


def format_checklist(cases: list[dict]) -> str:
    return "\n".join(f"- {name}" + (f" (seen in {', '.join(f'case:{u}' for u in uids)})" if uids else "")
                     for name, uids in checklist(cases))


LOCAL_PROMPTS = {
    "findxray-draft": DRAFT,
    "findxray-verify": VERIFY,
    "findxray-revise": REVISE,
    "findxray-look-again": LOOK_AGAIN,
    "findxray-amend": AMEND,
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
