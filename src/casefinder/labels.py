"""Map IU X-ray (Open-i) MeSH annotations to a compact set of finding groups.

IU studies come with manually assigned MeSH entries such as
"Cardiomegaly/borderline;Pulmonary Artery/enlarged" or "Lung/hypoinflation".
Many raw terms are anatomy ("Lung", "Spine", "Aorta") whose meaning lives in the
qualifier, so matching raw terms would make every "Lung" case "similar". We map
term (+ qualifier) to finding groups instead. Rules are deliberately simple and
listed in full so they can be reviewed.
"""

from __future__ import annotations

import numpy as np

NO_FINDING = "No Finding"
OTHER = "Other Abnormal"

TERM_TO_GROUP: dict[str, str] = {
    "normal": NO_FINDING,
    "Cardiomegaly": "Cardiomegaly",
    "Pulmonary Atelectasis": "Atelectasis",
    "Volume Loss": "Atelectasis",
    "Pleural Effusion": "Pleural Effusion",
    "Opacity": "Lung Opacity",
    "Airspace Disease": "Lung Opacity",
    "Infiltrate": "Lung Opacity",
    "Consolidation": "Lung Opacity",
    "Pneumonia": "Lung Opacity",
    "Pulmonary Edema": "Edema / Congestion",
    "Pulmonary Congestion": "Edema / Congestion",
    "Heart Failure": "Edema / Congestion",
    "Pneumothorax": "Pneumothorax",
    "Hydropneumothorax": "Pneumothorax",
    "Hemopneumothorax": "Pneumothorax",
    "Nodule": "Nodule / Mass",
    "Mass": "Nodule / Mass",
    "Calcified Granuloma": "Granuloma / Calcification",
    "Granulomatous Disease": "Granuloma / Calcification",
    "Granuloma": "Granuloma / Calcification",
    "Calcinosis": "Granuloma / Calcification",
    "Emphysema": "Hyperinflation / COPD",
    "Pulmonary Emphysema": "Hyperinflation / COPD",
    "Bullous Emphysema": "Hyperinflation / COPD",
    "Pulmonary Disease, Chronic Obstructive": "Hyperinflation / COPD",
    "Lung, Hyperlucent": "Hyperinflation / COPD",
    "Fractures, Bone": "Fracture",
    "Catheters, Indwelling": "Devices / Hardware",
    "Surgical Instruments": "Devices / Hardware",
    "Implanted Medical Device": "Devices / Hardware",
    "Tube, Inserted": "Devices / Hardware",
    "Medical Device": "Devices / Hardware",
    "Stents": "Devices / Hardware",
    "Sutures": "Devices / Hardware",
    "Spinal Fusion": "Devices / Hardware",
    "Breast Implants": "Devices / Hardware",
    "Foreign Bodies": "Devices / Hardware",
    "Osteophyte": "Spine Degenerative / Curvature",
    "Spondylosis": "Spine Degenerative / Curvature",
    "Arthritis": "Spine Degenerative / Curvature",
    "Scoliosis": "Spine Degenerative / Curvature",
    "Kyphosis": "Spine Degenerative / Curvature",
    "Atherosclerosis": "Aorta Abnormal",
    "Aortic Aneurysm": "Aorta Abnormal",
    "Aorta, Thoracic": "Aorta Abnormal",
    "Cicatrix": "Scarring / Fibrosis",
    "Pulmonary Fibrosis": "Scarring / Fibrosis",
    "Lung Diseases, Interstitial": "Scarring / Fibrosis",
    "Fibrosis": "Scarring / Fibrosis",
    "Thickening": "Scarring / Fibrosis",
    "Hernia, Hiatal": "Hiatal Hernia",
    "Hernia, Diaphragmatic": "Hiatal Hernia",
}

# (term, qualifier) -> group, for anatomy terms whose meaning is in the qualifier
QUALIFIER_TO_GROUP: dict[tuple[str, str], str] = {
    ("Cardiac Shadow", "enlarged"): "Cardiomegaly",
    ("Lung", "hypoinflation"): "Low Lung Volume",
    ("Lung", "hyperdistention"): "Hyperinflation / COPD",
    ("Diaphragm", "flattened"): "Hyperinflation / COPD",
    ("Costophrenic Angle", "blunted"): "Pleural Effusion",
    ("Density", "lung"): "Lung Opacity",
    ("Thoracic Vertebrae", "degenerative"): "Spine Degenerative / Curvature",
    ("Spine", "degenerative"): "Spine Degenerative / Curvature",
    ("Aorta", "tortuous"): "Aorta Abnormal",
}

# Terms that say nothing about the patient's findings
IGNORED_TERMS = {"No Indexing", "Technical Quality of Image Unsatisfactory"}

GROUPS: list[str] = sorted(
    set(TERM_TO_GROUP.values()) | set(QUALIFIER_TO_GROUP.values()) | {OTHER}
)


def mesh_to_groups(mesh: str | None) -> set[str] | None:
    """Return the finding groups for one study's MeSH string.

    Returns None when the study carries no usable annotation (missing or only
    ignored terms) so callers can drop it from evaluation.
    """
    if not mesh or not mesh.strip():
        return None
    groups: set[str] = set()
    usable = False
    for entry in mesh.split(";"):
        parts = [p.strip() for p in entry.split("/") if p.strip()]
        if not parts:
            continue
        term = parts[0]
        if term in IGNORED_TERMS:
            continue
        usable = True
        if term in TERM_TO_GROUP:
            groups.add(TERM_TO_GROUP[term])
        for qual in parts[1:]:
            if (term, qual) in QUALIFIER_TO_GROUP:
                groups.add(QUALIFIER_TO_GROUP[(term, qual)])
    if not usable:
        return None
    if NO_FINDING in groups and len(groups) > 1:
        groups.discard(NO_FINDING)
    if not groups:
        groups.add(OTHER)  # annotated as abnormal, but nothing we map
    return groups


def to_multihot(group_sets: list[set[str]]) -> np.ndarray:
    index = {g: i for i, g in enumerate(GROUPS)}
    out = np.zeros((len(group_sets), len(GROUPS)), dtype=bool)
    for row, gs in enumerate(group_sets):
        for g in gs:
            out[row, index[g]] = True
    return out
