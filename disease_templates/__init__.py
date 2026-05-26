from __future__ import annotations

from disease_templates.base import FaceDiseaseTemplate
from disease_templates.chin_deviation_left import ChinDeviationLeftTemplate
from disease_templates.chin_deviation_right import ChinDeviationRightTemplate
from disease_templates.mandibular_protrusion import MandibularProtrusionTemplate
from disease_templates.maxillary_protrusion import MaxillaryProtrusionTemplate
from disease_templates.occlusal_plane_cant import OcclusalPlaneCantTemplate


TEMPLATES: tuple[FaceDiseaseTemplate, ...] = (
    MandibularProtrusionTemplate(),
    MaxillaryProtrusionTemplate(),
    ChinDeviationLeftTemplate(),
    ChinDeviationRightTemplate(),
    OcclusalPlaneCantTemplate(),
    OcclusalPlaneCantTemplate(
        direction=1.0,
        canonical_name="occlusal_plane_cant_right",
        aliases=("occlusal_cant_right", "cant_right"),
    ),
)

SUPPORTED_DISEASE_CHOICES = tuple(template.canonical_name for template in TEMPLATES)


def get_template(name: str) -> FaceDiseaseTemplate:
    normalized = name.strip().lower().replace(" ", "_")
    for template in TEMPLATES:
        if template.matches(normalized):
            return template
    available = ", ".join(SUPPORTED_DISEASE_CHOICES)
    raise KeyError(f"Unsupported disease template: {name}. Available: {available}")
