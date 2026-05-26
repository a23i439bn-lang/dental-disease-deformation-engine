from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.disease_priors import resolve_preset_disease_name

''' プロンプトを構築する関数 '''
def load_disease_prompts(path: str | Path = "presets/disease_prompts.json") -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

''' 症状の重症度をラベルに変換する関数 '''
def severity_to_label(severity: float) -> str:
    if severity < 0.34:
        return "mild"
    if severity < 0.67:
        return "moderate"
    return "severe"

''' 疾患名のプリセットからプロンプトを構築する関数 '''
def build_research_prompt(
    disease_names: list[str],
    severity: float,
    preset_path: str | Path = "presets/disease_prompts.json",
) -> tuple[str, str, dict[str, Any]]:
    """
    Research note:
    Using only raw disease names is too weak for diffusion. We convert severity into
    a clinically descriptive sentence so disease evidence is explicit in the text path.
    """
    preset_data = load_disease_prompts(preset_path)
    common = preset_data["common"]
    diseases = preset_data["diseases"]
    strength = severity_to_label(severity)

    prompt_parts = [common["baseline_prompt"]]
    guidance_scale = common.get("baseline_guidance_scale", 4.5)
    steps = common.get("baseline_steps", 24)
    denoise_strength = common.get("baseline_denoise_strength", 0.12)

    for disease in disease_names:
        try:
            resolved_name = resolve_preset_disease_name(disease, diseases.keys()) if disease not in diseases else disease
        except KeyError:
            prompt_parts.append(disease)
            continue
        disease_entry = diseases.get(resolved_name)
        if disease_entry is None:
            prompt_parts.append(disease)
            continue
        prompt_parts.append(disease_entry["prompts"][strength])
        guidance_scale = max(guidance_scale, disease_entry.get("guidance_scale", guidance_scale))
        steps = max(steps, disease_entry.get("steps", steps))
        denoise_strength = max(denoise_strength, disease_entry.get("denoise_strength", denoise_strength))

    prompt_parts.extend(
        [
            "clinical dental photo",
            "same person",
            "preserve identity",
            "visible pathology only in oral region",
            "realistic enamel",
            "realistic gum texture",
        ]
    )
    prompt = ", ".join(prompt_parts)
    negative_prompt = common["negative_prompt"]
    render_overrides = {
        "strength_label": strength,
        "guidance_scale": guidance_scale,
        "steps": steps,
        "denoise_strength": denoise_strength,
    }
    return prompt, negative_prompt, render_overrides
