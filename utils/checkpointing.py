from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from diffusion.pipeline import DiseaseDiffusionRenderer


def load_config(config_path: str) -> dict[str, Any]:
    content = Path(config_path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(content)
    except Exception:
        return json.loads(content)


def safe_load_module_state(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    module_name: str,
    strict: bool = True,
) -> dict[str, list[str]]:
    result = module.load_state_dict(state_dict, strict=strict)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)
    print(f"[{module_name}] missing_keys={missing_keys}")
    print(f"[{module_name}] unexpected_keys={unexpected_keys}")
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(f"{module_name} state_dict mismatch detected.")
    return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}


def save_research_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    disease_encoder: nn.Module,
    deformation_policy: nn.Module,
    texture_branch: nn.Module,
    severity_actor: nn.Module,
    renderer: DiseaseDiffusionRenderer,
    training_meta: dict[str, Any],
) -> None:
    checkpoint = {
        "config": config,
        "disease_encoder": disease_encoder.state_dict(),
        "deformation_policy": deformation_policy.state_dict(),
        "texture_branch": texture_branch.state_dict(),
        "severity_actor": severity_actor.state_dict(),
        "renderer_state": renderer.renderer_state_dict(),
        "training_meta": training_meta,
    }
    if renderer.lora_manager is not None:
        checkpoint["lora_active"] = True
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_research_checkpoint(
    checkpoint_path: str | Path,
    disease_encoder: nn.Module,
    deformation_policy: nn.Module,
    texture_branch: nn.Module,
    severity_actor: nn.Module,
    renderer: DiseaseDiffusionRenderer,
    map_location: str,
    strict: bool = True,
) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location=map_location)
    safe_load_module_state(disease_encoder, state["disease_encoder"], "disease_encoder", strict=strict)
    safe_load_module_state(deformation_policy, state["deformation_policy"], "deformation_policy", strict=strict)
    safe_load_module_state(texture_branch, state["texture_branch"], "texture_branch", strict=strict)
    safe_load_module_state(severity_actor, state.get("severity_actor", {}), "severity_actor", strict=False)
    renderer_mismatch = renderer.load_renderer_state(state["renderer_state"], strict=strict)
    state["renderer_mismatch"] = renderer_mismatch
    return state
