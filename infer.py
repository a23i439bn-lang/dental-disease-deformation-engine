from __future__ import annotations
# このファイルの役割:
# 学習済みモデルを使って最小構成の推論を行うスクリプトです。
# Stable Diffusion ベースの推論パイプラインを読み込み、
# prompt や追加 U-Net を使って `rendered.png` を出力します。

"""学習済みの disease-aware パイプライン全体を推論する入口。

Deformation、Texture、Renderer をまとめて通し、
最終的な rendered.png を出す本番寄りの推論スクリプト。
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse {path}. This minimal baseline expects JSON-compatible YAML."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal stable inference for Stable Diffusion.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/minimal_sd_infer")
    parser.add_argument("--unet-path", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def resolve_device(requested_device: str, config_device: str) -> str:
    if requested_device != "auto":
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested explicitly, but CUDA is not available.")
        return requested_device

    if config_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(model_id: str, dtype: torch.dtype) -> StableDiffusionPipeline:
    try:
        return StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
    except Exception:
        return StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_id = str(config["model"]["model_id"])
    infer_cfg = config["inference"]

    device = resolve_device(args.device, str(infer_cfg["device"]))
    dtype = torch.float16 if device == "cuda" else torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipe = load_pipeline(model_id, dtype)
        if args.unet_path:
            pipe.unet = UNet2DConditionModel.from_pretrained(args.unet_path, torch_dtype=dtype)

        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()

        generator = torch.Generator(device=device).manual_seed(args.seed)
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=float(infer_cfg["guidance_scale"]),
            num_inference_steps=int(infer_cfg["num_inference_steps"]),
            generator=generator,
        )
        image = result.images[0]
        image.save(output_dir / "rendered.png")

        metadata = {
            "model_id": model_id,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "guidance_scale": float(infer_cfg["guidance_scale"]),
            "num_inference_steps": int(infer_cfg["num_inference_steps"]),
            "device": device,
            "torch_dtype": str(dtype),
            "unet_path": None if args.unet_path is None else str(Path(args.unet_path).resolve()),
            "output_image": str((output_dir / "rendered.png").resolve()),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA out of memory during inference.") from exc
    except Exception as exc:
        raise RuntimeError(f"Inference failed safely before saving invalid output: {exc}") from exc


if __name__ == "__main__":
    main()
