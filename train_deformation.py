from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset.dataset_builder import DatasetBuilderConfig, DiseaseResearchDataset, build_manifest, collate_fn
from debug_visualize_delta import visualize_delta
from diffusion.pipeline import DenseWarpLayer
from models.deformation_policy import DeformationPolicyConfig, DeformationPolicyNetwork
from models.disease_encoder import DiseaseEncoder, DiseaseEncoderConfig
from utils.disease_priors import ANCHOR_INDICES, MOUTH_INDICES, build_teacher_target_landmarks, select_structural_disease


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1 deformation training without diffusion.")
    parser.add_argument("--manifest-path", default="data/research_manifest_geometry.json")
    parser.add_argument("--train-data-dir", default="data/inputs")
    parser.add_argument("--references-dir", default="data/references")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--samples-per-image", type=int, default=2)
    parser.add_argument("--auto-build-manifest", action="store_true")
    parser.add_argument("--output-dir", default="outputs/deformation_phase1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warp-sigma", type=float, default=28.0)
    parser.add_argument("--mouth-loss-weight", type=float, default=1.0)
    parser.add_argument("--warp-loss-weight", type=float, default=0.2)
    parser.add_argument("--anchor-loss-weight", type=float, default=0.5)
    parser.add_argument("--non-mouth-loss-weight", type=float, default=0.25)
    parser.add_argument("--save-debug-every", type=int, default=10)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested explicitly, but CUDA is not available.")
        return "cuda"
    if requested_device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_manifest(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.manifest_path)
    if manifest_path.exists():
        return manifest_path
    if not args.auto_build_manifest:
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Re-run with --auto-build-manifest to create one from inputs/references."
        )
    config = DatasetBuilderConfig(
        input_dir=args.train_data_dir,
        references_dir=args.references_dir,
        manifest_path=str(manifest_path),
        image_size=args.image_size,
        diseases=("出っ歯", "すきっ歯"),
        samples_per_image=args.samples_per_image,
    )
    build_manifest(config)
    return manifest_path


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0)
    if array.ndim == 4:
        array = array[0]
    array = (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def flow_to_image(flow: torch.Tensor) -> Image.Image:
    flow = flow.detach().cpu()
    magnitude = torch.linalg.norm(flow, dim=1)
    if magnitude.ndim == 3:
        magnitude = magnitude[0]
    magnitude = magnitude / magnitude.max().clamp_min(1e-6)
    image = (magnitude.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(image, mode="L")


def save_debug_images(
    output_dir: Path,
    global_step: int,
    source_image: torch.Tensor,
    landmarks: torch.Tensor,
    delta: torch.Tensor,
    teacher_delta: torch.Tensor,
    warped_image: torch.Tensor,
    teacher_warped_image: torch.Tensor,
    dense_flow: torch.Tensor,
) -> None:
    step_dir = output_dir / "debug" / f"step_{global_step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    source_pil = tensor_to_image(source_image)
    source_pil.save(step_dir / "source.png")
    tensor_to_image(warped_image).save(step_dir / "warped.png")
    tensor_to_image(teacher_warped_image).save(step_dir / "teacher_warped.png")
    flow_to_image(dense_flow).save(step_dir / "flow_magnitude.png")
    source_np = np.array(source_pil)
    visualize_delta(
        source_np,
        landmarks[0].detach().cpu().numpy(),
        delta[0].detach().cpu().numpy(),
        step_dir / "delta_arrows.png",
        mouth_indices=MOUTH_INDICES,
    )
    visualize_delta(
        source_np,
        landmarks[0].detach().cpu().numpy(),
        teacher_delta[0].detach().cpu().numpy(),
        step_dir / "teacher_delta_arrows.png",
        mouth_indices=MOUTH_INDICES,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Phase-1 deformation training currently expects --batch-size 1 to avoid variable-size collation issues.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    manifest_path = ensure_manifest(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DiseaseResearchDataset(str(manifest_path), image_size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
        collate_fn=collate_fn,
    )

    disease_encoder = DiseaseEncoder(
        DiseaseEncoderConfig(
            num_diseases=len(dataset.diseases),
            embed_dim=256,
            hidden_dim=256,
        )
    ).to(device)
    deformation_policy = DeformationPolicyNetwork(
        DeformationPolicyConfig(
            num_landmarks=468,
            landmark_dim=2,
            disease_dim=256,
            hidden_dim=192,
        )
    ).to(device)
    warp_layer = DenseWarpLayer(sigma=args.warp_sigma).to(device)

    optimizer = torch.optim.AdamW(
        list(disease_encoder.parameters()) + list(deformation_policy.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    mouth_idx = torch.tensor(MOUTH_INDICES, device=device, dtype=torch.long)
    anchor_idx = torch.tensor(ANCHOR_INDICES, device=device, dtype=torch.long)
    non_mouth_mask = torch.ones(468, device=device, dtype=torch.bool)
    non_mouth_mask[mouth_idx] = False

    global_step = 0
    history: list[dict[str, float | int]] = []

    try:
        for epoch in range(args.epochs):
            for step, batch in enumerate(dataloader, start=1):
                optimizer.zero_grad(set_to_none=True)

                image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
                landmarks = batch["landmarks_px"].to(device=device, dtype=torch.float32, non_blocking=True)
                multi_hot = batch["multi_hot"].to(device=device, dtype=torch.float32, non_blocking=True)
                severity = batch["severity"].to(device=device, dtype=torch.float32, non_blocking=True)

                disease_embed = disease_encoder(multi_hot, severity)
                delta = deformation_policy(landmarks, disease_embed)
                predicted_target = landmarks + delta
                teacher_target = build_teacher_target_landmarks(landmarks, batch["disease_names"], severity).detach()
                teacher_delta = teacher_target - landmarks

                warped_image, dense_flow = warp_layer(image, landmarks, predicted_target)
                with torch.no_grad():
                    teacher_warped_image, _ = warp_layer(image, landmarks, teacher_target)

                mouth_loss = F.smooth_l1_loss(predicted_target[:, mouth_idx], teacher_target[:, mouth_idx], reduction="mean")
                warp_loss = F.l1_loss(warped_image, teacher_warped_image, reduction="mean")
                anchor_loss = F.smooth_l1_loss(delta[:, anchor_idx], torch.zeros_like(delta[:, anchor_idx]), reduction="mean")
                non_mouth_loss = F.smooth_l1_loss(
                    delta[:, non_mouth_mask],
                    torch.zeros_like(delta[:, non_mouth_mask]),
                    reduction="mean",
                )
                teacher_delta_mean = teacher_delta[:, mouth_idx].abs().mean()
                predicted_delta_mean = delta[:, mouth_idx].abs().mean()
                magnitude_gap = F.l1_loss(predicted_delta_mean.unsqueeze(0), teacher_delta_mean.unsqueeze(0), reduction="mean")

                loss = (
                    args.mouth_loss_weight * mouth_loss
                    + args.warp_loss_weight * warp_loss
                    + args.anchor_loss_weight * anchor_loss
                    + args.non_mouth_loss_weight * non_mouth_loss
                    + 0.1 * magnitude_gap
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Loss became invalid at step {global_step + 1}: {loss.item()}")

                loss.backward()
                clip_grad_norm_(
                    list(disease_encoder.parameters()) + list(deformation_policy.parameters()),
                    max_norm=args.gradient_clip,
                )
                optimizer.step()

                global_step += 1
                record = {
                    "epoch": epoch + 1,
                    "step": step,
                    "global_step": global_step,
                    "disease_names": batch["disease_names"][0],
                    "structural_disease": select_structural_disease(batch["disease_names"][0]),
                    "loss": round(float(loss.detach().cpu()), 6),
                    "mouth_loss": round(float(mouth_loss.detach().cpu()), 6),
                    "warp_loss": round(float(warp_loss.detach().cpu()), 6),
                    "anchor_loss": round(float(anchor_loss.detach().cpu()), 6),
                    "non_mouth_loss": round(float(non_mouth_loss.detach().cpu()), 6),
                    "delta_abs_mean": round(float(delta.abs().mean().detach().cpu()), 6),
                    "delta_abs_max": round(float(delta.abs().max().detach().cpu()), 6),
                    "mouth_delta_abs_mean": round(float(predicted_delta_mean.detach().cpu()), 6),
                    "teacher_mouth_delta_abs_mean": round(float(teacher_delta_mean.detach().cpu()), 6),
                    "landmarks_shape": list(landmarks.shape),
                }
                history.append(record)
                print(json.dumps(record, ensure_ascii=False))
                print(f"delta mean: {delta.abs().mean().item():.6f}")
                print(f"delta max : {delta.abs().max().item():.6f}")

                if args.save_debug_every > 0 and global_step % args.save_debug_every == 0:
                    save_debug_images(output_dir, global_step, image, landmarks, delta, teacher_delta, warped_image, teacher_warped_image, dense_flow)

            checkpoint = {
                "disease_encoder": disease_encoder.state_dict(),
                "deformation_policy": deformation_policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "diseases": dataset.diseases,
                "args": vars(args),
                "epoch": epoch + 1,
                "global_step": global_step,
            }
            torch.save(checkpoint, output_dir / "checkpoint_last.pt")
            if args.save_every_epoch:
                torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt")

        (output_dir / "training_state.json").write_text(
            json.dumps(
                {
                    "device": device,
                    "epochs": args.epochs,
                    "global_step": global_step,
                    "manifest_path": str(manifest_path.resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "history_tail": history[-10:],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
