from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import numpy as np
import torch
from PIL import Image

from dataset.dataset_builder import FaceLandmarkDetector, resize_image
from debug_visualize_delta import visualize_delta
from diffusion.pipeline import DenseWarpLayer
from models.deformation_policy import DeformationPolicyConfig, DeformationPolicyNetwork
from models.disease_encoder import DiseaseEncoder, DiseaseEncoderConfig
from utils.disease_priors import MOUTH_INDICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect deformation delta without diffusion.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--disease", required=True)
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output-dir", default="outputs/deformation_infer")
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


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0)
    if array.ndim == 4:
        array = array[0]
    array = (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    diseases: list[str] = checkpoint["diseases"]
    if args.disease not in diseases:
        raise KeyError(f"Disease '{args.disease}' not found in checkpoint diseases: {diseases}")

    disease_encoder = DiseaseEncoder(
        DiseaseEncoderConfig(
            num_diseases=len(diseases),
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
    disease_encoder.load_state_dict(checkpoint["disease_encoder"])
    deformation_policy.load_state_dict(checkpoint["deformation_policy"])
    disease_encoder.eval()
    deformation_policy.eval()
    warp_layer = DenseWarpLayer(sigma=28.0).to(device)

    image = resize_image(Image.open(args.input).convert("RGB"), args.image_size)
    detector = FaceLandmarkDetector()
    try:
        landmarks_np = detector(image).astype(np.float32)
    finally:
        detector.close()

    image_tensor = pil_to_tensor(image).to(device)
    landmarks = torch.from_numpy(landmarks_np).unsqueeze(0).to(device)
    multi_hot = torch.zeros(1, len(diseases), device=device, dtype=torch.float32)
    multi_hot[0, diseases.index(args.disease)] = 1.0
    severity = torch.tensor([[args.severity]], device=device, dtype=torch.float32)

    with torch.no_grad():
        disease_embed = disease_encoder(multi_hot, severity)
        delta = deformation_policy(landmarks, disease_embed)
        target_landmarks = landmarks + delta
        warped_image, dense_flow = warp_layer(image_tensor, landmarks, target_landmarks)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_to_image(warped_image).save(output_dir / "warped.png")
    visualize_delta(
        np.array(image),
        landmarks_np,
        delta[0].detach().cpu().numpy(),
        output_dir / "delta_debug.png",
        mouth_indices=MOUTH_INDICES,
    )

    summary = {
        "disease": args.disease,
        "severity": args.severity,
        "diseases_in_checkpoint": diseases,
        "landmarks_shape": list(landmarks.shape),
        "delta_mean": float(delta.abs().mean().detach().cpu()),
        "delta_max": float(delta.abs().max().detach().cpu()),
        "mouth_delta_mean": float(delta[:, MOUTH_INDICES].abs().mean().detach().cpu()),
        "mouth_flow_mean": float(dense_flow.abs().mean().detach().cpu()),
        "output_dir": str(output_dir.resolve()),
    }
    print(f"delta mean: {summary['delta_mean']:.6f}")
    print(f"delta max : {summary['delta_max']:.6f}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
