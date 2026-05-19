from __future__ import annotations

"""disease-aware パイプライン全体の学習スクリプト。

変形、テクスチャ、描画条件付けまで含めて学習したいときの入口。
現在の幾何-only検証より後の段階で主に使う。
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse {path}. This minimal baseline expects JSON-compatible YAML."
        ) from exc


class ImagePromptDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int, default_prompt: str) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {self.data_dir}")

        self.records: list[dict[str, str]] = []
        for image_path in sorted(self.data_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            prompt_path = image_path.with_suffix(".txt")
            prompt = default_prompt
            if prompt_path.exists():
                prompt = prompt_path.read_text(encoding="utf-8").strip() or default_prompt
            self.records.append({"image_path": str(image_path), "prompt": prompt})

        if not self.records:
            raise ValueError(f"No training images found in {self.data_dir}")

        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        image = resize_and_center_crop(image, self.image_size)
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(image.size[1], image.size[0], 3)
        tensor = tensor.permute(2, 0, 1) / 127.5 - 1.0
        return {
            "pixel_values": tensor,
            "prompt": record["prompt"],
            "image_path": record["image_path"],
        }


def resize_and_center_crop(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    scale = image_size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - image_size) // 2)
    top = max(0, (resized.height - image_size) // 2)
    return resized.crop((left, top, left + image_size, top + image_size))


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in batch], dim=0),
        "prompt": [sample["prompt"] for sample in batch],
        "image_path": [sample["image_path"] for sample in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal stable UNet fine-tuning for Stable Diffusion.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-data-dir", default="data/inputs")
    parser.add_argument("--output-dir", default="outputs/minimal_sd_train")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--default-prompt", default="a clean portrait photo")
    parser.add_argument("--save-every-epoch", action="store_true")
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_id = str(config["model"]["model_id"])
    train_cfg = config["training"]
    device = resolve_device(args.device)
    weight_dtype = torch.float16 if device == "cuda" else torch.float32
    autocast_enabled = device == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImagePromptDataset(
        data_dir=args.train_data_dir,
        image_size=args.image_size,
        default_prompt=args.default_prompt,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=weight_dtype)
    pipe = pipe.to(device)

    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    noise_scheduler = pipe.scheduler

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)
    vae.eval()
    text_encoder.eval()
    unet.train()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=float(train_cfg["lr"]))
    grad_scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled)
    gradient_clip = float(train_cfg["gradient_clip"])
    epochs = int(train_cfg["epochs"])

    global_step = 0

    try:
        for epoch in range(epochs):
            for step, batch in enumerate(dataloader, start=1):
                optimizer.zero_grad(set_to_none=True)
                pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32, non_blocking=True)

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    if latents.ndim != 4:
                        raise RuntimeError(f"Unexpected latent shape: {tuple(latents.shape)}")

                    tokenized = tokenizer(
                        batch["prompt"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    encoder_hidden_states = text_encoder(
                        tokenized.input_ids.to(device),
                        attention_mask=tokenized.attention_mask.to(device),
                    )[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    low=0,
                    high=noise_scheduler.config.num_train_timesteps,
                    size=(latents.shape[0],),
                    device=device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Loss became invalid at step {global_step + 1}: {loss.item()}")

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                clip_grad_norm_(unet.parameters(), max_norm=gradient_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                global_step += 1
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "global_step": global_step,
                            "loss": round(float(loss.detach().cpu()), 6),
                            "batch_size": len(batch["prompt"]),
                        },
                        ensure_ascii=False,
                    )
                )

            if args.save_every_epoch:
                epoch_dir = output_dir / f"epoch_{epoch + 1:03d}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                unet.save_pretrained(epoch_dir / "unet")

        unet.save_pretrained(output_dir / "unet")
        training_state = {
            "model_id": model_id,
            "epochs": epochs,
            "learning_rate": float(train_cfg["lr"]),
            "batch_size": int(train_cfg["batch_size"]),
            "gradient_clip": gradient_clip,
            "total_steps": global_step,
            "train_data_dir": str(Path(args.train_data_dir).resolve()),
            "image_size": args.image_size,
            "device": device,
            "torch_dtype": str(weight_dtype),
        }
        (output_dir / "training_state.json").write_text(
            json.dumps(training_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory. Lower --image-size or training.batch_size and retry."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Training failed safely before producing invalid weights: {exc}") from exc


if __name__ == "__main__":
    main()
