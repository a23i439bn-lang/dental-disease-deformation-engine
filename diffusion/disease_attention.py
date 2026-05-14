from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DiseaseAttentionConfig:
    disease_dim: int = 256
    cross_attention_dim: int = 768
    num_disease_tokens: int = 4


def run_cross_attention_safety_checks(
    disease_embed: torch.Tensor,
    expected_disease_dim: int,
    expected_num_tokens: int,
) -> None:
    """
    Research note:
    Diffusion collapse often starts with silent conditioning mismatches.
    We fail fast here so that broken disease tokens never reach the UNet.
    """
    if disease_embed.ndim != 2:
        raise ValueError(f"disease_embed must be 2D (B, D), got shape {tuple(disease_embed.shape)}")
    if disease_embed.shape[-1] != expected_disease_dim:
        raise ValueError(
            f"Disease embedding dim mismatch: expected {expected_disease_dim}, got {disease_embed.shape[-1]}"
        )
    if expected_num_tokens <= 0:
        raise ValueError(f"num_disease_tokens must be > 0, got {expected_num_tokens}")
    if torch.isnan(disease_embed).any() or torch.isinf(disease_embed).any():
        raise ValueError("Disease embedding contains NaN or Inf values.")


class DiseaseTokenProjector(nn.Module):
    def __init__(self, config: DiseaseAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.disease_dim, config.cross_attention_dim * config.num_disease_tokens)

    def forward(self, disease_embed: torch.Tensor) -> torch.Tensor:
        run_cross_attention_safety_checks(
            disease_embed=disease_embed,
            expected_disease_dim=self.config.disease_dim,
            expected_num_tokens=self.config.num_disease_tokens,
        )
        tokens = self.proj(disease_embed)
        return tokens.view(disease_embed.shape[0], self.config.num_disease_tokens, self.config.cross_attention_dim)


class DiseaseCrossAttentionProcessor(nn.Module):
    """
    Diffusers attention processor that appends disease tokens only to cross-attention.

    Research note:
    We intentionally avoid touching self-attention blocks, because doing so tends to
    destabilize diffusion dynamics and produces visually collapsed `rendered.png`.
    """

    def __init__(self, projector: DiseaseTokenProjector) -> None:
        super().__init__()
        self.projector = projector
        self._disease_embed: torch.Tensor | None = None

    def set_condition(self, disease_embed: torch.Tensor | None) -> None:
        self._disease_embed = disease_embed

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)
        use_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if use_cross_attention and self._disease_embed is not None:
            disease_tokens = self.projector(self._disease_embed.to(encoder_hidden_states.device, encoder_hidden_states.dtype))
            if disease_tokens.shape[0] != encoder_hidden_states.shape[0]:
                if encoder_hidden_states.shape[0] % disease_tokens.shape[0] != 0:
                    raise ValueError(
                        "Classifier-free guidance expanded batch is incompatible with disease tokens: "
                        f"encoder batch={encoder_hidden_states.shape[0]}, disease batch={disease_tokens.shape[0]}"
                    )
                repeat_factor = encoder_hidden_states.shape[0] // disease_tokens.shape[0]
                disease_tokens = disease_tokens.repeat_interleave(repeat_factor, dim=0)
            if disease_tokens.shape[-1] != encoder_hidden_states.shape[-1]:
                raise ValueError(
                    "Cross-attention token dim mismatch: "
                    f"encoder hidden dim={encoder_hidden_states.shape[-1]}, disease token dim={disease_tokens.shape[-1]}"
                )
            encoder_hidden_states = torch.cat([encoder_hidden_states, disease_tokens], dim=1)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if torch.isnan(attention_probs).any() or torch.isinf(attention_probs).any():
            raise ValueError("Cross-attention probabilities became NaN/Inf.")
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
