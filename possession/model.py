"""Team-exchange-equivariant, native-rate possession encoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from possession.interaction import PlayerInteractionPool


class PossessionModel(nn.Module):
    """Encode unordered player sets and two ordered views of the same match.

    Each temporal view sees (own team, other team, shared context). The same
    parameters process both views, so swapping teams exchanges their outputs.
    Input units and feature order are defined by the feature adapter.

    `pooling="mean"` preserves the original checkpoint architecture. With
    `pooling="ball_attention"`, a shared player scorer uses the existing
    ball-relative input channels to select controller evidence, concatenated
    with the team mean. When the ball is unseen, both summaries use the mean.
    `pooling="interaction"` adds a shared spatial message-passing summary
    for each team alongside that team's original mean.
    """

    def __init__(
        self,
        player_dim: int = 8,
        global_dim: int = 9,
        width: int = 64,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if player_dim < 1 or global_dim < 1 or width < 1 or not dilations:
            raise ValueError("feature dimensions, width, and dilations must be positive")
        if any(d < 1 for d in dilations):
            raise ValueError("dilations must be positive")
        if pooling not in ("mean", "ball_attention", "interaction"):
            raise ValueError("pooling must be 'mean', 'ball_attention', or 'interaction'")
        if pooling == "ball_attention" and (player_dim < 8 or global_dim < 3):
            raise ValueError("ball_attention requires ball-relative and visibility channels")
        if pooling == "interaction" and player_dim < 2:
            raise ValueError("interaction requires normalized player coordinates")
        self.player_dim = player_dim
        self.global_dim = global_dim
        self.width = width
        self.dilations = tuple(int(d) for d in dilations)
        self.pooling = pooling
        self.receptive_field = 1 + 2 * sum(self.dilations)
        self.radius = sum(self.dilations)

        self.player_mlp = nn.Sequential(
            nn.Linear(player_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        if pooling == "ball_attention":
            self.player_attention = nn.Sequential(
                nn.Linear(width + 2, width),
                nn.GELU(),
                nn.Linear(width, 1),
            )
        if pooling == "interaction":
            self.player_interaction = PlayerInteractionPool(width)
        team_dim = width if pooling == "mean" else 2 * width
        self.input_projection = nn.Linear(2 * team_dim + global_dim, width)
        self.temporal_convs = nn.ModuleList(
            nn.Conv1d(width, width, kernel_size=3, padding=d, dilation=d)
            for d in self.dilations
        )
        self.temporal_norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.dilations)
        self.live_head = nn.Linear(2 * width, 1)
        self.owner_head = nn.Linear(width, 1)
        self.switch_head = nn.Linear(width, 1)
        self.exit_head = nn.Linear(width, 1)
        self.entry_head = nn.Linear(width, 1)

    def forward(
        self,
        player_tokens: Tensor,
        player_mask: Tensor,
        global_tokens: Tensor,
        time_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if player_tokens.ndim != 5 or player_tokens.shape[2] != 2:
            raise ValueError("player_tokens must have shape [B,T,2,P,F]")
        batch, frames, _, players, features = player_tokens.shape
        if features != self.player_dim:
            raise ValueError("wrong player feature width")
        if player_mask.shape != (batch, frames, 2, players):
            raise ValueError("player_mask shape mismatch")
        if global_tokens.shape != (batch, frames, self.global_dim):
            raise ValueError("global_tokens shape mismatch")
        if time_valid is None:
            time_valid = torch.ones(batch, frames, dtype=torch.bool, device=player_tokens.device)
        elif time_valid.shape != (batch, frames):
            raise ValueError("time_valid shape mismatch")

        # Invalid slots never enter the MLP, even when their raw coordinates
        # contain NaNs. Mean pooling is permutation invariant and zero for an
        # empty team; no slot identifier or team-specific parameter is used.
        valid_players = player_mask.bool() & time_valid.bool().unsqueeze(-1).unsqueeze(-1)
        tokens = torch.where(valid_players.unsqueeze(-1), player_tokens, 0)
        if self.pooling == "ball_attention":
            ball_visible = (global_tokens[..., 2] > 0.5) & time_valid.bool()
            # Unobserved ball-relative inputs are undefined. Remove even finite
            # stale values and NaNs before the shared player encoder sees them.
            relative = torch.where(ball_visible[:, :, None, None, None], tokens[..., 5:8], 0)
            tokens = torch.cat((tokens[..., :5], relative, tokens[..., 8:]), dim=-1)
        embedded = self.player_mlp(tokens)
        pooled = (embedded * valid_players.unsqueeze(-1)).sum(dim=3)
        pooled = pooled / valid_players.sum(dim=3).clamp_min(1).unsqueeze(-1)
        if self.pooling == "ball_attention":
            if players:
                attention_inputs = torch.cat((embedded, tokens[..., 5:7]), dim=-1)
                attention_logits = self.player_attention(attention_inputs).squeeze(-1)
                attention_logits = attention_logits.masked_fill(
                    ~valid_players, torch.finfo(attention_logits.dtype).min
                )
                weights = torch.softmax(attention_logits, dim=3) * valid_players
                weights = weights / weights.sum(dim=3, keepdim=True).clamp_min(1e-12)
                attended = (embedded * weights.unsqueeze(-1)).sum(dim=3)
            else:
                attended = torch.zeros_like(pooled)
            attended = torch.where(ball_visible[:, :, None, None], attended, pooled)
            pooled = torch.cat((pooled, attended), dim=-1)
        elif self.pooling == "interaction":
            contextual = self.player_interaction(embedded, tokens[..., :2], valid_players)
            pooled = torch.cat((pooled, contextual), dim=-1)

        shared = torch.where(time_valid.bool().unsqueeze(-1), global_tokens, 0)
        shared = shared.unsqueeze(2).expand(-1, -1, 2, -1)
        views = torch.cat((pooled, pooled.flip(2), shared), dim=-1)
        stream = self.input_projection(views.permute(0, 2, 1, 3).reshape(batch * 2, frames, -1))
        valid_stream = time_valid.bool()[:, None, :, None].expand(batch, 2, frames, 1)
        valid_stream = valid_stream.reshape(batch * 2, frames, 1)
        stream = stream * valid_stream
        for convolution, normalization in zip(self.temporal_convs, self.temporal_norms):
            update = convolution(stream.transpose(1, 2)).transpose(1, 2)
            stream = (stream + F.gelu(normalization(update))) * valid_stream

        team_stream = stream.reshape(batch, 2, frames, self.width).permute(0, 2, 1, 3)
        home, away = team_stream.unbind(dim=2)
        live = self.live_head(torch.cat((home + away, (home - away).abs()), dim=-1))
        owner = self.owner_head(home) - self.owner_head(away)
        switch_h, switch_a = self.switch_head(team_stream).squeeze(-1).unbind(dim=2)
        exit_h, exit_a = self.exit_head(team_stream).squeeze(-1).unbind(dim=2)
        entry_h, entry_a = self.entry_head(team_stream).squeeze(-1).unbind(dim=2)
        zero = torch.zeros_like(switch_h)
        boundary = torch.stack(
            (
                torch.stack((zero, switch_h, exit_h), dim=-1),
                torch.stack((switch_a, zero, exit_a), dim=-1),
                torch.stack((entry_h, entry_a, zero), dim=-1),
            ),
            dim=-2,
        )
        return {"live_logits": live.squeeze(-1), "owner_logits": owner.squeeze(-1), "boundary": boundary}


def predict_chunked(
    model: PossessionModel,
    player_tokens: Tensor,
    player_mask: Tensor,
    global_tokens: Tensor,
    time_valid: Tensor | None = None,
    center_size: int = 512,
    halo: int = 256,
) -> dict[str, Tensor]:
    """Keep center predictions with enough real-frame context for exact seams.

    Inputs contain one half only. Context is clipped at its true edges, where
    the convolution's own zero padding matches a full-half forward pass.
    """

    if center_size < 1 or halo < model.radius:
        raise ValueError(f"center_size must be positive and halo >= {model.radius}")
    frames = player_tokens.shape[1]
    if frames < 1:
        raise ValueError("a half must contain at least one frame")
    pieces: dict[str, list[Tensor]] = {"live_logits": [], "owner_logits": [], "boundary": []}
    for start in range(0, frames, center_size):
        stop = min(start + center_size, frames)
        left = max(0, start - halo)
        right = min(frames, stop + halo)
        scores = model(
            player_tokens[:, left:right],
            player_mask[:, left:right],
            global_tokens[:, left:right],
            None if time_valid is None else time_valid[:, left:right],
        )
        for name, value in scores.items():
            pieces[name].append(value[:, start - left : stop - left])
    return {name: torch.cat(parts, dim=1) for name, parts in pieces.items()}
