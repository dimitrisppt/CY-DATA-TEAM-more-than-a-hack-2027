"""Permutation-equivariant, team-symmetric spatial player interaction pool."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PlayerInteractionPool(nn.Module):
    """Contextualize both teams jointly, then return one summary per team.

    Player slots have no identity or position parameters. The only relation
    parameters distinguish same-team from opponent pairs, so swapping the two
    teams exchanges outputs. Coordinates are the normalized feature channels
    0:2; invalid coordinates are sanitized by the caller before this module.
    """

    def __init__(self, width: int, *, distance_bias: bool = True) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        self.width = width
        self.distance_bias = distance_bias
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.relation_bias = nn.Parameter(torch.zeros(2))  # same team, opponent
        if distance_bias:
            self.distance_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, embedded: Tensor, coords: Tensor, valid: Tensor) -> Tensor:
        """Map [B,T,2,P,W] to [B,T,2,W], including empty sets."""
        batch, frames, teams, players, width = embedded.shape
        if teams != 2 or width != self.width or coords.shape != (batch, frames, 2, players, 2):
            raise ValueError("interaction embedding or coordinate shape mismatch")
        if valid.shape != (batch, frames, 2, players):
            raise ValueError("interaction mask shape mismatch")
        if players == 0:
            return embedded.new_zeros((batch, frames, 2, width))

        count = 2 * players
        hidden = embedded.reshape(batch, frames, count, width)
        mask = valid.bool().reshape(batch, frames, count)
        positions = coords.reshape(batch, frames, count, 2)
        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(width)

        team = torch.arange(2, device=embedded.device).repeat_interleave(players)
        relations = (team[:, None] != team[None, :]).long()
        logits = logits + self.relation_bias[relations]
        if self.distance_bias:
            distances = torch.cdist(positions, positions)
            logits = logits - F.softplus(self.distance_scale) * distances

        # All-missing frames have no usable keys. Multiplication and
        # renormalization make their messages exactly zero without NaNs.
        logits = logits.masked_fill(~mask.unsqueeze(-2), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1) * mask.unsqueeze(-2)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        message = self.output(torch.matmul(weights, value))
        contextual = self.norm(hidden + message) * mask.unsqueeze(-1)
        contextual = contextual.reshape(batch, frames, 2, players, width)
        team_sum = contextual.sum(dim=3)
        return team_sum / valid.sum(dim=3).clamp_min(1).unsqueeze(-1)
