"""Standalone CRAF residual head with optional detached decision conditioning.

The caller owns encoders and supplies finite floating-point inputs. Embedding
autograd connections and the direct baseline-logit connection are preserved;
only the optional decision features are detached. Gates are learned through
the caller's loss and are not calibrated reliability estimates.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def correction_penalty(residual_logits: Tensor) -> Tensor:
    """Mean squared binary margin correction, invariant to shared-logit shifts."""
    return (residual_logits[..., 1] - residual_logits[..., 0]).square().mean()


class _Projection(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class _Classifier(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, condition_on_decision: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * hidden_dim + 5 + (3 if condition_on_decision else 0), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class FrozenCRAFResidual(nn.Module):
    """Add a zero-initialized correction to binary baseline logits.

    With conditioning disabled, constructor defaults, state keys/shapes, and
    forward operations match v1. Both modalities are always present and all
    ablations retain their parameters. Conditioning appends, in order, fake
    probability, bounded signed margin, and normalized binary entropy. These
    three features depend only on detached baseline logits, never labels.

    Shapes are text/image ``[batch, feature_dim]`` and baseline ``[batch, 2]``.
    Inputs must share floating dtype/device. Finite values are a caller
    precondition; forward performs no tensor-wide scans or host synchronization.
    """

    def __init__(
        self,
        feature_dim=768,
        hidden_dim=256,
        dropout=.2,
        use_reliability=True,
        use_consistency=True,
        use_interactions=True,
        condition_on_decision=False,
    ):
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.feature_dim = feature_dim
        self.use_reliability = use_reliability
        self.use_consistency = use_consistency
        self.use_interactions = use_interactions
        self.condition_on_decision = condition_on_decision
        self.text_projection = _Projection(feature_dim, hidden_dim, dropout)
        self.image_projection = _Projection(feature_dim, hidden_dim, dropout)
        self.text_reliability = nn.Linear(hidden_dim, 1)
        self.image_reliability = nn.Linear(hidden_dim, 1)
        self.consistency_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = _Classifier(hidden_dim, dropout, condition_on_decision)
        nn.init.zeros_(self.classifier.net[3].weight)
        nn.init.zeros_(self.classifier.net[3].bias)

    @staticmethod
    def _decision_features(baseline_logits: Tensor) -> Tensor:
        logits = baseline_logits.detach()
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = nn.functional.log_softmax(logits, dim=-1)
        margin = logits[:, 1] - logits[:, 0]
        bounded_margin = margin / (1 + margin.abs())
        # Subtraction can overflow even when both supplied logits are finite.
        bounded_margin = torch.where(margin.isinf(), margin.sign(), bounded_margin)
        # At extreme finite logits, log_softmax can be -inf where p is zero.
        entropy_terms = torch.where(
            probabilities == 0,
            torch.zeros_like(probabilities),
            probabilities * log_probabilities,
        )
        entropy = -entropy_terms.sum(dim=-1) / math.log(2)
        return torch.stack((probabilities[:, 1], bounded_margin, entropy), dim=-1)

    def forward(self, text, image, baseline_logits, return_details=False):
        inputs = (("text", text), ("image", image), ("baseline_logits", baseline_logits))
        for name, value in inputs:
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            width = 2 if name == "baseline_logits" else self.feature_dim
            if value.ndim != 2 or value.shape[1] != width:
                raise ValueError(f"{name} must have shape [batch, {width}]")
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
        for name, value in inputs[1:]:
            if value.shape[0] != text.shape[0]:
                raise ValueError(f"{name} must have the same batch size as text")
            if value.device != text.device or value.dtype != text.dtype:
                raise ValueError(f"{name} must share text's device and dtype")

        mask = text.new_ones((text.shape[0], 2))
        text = self.text_projection(text) * mask[:, 0:1]
        image = self.image_projection(image) * mask[:, 1:2]
        reliability_logits = None
        reliability_scores = torch.zeros_like(mask)
        if self.use_reliability:
            reliability_logits = torch.cat(
                [self.text_reliability(text), self.image_reliability(image)], dim=-1
            )
            reliability_scores = torch.sigmoid(reliability_logits) * mask
            log_scores = nn.functional.logsigmoid(reliability_logits)
            log_scores = log_scores.masked_fill(mask == 0, torch.finfo(mask.dtype).min)
            modality_weights = torch.softmax(log_scores, dim=-1) * mask
        else:
            modality_weights = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

        difference = (text - image).abs()
        product = text * image
        if not self.use_interactions:
            difference = torch.zeros_like(difference)
            product = torch.zeros_like(product)
        pair_features = torch.cat([text, image, difference, product], dim=-1)
        pair_mask = (mask[:, 0] * mask[:, 1]).unsqueeze(-1)
        consistency_logits = None
        consistency_scores = torch.zeros_like(pair_mask)
        if self.use_consistency:
            consistency_logits = self.consistency_head(pair_features)
            consistency_scores = torch.sigmoid(consistency_logits) * pair_mask
        fused = torch.cat(
            [
                text * modality_weights[:, 0:1],
                image * modality_weights[:, 1:2],
                difference,
                product,
                consistency_scores,
                reliability_scores,
                mask,
            ],
            dim=-1,
        )
        decision_features = None
        if self.condition_on_decision:
            decision_features = self._decision_features(baseline_logits)
            fused = torch.cat([fused, decision_features], dim=-1)
        residual_logits = self.classifier(fused)
        logits = baseline_logits + residual_logits
        if not return_details:
            return logits
        return logits, {
            "decision_features": decision_features,
            "residual_logits": residual_logits,
            "fused_embedding": fused,
            "modality_mask": mask,
            "modality_weights": modality_weights,
            "reliability_logits": reliability_logits,
            "reliability_scores": reliability_scores,
            "consistency_logits": consistency_logits,
            "consistency_scores": consistency_scores,
            "text_embedding": text,
            "image_embedding": image,
        }
