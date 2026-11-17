"""Standalone CRAF residual head over two already-computed embeddings.

The caller owns encoder freezing/detaching and supplies finite floating-point
inputs. This module owns no encoders and preserves input autograd connections.
Its gates receive supervision only through the caller's binary cross entropy
on corrected logits; gate values are not calibrated reliability estimates.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


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
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * hidden_dim + 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class FrozenCRAFResidual(nn.Module):
    """Add a zero-initialized CRAF correction to binary baseline logits.

    State keys/shapes match CRAFClassifier with equal text/image dimensions
    and two classes. Both modalities are always present. Ablations retain all
    parameters for strict checkpoint compatibility. Loading an existing CRAF
    checkpoint replaces the zero initialization with its learned correction.

    Shapes are text/image ``[batch, feature_dim]`` and baseline ``[batch, 2]``.
    Inputs must share floating dtype/device. Finite values are a caller
    precondition: forward does no tensor-wide scans or host synchronization.
    """

    def __init__(
        self,
        feature_dim=768,
        hidden_dim=256,
        dropout=.2,
        use_reliability=True,
        use_consistency=True,
        use_interactions=True,
    ):
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.feature_dim = feature_dim
        self.use_reliability = use_reliability
        self.use_consistency = use_consistency
        self.use_interactions = use_interactions
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
        self.classifier = _Classifier(hidden_dim, dropout)
        nn.init.zeros_(self.classifier.net[3].weight)
        nn.init.zeros_(self.classifier.net[3].bias)

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
        residual_logits = self.classifier(fused)
        logits = baseline_logits + residual_logits
        if not return_details:
            return logits
        return logits, {
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
