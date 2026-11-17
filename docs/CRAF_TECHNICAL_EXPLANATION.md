# CRAF development: rationale, code, and experimental results

**Technical explanation of the v1/v2 decision correction added to HAMMER**  
Muhammed Emir Koçak - YZM5110 - November 13, 2026

## 1. What did we develop, and why?

CRAF is the name of the small network in this project that uses HAMMER's image and text representations to add a learned correction to its binary classification decision. It is not a component of the original HAMMER paper. This document does not claim that we introduced the name CRAF to the literature or established the scientific novelty of the design.

Our research question is: “Can a small network using summary representations of image and text correct some errors without updating the trained HAMMER model again?” HAMMER produces a strong decision from images, text, and their fusion. CRAF adds two example-dependent numbers to that decision. These numbers are learned from labeled training data rather than specified by hand.

In the project, HAMMER was first retrained on DGM4 from ALBEF initialization weights. The supplied log contains 11 completed epochs, indexed 0–10, against a planned 25. The selected HAMMER model was then frozen and CRAF was trained. “Without retraining HAMMER” describes only this second stage; it does not eliminate the earlier HAMMER training.

## 2. Where are the two modalities, and where does CRAF connect?

```text
Image -> HAMMER ViT  -> image CLS [B, 768] --+
Text  -> HAMMER BERT -> text CLS  [B, 768] --+-> CRAF -> delta [B, 2]
                                          |
HAMMER multimodal fusion -> logits [B, 2] -+-> addition -> final logits

HAMMER stage: encoders and task heads are trained.
CRAF stage: HAMMER is fixed; only CRAF parameters are trained.
```

B is the number of examples in a mini-batch. CLS is a global vector summarizing an image or text sequence. The two vectors used by CRAF are taken before HAMMER's cross-modal fusion. Baseline logits come from HAMMER's own multimodal decision. CRAF has distinct text and image inputs; it is not a model receiving only two class scores.

| Input / output | Shape | Meaning |
|---|---|---|
| `text` | B × 768 | HAMMER text CLS representation |
| `image` | B × 768 | HAMMER image CLS representation |
| `baseline_logits` | B × 2 | HAMMER scores for real and manipulated classes |
| `residual_logits` | B × 2 | Learned additions/subtractions produced by CRAF |
| `logits` | B × 2 | Final decision scores |

Class order is 0=real/orig, 1=manipulated. A logit is a score before conversion to a probability. Final probabilities are computed with `softmax(logits)`.

CRAF does not update HAMMER's manipulation-type, box, or token heads. This development is **binary decision correction using image and text representations**, not a new encoder or image/text grounding method.

## 3. How does V1 compute the correction, step by step?

### 3.1. Reducing the two representations

Separate learnable projections are used for image and text. Each maps 768 dimensions to 256; LayerNorm controls scale, GELU provides a nonlinear transformation, and dropout randomly zeros some activations during training. The two projections do not share weights.

Source: `gpu/craf_residual.py`. The block below is taken from the actual code for the described component; it is not a standalone program.

```python
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
```

### 3.2. Modality weights

Each 256-dimensional representation passes through a linear layer to produce a score. The code applies sigmoid to these scores, then normalizes the two positive values to sum to 1. Numerically, the implementation computes the same result through `softmax(logsigmoid(...))`.

```text
r_text  = sigmoid(a_text)
r_image = sigmoid(a_image)
w_text  = r_text / (r_text + r_image)
w_image = r_image / (r_text + r_image)
```

For example, raw sigmoid scores of 0.8 and 0.4 give weights of approximately 0.667 and 0.333. This gives the weighted text branch more influence for that example. However, these scores are not actual “reliability probabilities”; they are not trained against reliability labels. The learning signal comes from the final classification loss. Since difference and product branches are also used, the weights alone do not explain the model's entire decision.

Source: `gpu/craf_residual.py`. The block below is taken from the actual code for the described component; it is not a standalone program.

```python
            reliability_logits = torch.cat(
                [self.text_reliability(text), self.image_reliability(image)], dim=-1
            )
            reliability_scores = torch.sigmoid(reliability_logits) * mask
            log_scores = nn.functional.logsigmoid(reliability_logits)
            log_scores = log_scores.masked_fill(mask == 0, torch.finfo(mask.dtype).min)
            modality_weights = torch.softmax(log_scores, dim=-1) * mask
```

### 3.3. Constructing cross-modal relationship features

The absolute difference and elementwise product of the two vectors are computed. The difference provides inputs representing separation in the learned space; the product provides inputs representing co-activation. They are not assumed to measure semantic consistency automatically. Text and image do not naturally share a coordinate system; separate projections are learned jointly through the task loss.

Source: `gpu/craf_residual.py`. The block below is taken from the actual code for the described component; it is not a standalone program.

```python
        difference = (text - image).abs()
        product = text * image
        if not self.use_interactions:
            difference = torch.zeros_like(difference)
            product = torch.zeros_like(product)
        pair_features = torch.cat([text, image, difference, product], dim=-1)
```

Another small network maps this 4 × 256-dimensional pair representation to a sigmoid `consistency_scores` value between 0 and 1. This is also an auxiliary feature learned only through classification loss; it is neither a verified text–image match label nor an explainability guarantee.

### 3.4. Concatenation and correction

Weighted text, weighted image, absolute difference, and product total 1024 dimensions. One consistency score, two raw sigmoid scores, and two presence/mask values are appended, giving V1 a **1029-dimensional** input. Both modalities are always present in the current implementation; mask values are [1,1]. Having a mask field does not demonstrate robustness to missing modalities.

Source: `gpu/craf_residual.py`. The block below is taken from the actual code for the described component; it is not a standalone program.

```python
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
```

`self.classifier` is a 1029 → 256 → 2 MLP. The final two numbers can be read as `delta_real` and `delta_fake`; they may be positive or negative. The core equation is:

```text
z_final = z_HAMMER + delta_CRAF
p_fake = softmax(z_final)[1]
```

In V1, the correction is computed from image/text features; baseline logits are not a direct input to the correction network and are added at the end. V2 decision conditioning changes this point.

### 3.5. Matching HAMMER before learning begins

The final linear layer's weights and bias are initialized to zero. Initially, the residual is therefore zero and final logits are identical to HAMMER's. This does not guarantee improvement after training. During the first backward pass, the final layer learns first; gradients to preceding layers through the zero final layer are initially zero. Earlier layers can learn as the final layer is updated.

Source: `gpu/craf_residual.py`. The block below is taken from the actual code for the described component; it is not a standalone program.

```python
        nn.init.zeros_(self.classifier.net[3].weight)
        nn.init.zeros_(self.classifier.net[3].bias)
```

## 4. Where do the correction values come from? A numerical example

The following values are illustrative; they do not come from an actual test news item.

```text
HAMMER logits:      [1.2, 0.8]  -> p_fake = 0.4013 -> real
CRAF residual:     [-0.1, 0.6]
Final logits:       [1.1, 1.4]  -> p_fake = 0.5744 -> manipulated

HAMMER margin: 0.8 - 1.2 = -0.4
Correction margin: 0.6 - (-0.1) = +0.7
Final margin: -0.4 + 0.7 = +0.3
```

Why CRAF produces [-0.1, 0.6] depends on that example's two modality representations and the learned network weights. There is no single hand-written correction formula: the forward pass computes these numbers, and training updates the parameters that produce them.

If the true label is manipulated, this change reduces the example's cross-entropy from approximately 0.9130 to 0.5544. If the true label is orig, the same change is wrong and increases the loss. The network tries to reduce the aggregate loss over training examples; it is not guaranteed to correct every example.

Adding the same value to both binary logits does not change probabilities. For example, adding [+5,+5] leaves the margin unchanged. The V2 penalty therefore targets the difference between the two correction values rather than their absolute magnitudes.

## 5. What did we add in V2?

### 5.1. Supplying HAMMER's decision to the correction network

When `condition_on_decision=True`, V2 appends three numbers to the 1029-dimensional vector, producing a **1032-dimensional** input. These are manipulated-class probability, bounded signed logit margin, and normalized binary entropy.

```text
p_fake = softmax(z)[1]
margin = z_fake - z_real
bounded_margin = margin / (1 + abs(margin))
entropy = -sum(p * log(p)) / log(2)
```

The signed margin carries decision direction; its absolute magnitude reflects separation of the two class scores. Entropy closer to 1 indicates greater indecision between the classes. That indecision does not tell us the true class.

Source: `gpu/craf_residual_v2.py`. The block below is taken from the actual code for the described component; it is not a standalone program.

```python
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
```

Here, `detach()` blocks gradients to the baseline through these three features. However, this line alone does not freeze all of HAMMER: freezing is enforced by the feature extractor and cache workflow. The head class itself owns no encoders.

### 5.2. Penalizing large decision changes

Source: `gpu/craf_residual_v2.py`. The block below is taken from the actual code for the described component; it is not a standalone program.

```python
def correction_penalty(residual_logits: Tensor) -> Tensor:
    """Mean squared binary margin correction, invariant to shared-logit shifts."""
    return (residual_logits[..., 1] - residual_logits[..., 0]).square().mean()
```

```text
L_v1 = CE(z_HAMMER + delta, y)
L_v2 = CE(z_HAMMER + delta, y) + lambda * mean((delta_fake - delta_real)^2)
```

The combined experiment uses lambda=0.01. In the single example above, the correction margin is 0.7, giving a penalty of 0.49 and an added loss of 0.0049. This encourages the model to avoid unnecessarily large decision changes. It does not guarantee calibration or higher accuracy.

Source: `gpu/hammer_craf_v2_server.py`. The block below is taken from the actual code for the described component; it is not a standalone program.

```python
                corrected, details = model(text.to(device), image.to(device), baseline_logits.to(device), return_details=True)
                ce = nn.functional.cross_entropy(corrected, labels.to(device))
                penalty = correction_penalty(details['residual_logits'])
                loss = ce + args.correction_lambda * penalty
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
```

In the code, `loss.backward()` computes gradients, gradient clipping limits the norm to 1, and `optimizer.step()` changes only the CRAF parameters supplied to the optimizer. The long `model(...)` line may be visually wrapped in the PDF to save space; the copyable source is in Markdown.

### 5.3. Four V2 controls

| Condition | Decision information | Penalty lambda | Question |
|---|---|---:|---|
| control | Off | 0 | A V1-like baseline comparison |
| penalty | Off | 0.01 | What does the penalty alone change? |
| decision | On | 0 | Is decision information alone useful? |
| combined | On | 0.01 | Are the two useful together? |

All four runs have validation results. A separate test output exists only for combined. Equality of low-LR V1 and V2 control validation results does not replace the missing control test output.

## 6. How do we keep HAMMER fixed?

The feature extractor strictly loads the trained HAMMER checkpoint, switches to evaluation mode, and disables gradients for all parameters:

Source: `gpu/hammer_craf_server.py`. The block below is taken from the actual code for the described component; it is not a standalone program.

```python
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval().requires_grad_(False)
```

Extraction runs inside `torch.inference_mode()`. For each example, the two CLS vectors, HAMMER logits, and true label are cached. The label is the dataset's orig/manipulated annotation; it is not appended to model input features. Deterministic evaluation transforms are used for train, validation, and test; no new image augmentation occurs during head training on the cache.

```python
# Illustrative summary; the actual source flow is in gpu/hammer_craf_server.py.
with torch.inference_mode():
    # HAMMER's forward pass and hooks extract the two CLS vectors.
    text_cls, image_cls, baseline_logits = extract_features(batch)

# Reading from cache: HAMMER no longer runs in this training loop.
corrected = craf(text_cls, image_cls, baseline_logits)
loss = cross_entropy(corrected, labels)
```

This block is educational pseudocode: there is no executable project function named `extract_features`. In the actual extractor, hooks fill `capture['text']` and `capture['image']`; binary logits are taken from `outputs[0]`.

The head optimizer receives only the small network's parameters:

Source: `gpu/hammer_craf_server.py`. The block below is taken from the actual code for the described component; it is not a standalone program.

```python
        model = FrozenCRAFResidual(**config['head']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```

Thus, CRAF training is not HAMMER retraining. A new HAMMER checkpoint or different preprocessing invalidates the old feature cache and requires new extraction. Manifests record backbone, tokenizer, preprocessing, and source-file hashes to check this match.

## 7. A small runnable usage example

This example does not predict an actual news item. It demonstrates the head interface, initial parity, and shapes using synthetic representations. It uses CUDA if available and can be run from the project root.

```python
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path.cwd() / 'gpu'))
from craf_residual_v2 import FrozenCRAFResidual, correction_penalty

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
head = FrozenCRAFResidual(condition_on_decision=True).to(device).eval()
text = torch.randn(2, 768, device=device)
image = torch.randn(2, 768, device=device)
baseline = torch.tensor([[1.2, 0.8], [0.2, 1.1]], device=device)

with torch.no_grad():
    corrected, details = head(text, image, baseline, return_details=True)
    assert torch.equal(corrected, baseline)  # Because of zero initialization.
    assert details['fused_embedding'].shape == (2, 1032)
    assert correction_penalty(details['residual_logits']).item() == 0.0
    print(torch.softmax(corrected, dim=-1)[:, 1])
```

To observe a trained correction, strictly load the appropriate CRAF checkpoint's `model` state_dict. Input representations must also have been produced by the HAMMER model and preprocessing specified in that checkpoint's manifest. The synthetic example's output is not a performance metric.

## 8. How did we train, and what did we obtain?

All head experiments used seed 42, batch 512, hidden size 256, dropout 0.2, and AdamW weight decay 0.01. V1 high LR used 1e-4/patience 5, low LR used 1e-5/patience 8, and all four V2 runs used 1e-5/patience 8. The target was 30 epochs; completed counts differ because of early stopping. Checkpoints are selected by validation AUC.

V1 has 922,117 additional trainable parameters; decision-conditioned V2 has 922,885. These are not HAMMER parameter counts. Heads run on a single GPU using an existing cache; four-GPU HAMMER retraining was a separate stage.

| Model / condition | Validation AUC | Validation CE | Selected epoch (one-based) |
|---|---:|---:|---:|
| HAMMER cache baseline | 0.92324647 | 0.541329 | HAMMER 11 |
| V1 high LR | 0.92502938 | 0.582958 | 2 |
| V1 low LR / V2 control | 0.92493543 | 0.584701 | 9 |
| V2 penalty | 0.92385134 | 0.547143 | 16 |
| V2 decision | 0.92497965 | 0.570732 | 3 |
| V2 combined | 0.92382936 | 0.547165 | 13 |

Decision information increased validation AUC by only 0.00004422 over control. The penalty reduced CE deterioration but also reduced the AUC gain. CE remained worse than HAMMER's.

| Test model | Accuracy % | AUC % | CE | Net additional correct predictions |
|---|---:|---:|---:|---:|
| HAMMER baseline | 85.3427 | 92.6660 | 0.526091 | - |
| CRAF v1, 1e-4 | 85.5142 | 92.8615 | 0.563057 | +87 |
| CRAF v1, 1e-5 | 85.5340 | 92.8592 | 0.564652 | +97 |
| CRAF v2 combined | 85.4156 | 92.7468 | 0.530419 | +37 |

The test contains 50,705 examples. Low-LR V1 fixed 730 HAMMER errors while introducing 633 new errors: net 97. The accuracy increase is approximately **0.1913 percentage points**. Correcting more examples does not mean no new errors were introduced. Since patience also changed in the V1 LR comparison, the gain cannot be attributed solely to LR.

CE worsened relative to baseline in all test conditions; combined also reduced balanced accuracy below baseline. Therefore, “CRAF is better in every respect” is not supported. Initial test results influenced later design decisions, making subsequent comparisons exploratory. A single seed, multiple examples per source news item, and the absence of an untouched final holdout limit generalization claims. Statistical significance, calibration, or robustness gains were not demonstrated.

## 9. What is in each file?

| File | Role |
|---|---|
| `gpu/craf_residual.py` | V1 projections, weights, relationship features, and residual network |
| `gpu/craf_residual_v2.py` | Decision features and margin-correction penalty |
| `gpu/hammer_craf_server.py` | HAMMER loading/freezing, cache extraction, V1 train/test |
| `gpu/hammer_craf_v2_server.py` | V2 train/test and the CE + penalty objective |
| `gpu/run_craf_v2_experiments.sh` | Launching the four V2 controls |
| `results/server_records/` | Original server run/history/validation/test records |
| `results/server_analysis.json` | Record/metric audit of six runs and three raw tests |
| `results/HAMMER_TRAINING_SUMMARY.md` | Summary of HAMMER training/testing before CRAF |
| `docs/GPU_USAGE.md` | Commands for real cache, training, and testing workflows |

These paths are relative to the package root. Actual code excerpts in Markdown can be checked against the named source files. The numerical example and synthetic usage test are explanatory and are not included in experimental results.

## 10. A short explanation for the instructor

“We first retrained HAMMER, which uses images and text, on DGM4. We then fixed HAMMER's summary representations for both modalities and its decision scores. A small network named CRAF learned corrections to add to the two class scores from those representations. In the second version, we added HAMMER's existing decision information and penalized large margin corrections. We observed small accuracy/AUC gains, but probabilistic loss worsened and the second version did not provide consistent superiority.”

The key distinction is that images and text are CRAF inputs; the component we developed learns **binary residual decision correction** from these representations. We do not claim to improve encoders or grounding heads during the CRAF stage.
