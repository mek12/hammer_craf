# The two modalities and the point of intervention

```text
Image -> HAMMER ViT  -> image CLS -----------------------+
Text  -> HAMMER BERT -> text CLS ------------------------+-> CRAF -> delta
                                                        |
HAMMER multimodal fusion -> two binary logits ----------+-> final logits = z + delta
```

HAMMER retraining updates the encoders and task heads. During the CRAF stage,
the entire trained HAMMER model is frozen; image/text CLS vectors and baseline
logits are extracted once and cached. CRAF sees more than the final score: it
uses separate representations of both modalities. V2 decision inputs are also
computed from the existing baseline logits.

The project therefore uses at least two modalities, but the developed component
is not an encoder or grounding mechanism. Its scope is
**binary decision correction over frozen multimodal representations**.

CRAF v1:

1. Project each modality from 768 to 256 dimensions.
2. Normalize two learned scores to obtain modality weights.
3. Concatenate text/image vectors, absolute differences, products, a consistency
   score, two scores, and presence information.
4. Use an MLP to produce residual logits for two classes. Zero initialization of
   the final layer reproduces the HAMMER prediction exactly before learning.

CRAF v2 optionally adds HAMMER p_fake, signed margin, and entropy information.
The penalty option adds the squared margin correction to the classification
loss. The two options can be disabled independently. Neither is assumed to
improve performance.

Observed behavior: the initial v1 variants yield small binary classification
gains and worse cross-entropy; combined reduces the CE deterioration while also
reducing the classification gain. This alone does not establish effective use
of both modalities or calibrated scores. Four validation ablations and paired
analysis of raw predictions from three tests are available; see
`SERVER_RESULTS_ANALYSIS.md`. Stronger reproducibility assessment requires
additional seeds and mappings at the source-news level.
