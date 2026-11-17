# HAMMER-centered paper reading guide

Preparation and source-access date: **2026-11-11**. These cards are preliminary notes to guide the student's personal reading; they do not assume the student has read the papers. The main paper is **R1 HAMMER**. R2–R5 are supplementary; R6 is an optional preprint. Bibliographic details, versions, and unverified items are in `LITERATURE_MATRIX.md`; BibTeX keys are `R1`–`R6`.

Reading order: R1 → R2 → R3 → R4 → R5 → optional R6. In each card, **Source finding** denotes the authors' model/experiment description; **Our interpretation** denotes an inference for this project. Full-text review covers the relevant method and experiment sections; it does not mean every equation or implementation was independently verified. Cards can be included vertically in a PDF appendix; do not shrink the wide matrix for printing.

## Fixed project definition

HAMMER is frozen and in eval mode. Pre-fusion image/text CLS vectors enter CRAF. The two-class CRAF correction is added to HAMMER binary logits; the final layer is zero-initialized. Type, image-box, and text-token heads are preserved. The learned intervention is a binary classification residual. Joint training, local alignment, frequency, or attention results from the literature cannot be presented as evidence of this intervention's success. This guide reports no new project results.

## R1 — HAMMER: main reading

**Identity and depth.** Shao, Wu, Liu; CVPR 2023. [2304.02556v1 full text](https://arxiv.org/html/2304.02556v1), [PDF](https://arxiv.org/pdf/2304.02556v1). Method, experiments/ablations, and appendix were reviewed. [Official publication record](https://openaccess.thecvf.com/content/CVPR2023/html/Shao_Detecting_and_Grounding_Multi-Modal_Media_Manipulation_CVPR_2023_paper.html).

**Research question and contribution — source finding.** Can a model jointly learn real/fake decisions for image–text pairs, manipulation types, and the locations of altered faces/words? The paper introduces both the DGM4 task/dataset and the HAMMER model.

**Mechanism and objective — source finding.** The shallow stage uses manipulation-aware contrastive alignment; the deep stage uses modality-aware cross-attention. The joint objective covers binary classification, manipulation type, box, and token tasks. Training labels are not inference inputs. [§4 and Appendices A–B](https://arxiv.org/html/2304.02556v1).

**Evidence and protocol — source finding.** The paper compares adapted CLIP/ViLT models, evaluates unimodal subsets, and includes modality/loss removal tables. DGM4's total example count is not the training-set size. Official split counts were not verified here. [§5, Tables 2–7](https://arxiv.org/html/2304.02556v1). The [official repository](https://github.com/rshaojimmy/MultiModal-DeepFake) includes train/test code and a checkpoint link; reproduction was not performed.

**Our interpretation.** Data generation and task definition may jointly affect method performance. Changes in other tasks after removing one task loss show the influence of joint training, but do not establish a specific semantic causal explanation alone. Uses beyond human-centered news and selected editing tools require separate evaluation.

**Student reading targets.** Start with Figure 1 and §3: explain an FS/FA/TS/TA example in your own words. Then use §4/Figure 3 to draw which representation feeds each head. In §5 Tables 2–7 and Appendix B, mark full-set/subset and metric distinctions. Finally, check whether Appendix A and repository setup describe the same configuration.

**Relationship to / difference from a frozen CLS residual.** R1 is our direct baseline. We do not realign R1 representations; we learn an extra correction to its binary decision. **Our interpretation:** claiming grounding improvement from training CRAF alone exceeds the intervention's scope. The student should leave open whether gains come from CLS interactions or additional classifier capacity.

## R2 — HAMMER++: distinguish local alignment

**Identity and depth.** Shao, Wu, Wu, Nie, Liu; TPAMI 2024. [2309.14203v1 full-text HTML](https://arxiv.org/html/2309.14203v1), especially III-D, IV-A, and V-A–C. The PDF could not be retrieved because of its size. Venue/year are confirmed in the [authors' repository](https://github.com/rshaojimmy/MultiModal-DeepFake#citation); the publication DOI match was not verified because of publisher access restrictions.

**Research question — source finding.** Does adding a local view to R1's global contrastive alignment represent subtle manipulation traces better? The contribution is the local MAC-L extension within the same research line.

**Mechanism and objective — source finding.** Alignment of global CLS with local patch/token representations in the other modality is added to MAC-G and task losses. This changes the representation learned by the encoders. [IV-A, especially local-view equations](https://arxiv.org/html/2309.14203v1).

**Evidence and protocol — source finding.** Tables VII–VIII include global/local loss controls; IX covers momentum. III-D describes an approximately 7:1:2 source-ID split and text deduplication. [V-C and III-D](https://arxiv.org/html/2309.14203v1). The shared code repository is accessible; a separate MAC-L implementation and HAMMER++ checkpoint match were not verified in this review.

**Our interpretation.** A controlled addition within the same data/model family is meaningful for assessing a mechanism, but does not replace external validity on an independent dataset. Improvement in one metric does not require equal improvement across all heads. Average gains should not be universalized without uncertainty estimates.

**Student reading targets.** Read III-D for split leakage. In IV-A, list MAC-G/MAC-L positives and negatives separately. In Tables VII–VIII, locate rows changing only one component and check the direction of the compared metric. Do not treat the resource budget in V-A as identical to our run.

**Relationship to / difference from a frozen CLS residual.** CRAF does not learn local patch/token alignment; it operates on two frozen CLS vectors. **Our interpretation:** R2 raises the question of what global summaries lose. Transferring MAC-L's benefit to CRAF is inappropriate. The student should explicitly state which information CRAF never accesses.

## R3 — UFAFormer: the frequency view

**Identity and depth.** Liu, Tan, Chen, Wei, Zhao, Wang. [Publisher record](https://link.springer.com/article/10.1007/s11263-024-02245-x): IJCV 2025, online October 7, 2024. Technical reading: [2309.09667v1 HTML](https://arxiv.org/html/2309.09667v1) and [PDF](https://arxiv.org/pdf/2309.09667v1), §3–4. The final journal full text was unavailable; differences from v1 were not verified.

**Research question and mechanism — source finding.** Are image-derived DWT subbands useful alongside RGB? The model uses intra-/inter-band attention, a forgery-aware mutual module (FAMM), and a unified decoder. Its objective includes classification, L1+GIoU box loss, and token focal loss. [§3](https://arxiv.org/html/2309.09667v1). The frequency branch remains part of the inference architecture.

**Evidence/protocol/code — source finding.** DGM4; Table 1 distinguishes test from †val. Component and frequency-attention ablations appear in §4.3. [§4](https://arxiv.org/html/2309.09667v1). The [official repository](https://github.com/Michel-liu/UFAFormer) was found, but only LICENSE was visible at its root; it cannot be described as released implementation code. GPU model/count were not verified.

**Our interpretation.** A derived frequency view is not a new independent information source; it is another representation of the same pixels. Compression, resizing, and blur can change the distribution of these traces. Selected perturbation experiments do not establish robustness to every corruption or robustness of our model. A unified decoder does not necessarily imply lower total computation.

**Student reading targets.** Follow the relationship between DWT and position in §3.2, the RGB–frequency conflict rationale in §3.3, and task queries in §3.4. Read §4.1 and Table 1 footnotes before the results. Explain why removing FAMM and removing the frequency branch in §4.3 are different controls.

**Relationship to / difference from a frozen CLS residual.** We have no DWT encoder or new local queries. **Our interpretation:** CRAF does not implement this frequency mechanism. The question is which errors can be corrected using existing global representations instead of an additional view; the source paper cannot answer that for our model.

## R4 — ASAP: auxiliary information during training

**Identity and depth.** Zhang, Wang, Cheng, Zhong, Guo, Wang; [CVPR 2025 official record](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_ASAP_Advancing_Semantic_Alignment_Promotes_Multi-Modal_Manipulation_Detecting_and_Grounding_CVPR_2025_paper.html). Technical reading: [2412.12718v2 HTML](https://arxiv.org/html/2412.12718v2), §3–4 and supplementary §6–7. This version is dated October 22, 2025; equality with the accepted text was not verified.

**Research question and mechanism — source finding.** Can better semantic alignment be learned for manipulated pairs? LMA uses MLLM caption and LLM explanation generation; MGCA is GT-guided; PMM is an auxiliary objective. The total objective is DGM+LMA+αMGCA+λPMM. §3 states that the proposed auxiliary modules are used only in training. [§3, Equation 13](https://arxiv.org/html/2412.12718v2).

**Evidence/resources — source finding.** §4.4 includes a retrained HAMMER control and component, caption/explanation, and negative-selection ablations. §4.1 reports eight RTX 4090 GPUs. The [official repository](https://github.com/CriliasMiller/ASAP) contains actual train/test, model, and large-model scripts, rather than just a promise of future code. Execution and auxiliary-data integrity were not verified.

**Our interpretation.** Generated training text can introduce both extra information and generator-model bias. Removing an auxiliary objective at inference does not eliminate preparation and training costs. Equalizing parameter count alone may be insufficient when auxiliary information budgets differ. Attention visualization alone does not establish an explanation's causal correctness.

**Student reading targets.** Mark the training/inference distinction in §3; separate the inputs used to generate captions and explanations in §3.1. Trace which loss uses GT in §3.2. Read §4.4 Tables 2–4 and §6–7; distinguish the retrained baseline from the baseline in the original paper.

**Relationship to / difference from a frozen CLS residual.** We do not generate data with large models or update the backbone with MGCA or PMM. **Our interpretation:** using modality relationships is a conceptual commonality, but the training interventions differ. ASAP's findings cannot be converted into a claim that the residual layer improves alignment.

## R5 — CSCL: explicit local relationship supervision

**Identity and depth.** Li, Yang, Tan, Liu, Chen, Zhou, Lei; [CVPR 2025 record](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Unleashing_the_Potential_of_Consistency_Learning_for_Detecting_and_Grounding_CVPR_2025_paper.html). [2506.05890v1 HTML](https://arxiv.org/html/2506.05890v1), §3–4; limitation and loss sections in the official supplement were reviewed. The [IEEE publication DOI record](https://ieeexplore.ieee.org/document/11093344/) was also checked.

**Research question and mechanism — source finding.** Does explicit supervision of intra-modal contextual and cross-modal semantic consistency help locate fake content? Image and text branches use CCD followed by SCD; consistency matrices are learned from pair relationships. Contextual/semantic consistency objectives supplement task losses. [§3.2–3.4](https://arxiv.org/html/2506.05890v1).

**Evidence/protocol — source finding.** §4.4 contains decoder removal experiments; §4.5 analyzes design choices. The [official README](https://github.com/liyih/CSCL#data-preparation) states that released CSCL weights were trained on **train+val** and evaluated on **test**. Code/weight links exist; downloading and reproduction were not performed. §4.2 reports eight A100 GPUs.

**Our interpretation.** Local pair-label accuracy and ambiguous boundary patches can affect learned consistency. The contribution of local representations must be separated from that of large pretrained models. A train+val checkpoint cannot be presented as using the same training protocol as a train-only method. The [official supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Li_Unleashing_the_Potential_CVPR_2025_supplemental.pdf) shows small-face/failure cases and identifies future investigation of corruption robustness.

**Student reading targets.** Draw how local pair labels are constructed in §3.2–3.3; read §3.4 alongside supplementary loss definitions. Mark each removed decoder in §4.4. Note the README's train+val statement before reading the result table.

**Relationship to / difference from a frozen CLS residual.** CRAF does not learn these matrices or token/patch decoders. **Our interpretation:** a connection between global CLS correction and fine-grained localization evidence remains an assumption. CSCL does not establish grounding or robustness superiority for our CRAF.

## R6 — FMS: optional preprint

**Identity and depth.** Yu, Lu, Luo; *Fine-grained Multiple Supervisory Network for Multi-modal Manipulation Detecting and Grounding*. [2508.02479v1 HTML](https://arxiv.org/html/2508.02479v1), [PDF](https://arxiv.org/pdf/2508.02479v1), August 4, 2025. Method and experiment sections were reviewed; **this was not abstract-only reading**. A final peer-reviewed venue and official code were **not verified**. It is therefore not assigned the same publication status as R1–R5.

**Research question and mechanism — source finding.** Can unreliable unimodal decisions be prevented from damaging the joint decision? MDSC supervises and corrects modality decisions; UFMR separates fake traces at feature/example levels; MFAR learns consistent and inconsistent interactions. Auxiliary and DGM4 task objectives are jointly optimized. A modality-specific authenticity label is not necessarily the same as the pair's binary label. [Methodology](https://arxiv.org/html/2508.02479v1).

**Evidence/protocol — source finding.** DGM4 experiments, final-epoch selection, and eight A100 GPUs are reported; Table 4 removes the three components separately. Exact split manifests were not verified. [Experiments](https://arxiv.org/html/2508.02479v1). Result reproduction remains open because code/weight availability could not be verified.

**Our interpretation.** Defining reliability through a learned decision does not automatically establish calibration or reliability under distribution shift. Correct label derivation when one modality is altered requires separate checks. Multiple auxiliary objectives must be distinguished from additional capacity/optimization effects. We do not turn the authors' preprint superiority claims into a cross-protocol SOTA judgment.

**Student reading targets.** First distinguish unimodal and pair labels in MDSC. Follow feature/example levels in UFMR and masks/interaction constraints in MFAR. Match Full Objectives to Table 4, then note the final-epoch rule in Implementation Details.

**Relationship to / difference from a frozen CLS residual.** Correction and reliability are related concepts; FMS also trains local patch/token paths alongside post-interaction CLS. We produce a binary logit residual from frozen pre-fusion CLS. **Our interpretation:** conceptual similarity establishes neither algorithmic identity nor novelty of our approach.

## Notes for the student to complete after reading

For each source, write a paragraph distinguishing the authors' question, the mechanism in one sentence, the strongest controlled experiment, data/protocol limitations, and the information or learning pathway absent from our fixed intervention. Then pose an open question: which experiment could falsify this mechanism explanation?

When interpreting our results, first match checkpoints, train/val/test manifests, pretraining, example subsets, thresholds, and metric aggregation. A number or robustness statement from the literature is not an experiment on our method. Novelty, SOTA, and robustness are not conclusions reached by these cards.
