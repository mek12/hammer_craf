# Attribution and Third-Party Components

## Method and dataset

Rui Shao, Tianxing Wu and Ziwei Liu, *Detecting and Grounding Multi-Modal Media
Manipulation*, CVPR 2023.

- Official source: https://github.com/rshaojimmy/MultiModal-DeepFake
- Reproduction target commit: `33fc04f3a1f550f923848737c864eea32556f699`

This repository does not contain the complete official model code. Training, tokenizer
adaptation, evaluation, data transforms and the HAMMER computations come from the
official repository, which is used alongside this project (`REPO_DIR`). The S-Lab License
1.0 text is kept in [`docs/THIRD_PARTY_SLAB_LICENSE.txt`](docs/THIRD_PARTY_SLAB_LICENSE.txt).
Its terms, including the non-commercial-use condition, apply. No separate MIT or Apache
licence is claimed for the project as a whole.

## Initial weights

The initial weights come from the ALBEF project: https://github.com/salesforce/ALBEF.
They are used to re-train HAMMER. The CRAF stage then freezes the re-trained HAMMER
checkpoint. No weights are included in this repository.

## Data access

DGM4 is available from the official Hugging Face dataset:
https://huggingface.co/datasets/rshaojimmy/DGM4. Its current access and licence terms
must be checked separately. The licence of the source repository does not grant
unrestricted redistribution rights for the news images.

## This project

CRAF is a residual classification experiment designed and implemented for this course
project. It is not presented as a published method or as a first contribution in the
literature. AI assistance is disclosed in [`AI_USE.md`](AI_USE.md).
