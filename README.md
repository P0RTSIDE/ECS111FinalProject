# Cross-Generator Review Detector

A binary classifier for Amazon product reviews: human-written versus model-written. The evaluation asks whether a detector trained on some generators still catches reviews from a generator it never saw.

ECS 111 final project. Code and the review tables are in git. Checkpoints, run logs, and Weights & Biases files are gitignored and can be rebuilt.

## Data

Human reviews come from the McAuley Lab Amazon Reviews 2023 corpus (`notebooks/01_collect_human_reviews.ipynb`).

- English only
- 20 to 400 words
- Dated on or before 2022-11-30, before the public ChatGPT release
- Reservoir-sampled across 8 product categories
- Written to `data/raw/human_reviews.csv`

Model reviews are customer-style text for products drawn from that human pool. Target lengths follow the human length distribution. Each generator contributes on the order of 2,500 reviews, appended to `data/generated/ai_reviews.csv` with a `generator` column.

| Generator | Notebook | API |
| --- | --- | --- |
| GPT-5 mini | `notebooks/01b_generate_gpt5mini.ipynb` | OpenAI |
| DeepSeek-V4-Flash | `notebooks/01c_generate_open_models.ipynb` | Hugging Face Inference Providers |
| Gemma-4-31B-it | same | same |
| Qwen3.6-35B-A3B | same | same |

The study design originally named IBM Granite as the fourth generator. DeepSeek-V4-Flash is used instead because the Hugging Face router does not serve a Granite model. Qwen and DeepSeek are reasoning models. Generation turns thinking off and strips leftover reasoning traces so the saved text is the review.

The pool is about 10,000 human reviews and about 10,000 model reviews. Counts are not rounded to those figures: non-English rows and contaminated generations were dropped (on the order of 0.3 percent of human rows and 0.1 percent of model rows). Downstream code does not assume a fixed 2,500 or 10,000.

## Leave-one-generator-out splits

`notebooks/02_build_splits.ipynb` builds four rotations. Rotation order is `gpt5mini`, `deepseek`, `gemma`, `qwen`. Rotation `r` holds out `GENERATORS[r]`.

Reviews from the three training generators are split 80/10/10:

| File | Contents |
| --- | --- |
| `train.csv` | 80 percent of in-distribution model reviews, plus an equal number of human reviews |
| `val_indist.csv` | 10 percent, for tuning and early stopping, balanced 1:1 |
| `test_indist.csv` | 10 percent from generators seen in training, balanced 1:1 |
| `test_crossgen.csv` | Every review from the held-out generator, plus a human pool that does not overlap training |

The human pool is cut after the model split. Training takes as many human reviews as there are in-distribution model reviews. The remainder is the cross-generator human pool, so that test measures an unseen generator rather than memorized human text. The notebook checks that no `review_id` is in both `train.csv` and `test_crossgen.csv`.

Files: `data/splits/rotation_{0..3}/`.

## Model

Encoder: `microsoft/deberta-v3-base`. Token states are mean-pooled into one review vector.

Three heads sit on that vector. Only the classification head is the detector. The other two shape the encoder.

- **Classification.** Linear map to two classes (human, model), trained with cross-entropy.
- **Contrastive projection.** Linear map to 128 dimensions, L2-normalized, trained with supervised contrastive loss on the binary label. Model reviews are pulled together, and human reviews are pulled together, across generators.
- **Generator adversary.** Linear map over source classes (the three training generators plus human). A gradient reversal layer passes the vector through unchanged and negates the backward gradient. Minimizing this head's loss pushes the encoder to drop generator-specific style. The reversal weight ramps from 0 across training (DANN schedule), because a full adversarial term at step 0 is unstable.

Total loss: `L = L_ce + lambda_c * L_supcon + lambda_a * L_adv`.

| Variant | Active terms |
| --- | --- |
| `base` | Cross-entropy |
| `contrastive` | Cross-entropy and supervised contrastive |
| `adversarial` | Cross-entropy and generator adversary |
| `both` | All three |

Implementation: `src/deberta_multitask.py`, `src/multitask_trainer.py`, `src/variant_data.py`. Command-line training: `scripts/train_deberta_variants.py`.

A classification-only fine-tune (cross-entropy, one rotation at a time) is in `notebooks/03_train_deberta_baseline.ipynb` and `scripts/train_deberta.py`. `notebooks/04_eval_deberta_summary.ipynb` loads those checkpoints and prints the summary table.

## Evaluation

For each checkpoint, `test_indist` and `test_crossgen` are scored with F1 (positive class: model-written) and ROC-AUC. Metrics are averaged across the four rotations. A per-generator breakdown reports cross-generator F1 for each held-out model. The comparison that matters is the gap between in-distribution and cross-generator scores. Contrastive and adversarial terms are there to shrink that gap.

`src/eval_summary.py` prints those tables. The same column layout is used for a TF-IDF plus logistic regression baseline title.

Supervised contrastive loss needs at least two examples of a class in the batch. Training batches mix human and model reviews. Adversarial source labels are built per rotation, because the held-out generator is not in the training file. DeBERTa-v3's tokenizer needs `sentencepiece`.

## Run

From the clone root, with PyTorch, Transformers, scikit-learn, pandas, and NumPy installed:

```bash
python scripts/train_deberta_variants.py --variant contrastive --rotation 0
python scripts/train_deberta_variants.py --variant all --all --eval
```

`--skip-existing` leaves a finished `(variant, rotation)` checkpoint in place. The `base` variant is skipped when `checkpoints/deberta_ce/rotation_{r}/best_hf` is already present.

Set `ECS111_PROJECT_DIR` to the clone root when the process is not started there.

Notebooks `00_setup.ipynb` through `02_build_splits.ipynb` were written for Google Colab. `00_setup.ipynb` mounts Drive and syncs this repository to `MyDrive/ECS111FinalProject`, so code and CSVs share one folder. Local Jupyter uses `ECS111_PROJECT_DIR` the same way.

## Layout

```
notebooks/    collection, generation, LOGO splits, CE baseline, summary table
src/          multitask model, trainer, split helpers, evaluation tables
scripts/      command-line training and scoring
data/         human reviews, generated reviews, four LOGO rotations
```

## License

No license file is included with this repository.
