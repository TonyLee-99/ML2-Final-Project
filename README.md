# Abstractive News Summarization with Transformer Models

This repository contains our ML2 final project on abstractive news summarization using CNN/DailyMail 3.0.0. The project starts with a fine-tuned BART-base summarizer, then studies a practical limitation of transformer summarization: many news articles are longer than the model's input window.

The repository is intentionally centered on the executed notebooks and this README. The presentation story has been folded into the README so the project can be reviewed directly from the GitHub repository.

## Project Goal

The task is to generate concise abstractive summaries for long-form English news articles.

| Component | Description |
| --- | --- |
| Input | CNN/DailyMail news article |
| Model | Sequence-to-sequence Transformer summarizer |
| Target | Human-written `highlights` used as reference summaries |
| Evaluation structure | Official train / validation / test splits from CNN/DailyMail |

The central research question is:

> How well does a fine-tuned BART summarization model perform on CNN/DailyMail, and how much do context limits and long-article truncation affect summary quality?

## Repository Files

| File | Purpose |
| --- | --- |
| `01_main_bart_cnn_dailymail_experiment.ipynb` | Main experiment: dataset loading, EDA, BART-base fine-tuning, Lead-3 baseline, ROUGE/BERTScore evaluation, truncation analysis, and article-length breakdown. |
| `02_hierarchical_long_context_experiments.ipynb` | Long-article extension: compares truncated BART, hierarchical BART, LED, Lead-3, and BART-large-CNN on articles longer than 1024 BART tokens. |
| `03_factual_consistency_and_rubric.ipynb` | Structured qualitative evaluation: builds a 30-article, 5-method review sample and scores outputs for fluency, factual consistency, coverage, and conciseness. |
| `04_recent_news_demo.ipynb` | Demo notebook: applies the fine-tuned model to a recent real-world news article outside CNN/DailyMail. |
| `requirements.txt` | Main Python dependencies used by the notebooks. |

Large checkpoints, generated CSV/XLSX result files, progress snapshots, and slide files are not required for review. They are generated artifacts, while the executed notebooks and this README contain the complete project story.

## Why BART?

BART, or Bidirectional and Auto-Regressive Transformers, keeps the encoder-decoder Transformer structure that is natural for summarization:

- the encoder reads the source article,
- the decoder generates the summary,
- pretraining as a denoising autoencoder makes BART strong for text generation tasks.

We use `facebook/bart-base` as the main model because it is large enough to be a meaningful abstractive summarizer but still feasible to fine-tune in Colab.

## Systems Compared

| System | Type | Role in the project |
| --- | --- | --- |
| Lead-3 | Extractive baseline | Uses the first three article sentences. It is simple but strong for news because important facts often appear early. |
| Fine-tuned BART-base | Main model, about 140M parameters | Our trained abstractive model on CNN/DailyMail subsets. |
| BART-large-CNN | Reference only, about 406M parameters | A stronger model already fine-tuned on CNN/DailyMail. It is not counted as our own training result. |
| Hierarchical BART | Long-article extension | Splits long articles into chunks, summarizes chunks, then summarizes the chunk summaries. |
| Fine-tuned LED | Long-context extension | Uses a long-context encoder-decoder model that can directly process longer inputs than standard BART. |

Lead-3 matters because CNN/DailyMail articles often follow an inverted-pyramid structure: the main event and key facts appear near the beginning, followed by supporting details, quotes, examples, and background context.

## Training Setup

| Choice | Setting used |
| --- | --- |
| Main model | `facebook/bart-base` |
| Dataset | CNN/DailyMail 3.0.0 |
| Training sizes explored | 20k and 50k CNN/DailyMail training examples |
| Main reported training run | 50k examples |
| Validation subset | 1,500 examples |
| Test subset | 1,500 examples |
| Sequence lengths | 1024 input tokens, 128 target tokens |
| Hardware | Colab A100 GPU with BF16 mixed precision |
| Optimization | 2 epochs, learning rate `3e-5`, dynamic padding |
| Automatic metrics | ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum, BERTScore |

Dynamic padding avoids wasting memory on shorter articles, while 1024 input tokens uses the maximum standard BART context window.

Dataset sizes after cleaning:

| Split | Examples |
| --- | ---: |
| Train | 287,111 |
| Validation | 13,368 |
| Test | 11,490 |

## Main Evaluation Snapshot

Fine-tuned BART-base improves over Lead-3. More training data helps, but the gain from 20k to 50k examples is modest.

| System | ROUGE-Lsum | BERTScore F1 |
| --- | ---: | ---: |
| Lead-3 baseline | 36.46 | 24.39 |
| BART-base 20k | 38.07 | 31.67 |
| BART-base 50k | 38.36 | 32.10 |
| BART-large-CNN reference | 40.04 | N/A |

Full 50k-run metric table from Notebook 01:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lead-3 extractive baseline | 40.1828 | 17.5035 | 24.9322 | 36.4560 | 24.3850 |
| Fine-tuned `facebook/bart-base` | 41.3973 | 18.7031 | 28.0295 | 38.3034 | 32.1985 |
| `facebook/bart-large-cnn` reported reference | 42.9490 | 20.8150 | 30.6190 | 40.0380 | N/A |

Key interpretation:

- Fine-tuning works: BART-base beats the Lead-3 baseline on both ROUGE and BERTScore.
- More data helps modestly: the 50k run improves over the 20k run, but not dramatically.
- The BART-large-CNN reference remains stronger, which is expected because it is larger and already fine-tuned on CNN/DailyMail.

## Truncation Problem

The next question is whether context limits hurt summarization quality. Standard BART can only process up to 1024 input tokens, but many CNN/DailyMail articles are longer.

| Input limit | Cleaned training articles exceeding limit |
| ---: | ---: |
| 512 BART tokens | 79.54% |
| 1024 BART tokens | 30.04% |

Length distribution by split:

| Split | `<=512` | `513-1024` | `>1024` |
| --- | ---: | ---: | ---: |
| Train | 20.46% | 49.51% | 30.04% |
| Validation | 24.24% | 46.99% | 28.77% |
| Test | 23.74% | 47.04% | 29.22% |

This shows that 1024 tokens is much better than 512, but it still truncates about 30% of examples.

## Article Length vs. Summary Quality

Summary quality decreases as article length increases. This suggests long inputs are harder, although length alone does not prove truncation is the only cause.

| Model | Length group | ROUGE-Lsum |
| --- | --- | ---: |
| Fine-tuned BART-base | `<=512` tokens | 41.13 |
| Lead-3 baseline | `<=512` tokens | 39.72 |
| Fine-tuned BART-base | `513-1024` tokens | 38.37 |
| Lead-3 baseline | `513-1024` tokens | 36.57 |
| Fine-tuned BART-base | `>1024` tokens | 35.73 |
| Lead-3 baseline | `>1024` tokens | 33.43 |

Both models score lower as articles move from short to long groups. Longer articles may be more complex, so the project adds further reference-coverage and long-context experiments.

## News Lead Bias

CNN/DailyMail articles often put the main facts near the opening. This means the first 1024 tokens may already contain much of the information needed for a reference-style summary.

This matters because truncation is not equally harmful in every article. If the truncated tail mostly contains background or examples, losing it may not hurt ROUGE as much as expected. If the tail contains key facts, truncation is more damaging.

## Reference Coverage Analysis

Notebook 02 checks whether reference-summary content appears in the first 1024 tokens or only in the tail after the cutoff.

| Coverage measure | Mean | Median |
| --- | ---: | ---: |
| Prefix term coverage | 79.63% | 81.74% |
| Tail-only terms | 4.06% | 0.00% |
| Prefix entity coverage | 87.12% | 91.49% |
| Tail-only entities | 2.27% | 0.00% |

Interpretation: most reference-summary content is already covered by the first 1024 tokens. The tail adds some unique information, but tail-only reference terms and entities are relatively low.

## Long-Context Experiment Design

Notebook 02 evaluates all 440 test examples longer than 1024 BART tokens.

| Method | Design |
| --- | --- |
| Truncated BART | First 1024 tokens -> summary |
| Hierarchical BART | Article chunks -> chunk summaries -> final summary |
| Fine-tuned LED | Long-context encoder-decoder, up to 4096 tokens in this experiment |
| BART-large-CNN | Strong CNN/DailyMail reference model |

The goal is to test whether giving the model access to longer context improves performance over the simple 1024-token BART setup.

## Long-Article Results

Longer context did not automatically improve performance.

| Method | ROUGE-Lsum | BERTScore F1 | Examples |
| --- | ---: | ---: | ---: |
| BART-large-CNN reference | 38.65 | 26.21 | 440 |
| Fine-tuned BART-base, 1024-token truncation | 35.73 | 27.44 | 440 |
| Hierarchical fine-tuned BART | 33.58 | 24.85 | 440 |
| Lead-3 baseline | 33.43 | 19.33 | 440 |
| Fine-tuned LED-base-16384 CNN/DM | 30.16 | 16.44 | 440 |

Main finding:

- BART-large-CNN is strongest on ROUGE-Lsum.
- Our fine-tuned BART-base with 1024-token truncation remains competitive.
- Hierarchical BART may lose details during chunk-level compression.
- LED has longer input access, but longer context alone is not enough; the model can still suffer from style mismatch or unstable generation.

## Qualitative Evaluation

Notebook 03 adds a structured qualitative evaluation because automatic metrics do not fully capture factual consistency, omissions, or usefulness.

Evaluation design:

| Item | Value |
| --- | ---: |
| Articles sampled | 30 |
| Methods per article | 5 |
| Total model outputs reviewed | 150 |
| Rubric scale | 1 to 5 |

Rubric dimensions:

- fluency,
- factual consistency,
- coverage,
- conciseness,
- primary error type.

Method-level qualitative scores:

| Method | Fluency | Factual | Coverage | Concise | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| BART-large-CNN | 4.87 | 4.97 | 3.57 | 3.93 | 4.33 |
| Lead-3 | 4.80 | 5.00 | 2.50 | 3.40 | 3.93 |
| Fine-tuned BART | 4.37 | 4.77 | 2.80 | 3.37 | 3.83 |
| Hierarchical BART | 4.27 | 4.63 | 2.50 | 3.60 | 3.75 |
| LED | 3.20 | 3.80 | 2.27 | 3.03 | 3.08 |

Method-level error pattern:

- Lead-3 is very factually safe because it copies the opening, but it often misses reference-relevant details later in the article.
- Fine-tuned BART improves coverage over Lead-3, but still loses points for omissions and occasional unsupported or verbose details.
- Hierarchical BART accesses more text, yet chunk-level compression does not translate into better coverage or overall quality.
- LED has longer input access, but lower factual and fluency scores suggest style mismatch and less stable generation in this setup.

## Real-News Demo

Notebook 04 demonstrates the trained model on a recent article outside CNN/DailyMail. Since the article has no human reference summary, we do not compute ROUGE or BERTScore. The output is judged qualitatively for readability, coverage, and factual consistency.

Example generated summary from the demo:

> SpaceX's regulatory filing revealed a financially smart link between its launch-services division and profitable Starlink satellite internet operation. Its AI business looks shakier than its rockets. Revenue in the AI division has mostly come from X, which is not a pure-play AI venture.

## Final Takeaways

1. Fine-tuning BART-base improves over a simple but strong Lead-3 baseline.
2. Increasing training data from 20k to 50k examples helps, but the improvement is modest.
3. Context length matters: about 30% of cleaned training articles exceed 1024 BART tokens.
4. News lead bias reduces the damage of truncation because many key facts appear early.
5. Longer-context methods are not automatically better; hierarchical compression and LED generation both introduced weaknesses in this experiment.
6. Qualitative evaluation is necessary because ROUGE and BERTScore do not fully capture factuality, omissions, or summary usefulness.

## Reproducibility Notes

The notebooks were designed primarily for Google Colab with GPU acceleration and Google Drive storage. A full rerun can be expensive because it includes model fine-tuning, long-article generation, BERTScore evaluation, and qualitative scoring.

For review, re-running is not required. The notebooks already include executed outputs and show the full workflow.

If reproducing from scratch:

1. run Notebook 01 to create the fine-tuned checkpoint and prediction files,
2. run Notebook 02 to generate long-article comparison outputs,
3. run Notebook 03 to build and score the qualitative review sample,
4. run Notebook 04 only for the optional recent-news demo.

## Setup

Install the main Python dependencies:

```bash
pip install -r requirements.txt
```

Some cells assume Colab paths such as `/content/drive/MyDrive/...`. If running locally, update the storage paths and reduce batch sizes if GPU memory is limited.
