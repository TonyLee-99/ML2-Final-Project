# ML2 Final Project: CNN/DailyMail Summarization

This repository contains a four-notebook final project on abstractive news summarization using CNN/DailyMail 3.0.0. The project fine-tunes `facebook/bart-base`, compares it with strong baselines, studies the effect of input truncation, and adds long-article plus qualitative factual consistency analysis.

The repository is intentionally kept focused. It includes the executed notebooks, a dependency file, and project documentation. Large checkpoints, generated CSV/XLSX outputs, intermediate progress files, and presentation drafts are excluded because they are outputs of the notebooks rather than source material needed for review.

## Research Question

How well does a fine-tuned BART summarization model perform on CNN/DailyMail, and what happens when source articles exceed the model's 1024-token input limit?

The final project answers this through four connected parts:

1. fine-tune and evaluate BART-base on CNN/DailyMail,
2. measure how often article truncation occurs,
3. test long-article alternatives such as hierarchical summarization and LED,
4. evaluate summary quality with both automatic metrics and a structured qualitative rubric.

## Files

| File | Role in the project |
| --- | --- |
| `01_main_bart_cnn_dailymail_experiment.ipynb` | Main experiment: data loading, EDA, truncation analysis, BART-base fine-tuning, Lead-3 baseline, ROUGE/BERTScore evaluation, and length-group analysis. |
| `02_hierarchical_long_context_experiments.ipynb` | Long-article extension: evaluates all test examples longer than 1024 BART tokens and compares Lead-3, 1024-token BART, hierarchical BART, LED, and BART-large-CNN. |
| `03_factual_consistency_and_rubric.ipynb` | Qualitative evaluation: builds a fixed 30-article, 5-method review sample and applies a rubric for fluency, factual consistency, coverage, and conciseness. |
| `04_recent_news_demo.ipynb` | Presentation demo: applies the fine-tuned model to a recent news article outside CNN/DailyMail. |
| `requirements.txt` | Main Python packages used across the notebooks. |

## Notebook 01: Main BART Fine-Tuning Experiment

Notebook 01 is the core model experiment.

It performs:

- CNN/DailyMail 3.0.0 loading and cleaning,
- exploratory data analysis for article and summary length,
- BART-token length analysis at 512 and 1024 tokens,
- Lead-3 extractive baseline creation,
- fine-tuning `facebook/bart-base`,
- ROUGE and BERTScore evaluation on a held-out test subset,
- article-length breakdown for short, medium, and long articles.

Main configuration:

| Setting | Value |
| --- | --- |
| Dataset | `cnn_dailymail`, version `3.0.0` |
| Main model | `facebook/bart-base` |
| Training subset | 50,000 examples |
| Validation subset | 1,500 examples |
| Test subset | 1,500 examples |
| Max source length | 1024 BART tokens |
| Max target length | 128 tokens |
| Epochs | 2 |

Dataset sizes after cleaning:

| Split | Examples |
| --- | ---: |
| Train | 287,111 |
| Validation | 13,368 |
| Test | 11,490 |

Main automatic metrics on the 1,500-example test subset:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lead-3 extractive baseline | 40.1828 | 17.5035 | 24.9322 | 36.4560 | 24.3850 |
| Fine-tuned `facebook/bart-base` | 41.3973 | 18.7031 | 28.0295 | 38.3034 | 32.1985 |
| `facebook/bart-large-cnn` reported reference | 42.9490 | 20.8150 | 30.6190 | 40.0380 | N/A |

Takeaway: fine-tuned BART-base improves over Lead-3 on ROUGE and BERTScore, while BART-large-CNN remains a stronger external reference point.

## Truncation Findings

The project emphasizes truncation because CNN/DailyMail articles are often longer than common transformer input limits.

| Input limit | Training articles truncated | Average tokens lost when truncated | Median tokens lost |
| ---: | ---: | ---: | ---: |
| 512 tokens | 79.54% | 483.56 | 391 |
| 1024 tokens | 30.04% | 369.61 | 280 |

Article token groups in the cleaned training split:

| Length group | Examples | Share |
| --- | ---: | ---: |
| `<=512` tokens | 58,740 | 20.46% |
| `513-1024` tokens | 142,136 | 49.51% |
| `>1024` tokens | 86,235 | 30.04% |

This means a 1024-token model is much better than a 512-token model, but it still loses source context for about 30% of training articles.

## Length-Group Results

Notebook 01 also compares performance by article length.

| Model | Length group | ROUGE-1 | ROUGE-2 | ROUGE-Lsum |
| --- | --- | ---: | ---: | ---: |
| Lead-3 | `<=512 tokens` | 43.3070 | 21.3213 | 39.7190 |
| Fine-tuned BART-base | `<=512 tokens` | 43.8763 | 22.0443 | 41.1344 |
| Lead-3 | `513-1024 tokens` | 40.4707 | 17.4849 | 36.5670 |
| Fine-tuned BART-base | `513-1024 tokens` | 41.6884 | 18.6596 | 38.3721 |
| Lead-3 | `>1024 tokens` | 37.0630 | 14.1811 | 33.4335 |
| Fine-tuned BART-base | `>1024 tokens` | 38.8189 | 15.8185 | 35.7281 |

Takeaway: performance decreases as articles get longer, but fine-tuned BART-base remains stronger than Lead-3 in all length groups.

## Notebook 02: Long-Article Extension

Notebook 02 focuses only on test examples longer than 1024 BART tokens. In the 1,500-example test subset, there are 440 such long examples.

Methods compared on the same 440 long articles:

- Lead-3 baseline,
- fine-tuned BART-base with 1024-token truncation,
- hierarchical BART, where article chunks are summarized and then summarized again,
- fine-tuned LED-base-16384 CNN/DailyMail model,
- BART-large-CNN reference model.

Long-article automatic metrics:

| Method | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | BERTScore F1 | Examples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lead-3 baseline | 37.0630 | 14.1811 | 33.4335 | 19.3308 | 440 |
| Fine-tuned BART-base, 1024-token truncation | 38.8189 | 15.8185 | 35.7281 | 27.4355 | 440 |
| Hierarchical fine-tuned BART | 36.6385 | 14.0029 | 33.5842 | 24.8547 | 440 |
| Fine-tuned LED-base-16384 CNN/DM | 32.7555 | 10.9999 | 30.1644 | 16.4420 | 440 |
| BART-large-CNN reference | 41.6900 | 18.1841 | 38.6494 | 26.2091 | 440 |

Notebook 02 also adds a lexical reference-coverage analysis. On the 440 long examples, about 79.63% of reference terms appear within the first 1024 tokens on average, while about 4.06% are tail-only terms. This suggests that truncation can remove relevant evidence, but the beginning of the article still contains much of the reference information for many CNN/DailyMail examples.

Takeaway: for this dataset and setup, the fine-tuned 1024-token BART-base model remains competitive on long articles. The hierarchical and LED alternatives do not automatically improve results, while BART-large-CNN is the strongest reference model.

## Notebook 03: Factual Consistency and Qualitative Rubric

Notebook 03 adds structured qualitative evaluation because ROUGE and BERTScore do not fully measure factual correctness or usefulness.

The notebook creates a fixed review set:

| Item | Value |
| --- | ---: |
| Articles sampled | 30 |
| Methods per article | 5 |
| Total outputs reviewed | 150 |

Rubric dimensions:

- fluency,
- factual consistency,
- coverage,
- conciseness,
- primary error type.

The notebook includes both a human-review workbook workflow and an LLM-as-a-judge scoring section. The LLM judging is used as a structured qualitative aid, not as a replacement for human review.

LLM-judge summary from Notebook 03:

| Model | Fluency | Factual consistency | Coverage | Conciseness | Overall mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| BART-large-CNN reference | 4.867 | 4.967 | 3.567 | 3.933 | 4.334 |
| Lead-3 | 4.800 | 5.000 | 2.500 | 3.400 | 3.925 |
| Fine-tuned BART-base, 1024-token truncation | 4.367 | 4.767 | 2.800 | 3.367 | 3.825 |
| Hierarchical BART | 4.267 | 4.633 | 2.500 | 3.600 | 3.750 |
| Fine-tuned LED-base-16384 CNN/DM | 3.200 | 3.800 | 2.267 | 3.033 | 3.075 |

Takeaway: BART-large-CNN has the strongest qualitative average. Lead-3 is very factually safe but has weaker coverage. Fine-tuned BART-base improves coverage over Lead-3 but can still miss information or compress too aggressively.

## Notebook 04: Recent News Demo

Notebook 04 demonstrates how the fine-tuned model can be applied to a new article outside CNN/DailyMail.

This notebook is for presentation use:

- the user pastes a recent article,
- the fine-tuned BART-base checkpoint generates a 3-4 sentence summary,
- no ROUGE or BERTScore is computed because there is no human reference summary,
- the output is judged qualitatively for readability, factual consistency, and coverage.

## Main Conclusions

1. Fine-tuned BART-base performs better than the Lead-3 baseline on the main CNN/DailyMail test subset.
2. Input truncation is a real issue: 30.04% of cleaned training articles exceed 1024 BART tokens.
3. Longer articles are harder for both extractive and abstractive methods.
4. Hierarchical summarization and LED are reasonable extensions, but they did not outperform the fine-tuned 1024-token BART-base setup in this experiment.
5. Automatic metrics are useful but incomplete, so the project adds a structured qualitative evaluation for factual consistency and coverage.

## Reproducibility Notes

The notebooks were designed primarily for Google Colab with GPU acceleration and Google Drive storage. A full rerun can be expensive because it includes model fine-tuning, long-article generation, BERTScore evaluation, and qualitative scoring.

For review, re-running is not required. The notebooks already include executed outputs and show the full workflow.

If reproducing from scratch:

1. run Notebook 01 first to create the fine-tuned checkpoint and prediction files,
2. run Notebook 02 to generate long-article comparison outputs,
3. run Notebook 03 to build and score the qualitative review sample,
4. run Notebook 04 only for the optional presentation demo.

## Setup

Install the main Python dependencies:

```bash
pip install -r requirements.txt
```

Some cells assume Colab paths such as `/content/drive/MyDrive/...`. If running locally, update the storage paths and reduce batch sizes if GPU memory is limited.
