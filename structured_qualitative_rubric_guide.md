# Structured Qualitative Rubric

Use this rubric for the fixed 30-example stratified sample in `qualitative_factuality_rubric_stratified_30.csv`.

Score each model summary from 1 to 5 on four dimensions:

| Dimension | 5 | 3 | 1 |
|---|---|---|---|
| Fluency | Natural, grammatical, easy to read | Understandable but awkward | Hard to read or incoherent |
| Factual consistency | Fully supported by the article | Some unsupported/questionable detail | Major contradiction or hallucination |
| Coverage | Captures all/most main points | Captures some but misses important points | Misses the core story |
| Conciseness | Focused and compact | Some unnecessary detail | Verbose, repetitive, or unfocused |

Recommended sampling design:

- 10 examples from `<=512 tokens`
- 10 examples from `513-1024 tokens`
- 10 examples from `>1024 tokens`

Recommended reporting table:

| Model | Fluency | Factual Consistency | Coverage | Conciseness |
|---|---:|---:|---:|---:|
| Lead-3 | mean score | mean score | mean score | mean score |
| Fine-tuned BART-base | mean score | mean score | mean score | mean score |

Interpretation points:

- Lead-3 is usually factually safer because it copies source sentences, but it can be verbose and may miss later article information.
- BART is usually more concise and abstractive, but factual consistency must be checked because generation can introduce unsupported claims.
- Compare scores by length group if you have time; long articles are the most relevant group for truncation errors.
