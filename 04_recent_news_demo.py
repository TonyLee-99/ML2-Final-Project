"""
Recent-news summarization demo for the ML2 final project.

Use this in Colab after uploading/copying it to the same Google Drive project
folder. Paste a news article into ARTICLE_TEXT, run the script, and it will
generate a summary with the fine-tuned BART-base checkpoint from the main
notebook.
"""

from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# In Colab, mount Drive before running:
# from google.colab import drive
# drive.mount("/content/drive")

RUN_DIR = Path(
    "/content/drive/MyDrive/ml2_final_bart/"
    "bart_base_cnn_dm_train50000_src1024_epochs2"
)
CHECKPOINT_ROOT = RUN_DIR / "checkpoints"

# Paste a real news article body here. Do not paste only a headline.
ARTICLE_TITLE = "Paste article title here"
ARTICLE_SOURCE = "Paste source here, e.g. AP News / Reuters / BBC / NPR"
ARTICLE_TEXT = """
Paste the full article text here.
"""


MAX_SOURCE_LENGTH = 1024
MAX_SUMMARY_LENGTH = 128
MIN_SUMMARY_LENGTH = 30
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3


def find_latest_checkpoint(checkpoint_root: Path) -> Path:
    checkpoint_dirs = [
        path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir()
    ]
    if not checkpoint_dirs:
        return checkpoint_root
    return sorted(
        checkpoint_dirs,
        key=lambda path: int(path.name.split("-")[-1]),
    )[-1]


def summarize_article(article: str) -> str:
    model_dir = find_latest_checkpoint(CHECKPOINT_ROOT)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    inputs = tokenizer(
        article,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            num_beams=NUM_BEAMS,
            max_length=MAX_SUMMARY_LENGTH,
            min_length=MIN_SUMMARY_LENGTH,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            do_sample=False,
        )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    article = ARTICLE_TEXT.strip()
    if not article or article == "Paste the full article text here.":
        raise ValueError("Please paste a real article into ARTICLE_TEXT first.")

    summary = summarize_article(article)

    print("Article title:", ARTICLE_TITLE)
    print("Source:", ARTICLE_SOURCE)
    print()
    print("Model summary:")
    print(summary)
