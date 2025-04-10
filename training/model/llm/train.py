import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


from config import TESTING_DATA_CSV, TRAINING_DATA_CSV, DOCUMENT_TYPES
from collections import Counter
from statistics import mean


EXCLUDED_DOCUMENT_TYPES = {
    DOCUMENT_TYPES["unknown"],
    DOCUMENT_TYPES["proprietary-lease"],
    DOCUMENT_TYPES["tenant-correspondence"],
    DOCUMENT_TYPES["transfer-of-title"],
    DOCUMENT_TYPES["purchase-application"],
    DOCUMENT_TYPES["closing-document"],
    DOCUMENT_TYPES["alteration-document"],
    DOCUMENT_TYPES["renovation-document"],
    DOCUMENT_TYPES["refinance-document"],
    DOCUMENT_TYPES["transfer-document"],
}

# === 1. Load CSVs ===
train_df = pd.read_csv(TRAINING_DATA_CSV)
test_df = pd.read_csv(TESTING_DATA_CSV)

# Fill missing content fields just in case
train_df["content"] = train_df["content"].fillna("")
test_df["content"] = test_df["content"].fillna("")


# === 2. Preprocessing: Create context window + labels ===
def create_context_window(df, max_chars_per_page=5000, n_samples=None):
    df = df.sort_values(by=["file", "page"]).reset_index(drop=True)

    df = df[~df["type"].isin(EXCLUDED_DOCUMENT_TYPES)]

    if n_samples:
        df = df.head(n_samples)

    def clip(text):
        return str(text)[:max_chars_per_page] if isinstance(text, str) else ""

    texts, labels = [], []

    for i in range(len(df)):
        file = df.iloc[i]["file"]

        prev_text = (
            clip(df.iloc[i - 1]["content"])
            if i > 0 and df.iloc[i - 1]["file"] == file
            else ""
        )
        curr_text = clip(df.iloc[i]["content"])
        next_text = (
            clip(df.iloc[i + 1]["content"])
            if i < len(df) - 1 and df.iloc[i + 1]["file"] == file
            else ""
        )

        combined_text = f"{prev_text}\n{curr_text}\n{next_text}"
        label = 1 if df.iloc[i]["page"] == 1 else 0

        texts.append(combined_text)
        labels.append(label)

    return pd.DataFrame({"text": texts, "label": labels})


def oversample_minority(df):
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    multiplier = (len(neg) // len(pos)) // 2
    pos_upsampled = pd.concat([pos] * multiplier, ignore_index=True)

    balanced_df = pd.concat([neg, pos_upsampled]).sample(frac=1).reset_index(drop=True)
    return balanced_df


train_df = create_context_window(train_df, max_chars_per_page=1000, n_samples=7500)
train_df = oversample_minority(train_df)
test_df = create_context_window(test_df, max_chars_per_page=1000, n_samples=2000)


# === 3. Tokenizer and Dataset ===
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize(batch):
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=512
    )


def log_dataset_stats(name, df):
    print(f"\n== {name.upper()} DATASET ==")
    print(f"Total samples: {len(df)}")
    label_counts = Counter(df["label"])
    print(f"First pages (label=1): {label_counts[1]}")
    print(f"Non-first pages (label=0): {label_counts[0]}")

    token_counts = [len(tokenizer.tokenize(text)) for text in df["text"]]
    print(f"Average tokens: {mean(token_counts):.2f}")


log_dataset_stats("Train", train_df)
log_dataset_stats("Test", test_df)

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# === 4. Metrics ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    print("\n== CONFUSION MATRIX ==")
    print(cm)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# === 5. Model + Training ===
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=100,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    warmup_steps=50,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
