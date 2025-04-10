import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    SPLITTER_MODEL_PATH,
    TESTING_DATA_CSV,
    TRAINING_DATA_CSV,
    max_length,
    learning_rate,
    weight_decay,
    patience,
    factor,
    epochs,
    log_steps,
    eval_steps,
    training_mini_batch_size,
    testing_mini_batch_size,
)

from training.dataset import get_dataset


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()


class BigBirdBinaryClassifier(nn.Module):
    def __init__(self, model_name="google/bigbird-roberta-base"):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, problem_type="single_label_classification"
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        if labels is not None:
            return logits, probs
        return probs


def find_best_threshold(pred_probs, true_labels, target_metric="f1"):
    best_thresh = 0.5
    best_score = 0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (pred_probs >= t).astype(int)
        score = (
            f1_score(true_labels, preds, zero_division=0.0)
            if target_metric == "f1"
            else precision_score(true_labels, preds, zero_division=0.0)
        )
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")

    training_dataset, N0, N1 = get_dataset(TRAINING_DATA_CSV, training_mini_batch_size)
    testing_dataset, _, _ = get_dataset(TESTING_DATA_CSV, testing_mini_batch_size)
    # training_dataset = training_dataset[:2000]
    # testing_dataset = testing_dataset[:1000]

    model = BigBirdBinaryClassifier().to("cuda")
    loss_fn = FocalLoss().to("cuda")
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=factor
    )

    steps, best_f1_score = 0, 0
    accumulation_steps = 4

    for epoch in range(epochs):
        random.shuffle(training_dataset)
        epoch_steps = 0
        for mini_batch in training_dataset:
            epoch_steps += 1
            steps += 1
            texts = [
                str(feature) for feature in mini_batch["features"]
            ]  # Features already contain special tags
            encoding = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to("cuda")
            attention_mask = encoding["attention_mask"].to("cuda")
            labels = torch.tensor(mini_batch["labels"], dtype=torch.float32).to("cuda")

            with torch.amp.autocast_mode.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                logits, probs = model(input_ids, attention_mask, labels)
                loss = loss_fn(logits, labels) / accumulation_steps

            scaler.scale(loss).backward()
            if steps % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if steps % log_steps == 0:
                print(
                    f"step: {steps} - epoch: {epoch + (epoch_steps / len(training_dataset)):.2f} - lr: {optimizer.param_groups[0]['lr']:.6f} - loss: {loss.item():.4f}"
                )

            if steps % eval_steps == 0:
                model.eval()
                all_preds, all_labels, eval_losses = [], [], []

                with torch.no_grad():
                    for test_batch in testing_dataset:
                        texts = [str(feature) for feature in test_batch["features"]]
                        encoding = tokenizer(
                            texts,
                            truncation=True,
                            padding="max_length",
                            max_length=max_length,
                            return_tensors="pt",
                        )
                        input_ids = encoding["input_ids"].to("cuda")
                        attention_mask = encoding["attention_mask"].to("cuda")
                        labels = torch.tensor(
                            test_batch["labels"], dtype=torch.float32
                        ).to("cuda")

                        with torch.amp.autocast_mode.autocast(
                            device_type="cuda", dtype=torch.float16
                        ):
                            logits, probs = model(input_ids, attention_mask, labels)
                            loss = loss_fn(logits, labels)

                        eval_losses.append(loss.item())
                        all_preds.extend(probs.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                model.train()
                best_thresh, _ = find_best_threshold(
                    np.array(all_preds), np.array(all_labels), target_metric="precision"
                )
                final_preds = (np.array(all_preds) >= best_thresh).astype(int)
                mean_eval_loss = np.mean(eval_losses)
                acc = accuracy_score(all_labels, final_preds)
                prec = precision_score(all_labels, final_preds, zero_division=0.0)
                rec = recall_score(all_labels, final_preds, zero_division=0.0)
                f1 = f1_score(all_labels, final_preds, zero_division=0.0)

                print(
                    f"step: {steps} - eval loss: {mean_eval_loss:.4f} "
                    f"| acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f} | best_thresh: {best_thresh:.2f}"
                )

                scheduler.step(mean_eval_loss)
                if f1 > best_f1_score:
                    torch.save(model.state_dict(), SPLITTER_MODEL_PATH)
                    best_f1_score = f1
