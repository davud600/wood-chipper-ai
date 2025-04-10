from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random
import torch
import os

from transformers import AutoTokenizer
from utils import parse_formatted_content_batch_to_sections
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

from .model_2 import SplitterModel
from training.dataset import get_dataset


def find_best_threshold(pred_probs, true_labels, target_metric="f1"):
    best_thresh = 0.5
    best_score = 0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (pred_probs >= t).astype(int)
        if target_metric == "f1":
            score = f1_score(true_labels, preds)
        elif target_metric == "precision":
            score = precision_score(true_labels, preds)
        else:
            raise ValueError("Unsupported metric")
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score


def encode_batch(batch, tokenizer):
    input_ids = []
    attention_masks = []
    for feature in batch["features"]:
        prev, curr, next = parse_formatted_content_batch_to_sections(str(feature))

        # Concatenate with separators
        full_input = prev + tokenizer.sep_token + curr + tokenizer.sep_token + next

        enc = tokenizer(
            full_input,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids.append(enc["input_ids"].squeeze(0))
        attention_masks.append(enc["attention_mask"].squeeze(0))

    input_ids = torch.stack(input_ids).to("cuda")
    attention_masks = torch.stack(attention_masks).to("cuda")
    labels = torch.tensor(batch["labels"], dtype=torch.float32).to("cuda")

    return input_ids, attention_masks, labels


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    training_dataset, N0, N1 = get_dataset(TRAINING_DATA_CSV, training_mini_batch_size)
    testing_dataset, _, _ = get_dataset(TESTING_DATA_CSV, testing_mini_batch_size)

    model = SplitterModel(pos_weight=N0 / N1).to("cuda")
    scaler = torch.amp.grad_scaler.GradScaler()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=factor
    )

    steps = 0
    best_f1_score = 0

    for epoch in range(epochs):
        random.shuffle(training_dataset)
        epoch_steps = 0

        for mini_batch in training_dataset:
            epoch_steps += 1
            steps += 1

            input_ids, attention_mask, labels = encode_batch(mini_batch, tokenizer)

            with torch.amp.autocast_mode.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                loss, predicted = model(input_ids, attention_mask, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if steps % log_steps == 0:
                print(
                    f"step: {steps} - epoch: {epoch + (epoch_steps / len(training_dataset)):.2f} - "
                    f"lr: {optimizer.param_groups[0]['lr']} - loss: {loss.item():.4f}"
                )

            if steps % eval_steps == 0:
                eval_losses = []
                all_preds = []
                all_labels = []

                for test_batch in testing_dataset:
                    input_ids, attention_mask, labels = encode_batch(
                        test_batch, tokenizer
                    )

                    with torch.amp.autocast_mode.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        eval_loss, predicted = model(input_ids, attention_mask, labels)
                        eval_losses.append(eval_loss.detach().cpu().numpy())

                    all_preds.extend(predicted.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

                best_thresh, best_f1 = find_best_threshold(
                    np.array(all_preds), np.array(all_labels), target_metric="precision"
                )
                final_preds = (np.array(all_preds) >= best_thresh).astype(int)

                mean_eval_loss = np.mean(eval_losses)
                acc = accuracy_score(all_labels, final_preds)
                prec = precision_score(all_labels, final_preds)
                rec = recall_score(all_labels, final_preds)
                f1 = f1_score(all_labels, final_preds)

                print(
                    f"step: {steps} - eval loss: {mean_eval_loss:.4f} "
                    f"| acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f} "
                    f"| best_thresh: {best_thresh:.2f}"
                )

                scheduler.step(mean_eval_loss)

                if f1 > best_f1_score:
                    torch.save(model.state_dict(), SPLITTER_MODEL_PATH)
                    best_f1_score = f1
