from transformers import AutoTokenizer
import numpy as np
import random
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from model import ClassifierModel

from utils import (
    TESTING_DATA_CSV,
    TRAINING_DATA_CSV,
    get_dataset,
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

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096", device="cuda"
    )

    training_dataset = get_dataset(
        path=TRAINING_DATA_CSV, mini_batch_size=training_mini_batch_size
    )
    testing_dataset = get_dataset(
        path=TESTING_DATA_CSV, mini_batch_size=testing_mini_batch_size
    )

    model = ClassifierModel().to("cuda")
    scaler = torch.amp.grad_scaler.GradScaler()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=factor
    )
    steps = 0
    smallest_mean_eval_loss = 100

    for epoch in range(epochs):
        random.shuffle(training_dataset)
        epoch_steps = 0
        for mini_batch in training_dataset:
            epoch_steps += 1
            steps += 1
            tokenized = tokenizer(
                mini_batch["features"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

            features = tokenized.input_ids.to("cuda")
            page_labels = torch.stack(
                [
                    torch.tensor(1 if label == 1 else 0, dtype=torch.long)
                    for label in mini_batch["page_labels"]
                ]
            ).to("cuda")
            type_labels = torch.stack(
                [
                    torch.tensor(label, dtype=torch.long)
                    for label in mini_batch["type_labels"]
                ]
            ).to("cuda")

            with torch.amp.autocast_mode.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                loss, predicted_pages, predicted_types = model(
                    features, page_labels, type_labels
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if steps % log_steps == 0:
                print(
                    f"step: {steps} - epoch: {epoch + (epoch_steps / len(training_dataset))} - lr: {optimizer.param_groups[0]["lr"]} - loss: {loss.item() if loss is not None else "None"}"
                )

            if steps % eval_steps == 0:
                eval_losses = []
                for test_mini_batch in testing_dataset:
                    tokenized = tokenizer(
                        test_mini_batch["features"],
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                    )

                    test_features = tokenized.input_ids.to("cuda")
                    test_page_labels = torch.stack(
                        [
                            torch.tensor(1 if label == 1 else 0, dtype=torch.long)
                            for label in test_mini_batch["page_labels"]
                        ]
                    ).to("cuda")
                    test_type_labels = torch.stack(
                        [
                            torch.tensor(label, dtype=torch.long)
                            for label in test_mini_batch["type_labels"]
                        ]
                    ).to("cuda")

                    with torch.amp.autocast_mode.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        model_eval_loss, predicted_pages, predicted_types = model(
                            test_features, test_page_labels, test_type_labels
                        )
                        eval_losses.append(model_eval_loss.to("cpu").detach().numpy())

                mean_eval_loss = np.mean(eval_losses)
                print(f"step: {steps} - eval loss: {mean_eval_loss}")
                scheduler.step(mean_eval_loss)

                if mean_eval_loss < smallest_mean_eval_loss:
                    torch.save(obj=model.state_dict(), f="../model/model.pth")
                    smallest_mean_eval_loss = mean_eval_loss
