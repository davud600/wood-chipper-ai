from transformers import AutoTokenizer
import numpy as np
import torch.nn as nn
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from model import PageClassifier

from utils import (
    TESTING_DATA_CSV,
    TRAINING_DATA_CSV,
    get_dataset,
    max_length,
    learning_rate,
    weight_decay,
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

    page_classifier = PageClassifier().to("cuda")
    scaler = torch.amp.grad_scaler.GradScaler()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        page_classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    steps = 0

    for epoch in range(epochs):
        loss = None
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
            labels = torch.stack(
                [
                    torch.tensor(1 if label == 1 else 0, dtype=torch.long)
                    for label in mini_batch["labels"]
                ]
            ).to("cuda")

            with torch.amp.autocast_mode.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                predicted_pages = page_classifier(features)
                loss = loss_fn(predicted_pages, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # loss = loss_fn(predicted_pages, labels)
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            if steps % log_steps == 0:
                print(
                    f"step: {steps} - epoch: {epoch + (epoch_steps / len(training_dataset))} - lr: {optimizer.param_groups[0]["lr"]} - loss: {loss.item() if loss is not None else "None"}"
                )

            if steps % eval_steps == 0:
                eval_loss = np.array([])
                for test_mini_batch in testing_dataset:
                    tokenized = tokenizer(
                        test_mini_batch["features"],
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                    )

                    test_features = tokenized.input_ids.to("cuda")
                    test_labels = torch.stack(
                        [
                            torch.tensor(1 if label == 1 else 0, dtype=torch.long)
                            for label in test_mini_batch["labels"]
                        ]
                    ).to("cuda")

                    with torch.amp.autocast_mode.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        predicted_pages = page_classifier(test_features)
                        eval_loss = np.append(
                            eval_loss,
                            [
                                loss_fn(predicted_pages, test_labels)
                                .to("cpu")
                                .detach()
                                .numpy()
                            ],
                        )

                print(
                    f"step: {steps} - eval loss: {eval_loss.mean() if eval_loss is not None else "None"}"
                )

    torch.save(obj=page_classifier.state_dict(), f="./model/page_classifier.pth")
