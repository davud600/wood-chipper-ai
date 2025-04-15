import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import (
    device,
    training_mini_batch_size,
    testing_mini_batch_size,
    learning_rate,
    weight_decay,
    patience,
    factor,
    epochs,
    log_steps,
    eval_steps,
    cnn_warmup_steps,
    llm_warmup_steps,
)
from .utils import count_classes, eval_and_save
from .model import FusionModel
from .dataset.dataset import DocumentDataset
from config.settings import (
    TRAINING_DATA_CSV,
    TESTING_DATA_CSV,
    SPLITTER_MODEL_PATH,
    image_output_size,
)


def train_loop(
    train_dataset, test_dataset, N0, N1, model, cnn_warmup_steps, llm_warmup_steps
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_mini_batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=testing_mini_batch_size)

    # debug - start
    batch = next(iter(train_loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"[check] batch[{k}] on: {v.device}")
    # debug - end

    pos_weight = torch.tensor([(N0 / N1) ** 0.5], dtype=torch.float16).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    llm_params = model.reader_model.parameters()
    cnn_params = model.cnn_model.parameters()
    fusion_params = model.fusion_mlp.parameters()

    opt_llm = torch.optim.AdamW(llm_params, lr=learning_rate, weight_decay=weight_decay)
    opt_cnn = torch.optim.AdamW(cnn_params, lr=learning_rate, weight_decay=weight_decay)
    opt_fusion = torch.optim.AdamW(
        fusion_params, lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, factor=factor
    )
    scaler = torch.amp.grad_scaler.GradScaler()

    print("[INFO] Starting training...")
    step = 0
    best_f1 = 0

    for epoch in range(epochs):
        model.train()

        for batch in train_loader:
            step += 1

            with torch.amp.autocast_mode.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                if step <= cnn_warmup_steps:
                    opt_cnn.zero_grad()
                    _, loss = model.cnn_model.forward(batch, loss_fn)
                    scaler.scale(loss).backward()
                    scaler.step(opt_cnn)
                    scaler.update()

                    if step % log_steps == 0:
                        print(
                            f"[CNN Warmup] [STEP {step}] Epoch {epoch+1} | Loss: {loss.item():.4f}"
                        )

                elif step <= llm_warmup_steps + cnn_warmup_steps:
                    opt_llm.zero_grad()
                    _, loss = model.reader_model.forward(batch, loss_fn)
                    scaler.scale(loss).backward()
                    scaler.step(opt_llm)
                    scaler.update()

                    if step % log_steps == 0:
                        print(
                            f"[LLM Warmup] [STEP {step}] Epoch {epoch+1} | Loss: {loss.item():.4f}"
                        )

                else:
                    opt_fusion.zero_grad()
                    _, loss = model.forward(batch, loss_fn)
                    scaler.scale(loss).backward()
                    scaler.step(opt_fusion)
                    scaler.update()

                    if step % log_steps == 0:
                        print(
                            f"[STEP {step}] Epoch {epoch+1} | Loss: {loss.item():.4f}"
                        )

                if step % eval_steps == 0:
                    eval_and_save(model, scheduler, step, loss_fn, test_loader, best_f1)


if __name__ == "__main__":
    """
    Main training loop for the FusionModel.

    Initializes tokenizer, datasets, model, optimizer, scheduler, and AMP scaler.
    Trains the model with mixed precision and evaluates it periodically.
    Saves the best model based on F1 score.
    """

    print("[INFO] Initializing...")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model = FusionModel(image_size=image_output_size).to(device)
    # model.load_state_dict(
    #     torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
    # )
    print(f"[check] model on: {next(model.parameters()).device}")

    print("[training]")
    train_dataset = DocumentDataset(
        TRAINING_DATA_CSV,
        tokenizer,
        mode="train",
        image_size=image_output_size,
    )
    n1, n0 = count_classes(train_dataset)

    print("[testing]")
    test_dataset = DocumentDataset(
        TESTING_DATA_CSV,
        tokenizer,
        mode="test",
        image_size=image_output_size,
    )
    count_classes(test_dataset)

    train_loop(
        train_dataset, test_dataset, n0, n1, model, cnn_warmup_steps, llm_warmup_steps
    )
