import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .utils import count_classes, evaluate
from .model import FusionModel
from .dataset.dataset import DocumentDataset
from config.settings import (
    TRAINING_DATA_CSV,
    TESTING_DATA_CSV,
    SPLITTER_MODEL_PATH,
    image_output_size,
    learning_rate,
    weight_decay,
    training_mini_batch_size,
    testing_mini_batch_size,
    patience,
    factor,
    epochs,
    log_steps,
    eval_steps,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    Main training loop for the FusionModel.

    Initializes tokenizer, datasets, model, optimizer, scheduler, and AMP scaler.
    Trains the model with mixed precision and evaluates it periodically.
    Saves the best model based on F1 score.
    """

    print("[INFO] Initializing...")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model = FusionModel(image_size=image_output_size).to(device)

    print("[TRAINING]")
    train_dataset = DocumentDataset(
        TRAINING_DATA_CSV,
        tokenizer,
        mode="train",
        image_size=image_output_size,
    )
    N1, N0 = count_classes(train_dataset)

    print("[TESTING]")
    test_dataset = DocumentDataset(
        TESTING_DATA_CSV,
        tokenizer,
        mode="test",
        image_size=image_output_size,
    )
    count_classes(test_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=training_mini_batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=testing_mini_batch_size)

    pos_weight = torch.tensor([(N0 / N1) ** 0.5], dtype=torch.float16).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, factor=factor
    )
    scaler = torch.amp.grad_scaler.GradScaler()

    best_f1 = 0
    step = 0

    print("[INFO] Starting training...")
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            step += 1

            input_ids = batch["input_ids"].to(device)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            attention_mask = batch["attention_mask"].to(device)
            cnn_input = batch["cnn_input"].to(device)
            labels = batch["label"].to(device)
            prev_first_page_distance = batch["prev_first_page_distance"].to(device)

            # print("[LLM] tokens:", tokens[:40])
            # print("[LLM] attention mask sum:", attention_mask[0].sum().item())
            # print(
            #     "[CNN] min/max:",
            #     cnn_input.min().item(),
            #     cnn_input.max().item(),
            #     cnn_input.mean().item(),
            # )

            optimizer.zero_grad()
            with torch.amp.autocast_mode.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                # logits = model(
                #     input_ids, attention_mask, cnn_input
                # )
                # loss = criterion(logits, labels.to(torch.float16))

                alpha = 0.5
                fused_logits, llm_logits, cnn_logits = model(
                    input_ids,
                    attention_mask,
                    cnn_input,
                    prev_first_page_distance,
                    return_all_logits=True,
                )
                fused_loss = criterion(fused_logits, labels.to(torch.float16))
                aux_llm_loss = criterion(llm_logits, labels.to(torch.float16))
                aux_cnn_loss = criterion(cnn_logits, labels.to(torch.float16))
                loss = fused_loss + alpha * (aux_llm_loss + aux_cnn_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()

            if step % log_steps == 0:
                print(f"[STEP {step}] Epoch {epoch+1} | Loss: {loss.item():.4f}")

            # if step % config.log_steps == 0:
            #     pred_probs = torch.sigmoid(logits[:2]).detach().cpu().numpy()
            #     true_labels = labels[:2].cpu().numpy()
            #     print(f"[DEBUG] Predictions (first 2): {pred_probs}")
            #     print(f"[DEBUG] True labels (first 2): {true_labels}")

            if step % eval_steps == 0:
                eval_loss, acc, rec, prec, f1, cm = evaluate(
                    model, test_loader, criterion, device
                )
                scheduler.step(eval_loss)
                print(f"\n[Eval @ step {step}]")
                print(
                    f"  Loss: {eval_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}"
                )
                print(f"  Confusion Matrix:\n{cm}\n")

                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), SPLITTER_MODEL_PATH)
                    print(f"  ✅ Saved new best model (F1: {f1:.4f})")


if __name__ == "__main__":
    main()
