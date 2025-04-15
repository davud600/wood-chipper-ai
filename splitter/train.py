import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from splitter.models.cnn_model import CNNModel
from splitter.models.llm_model import ReaderModel

from .utils import count_classes, evaluate
from .model import FusionModel
from .dataset.dataset import DocumentDataset
from config.settings import (
    TRAINING_DATA_CSV,
    TESTING_DATA_CSV,
    SPLITTER_MODEL_PATH,
    image_output_size,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_mini_batch_size = 20
testing_mini_batch_size = 20
learning_rate = 0.000075
weight_decay = 0.0005
patience = 15
factor = 0.5
epochs = 15
log_steps = 50
eval_steps = 100
cnn_warmup_steps = 100
llm_warmup_steps = 100


def eval_and_save(model, scheduler, step, loss_fn, test_loader, best_f1):
    print(f"\n[Eval @ step {step}]")
    eval_loss, acc, rec, prec, f1, cm = evaluate(model, test_loader, loss_fn, device)
    scheduler.step(eval_loss)

    print(
        f"  Loss: {eval_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}"
    )

    print(f"  Confusion Matrix:\n{cm}\n")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), SPLITTER_MODEL_PATH)
        print(f"  ✅ Saved new best model (F1: {f1:.4f})")


def forward_and_get_loss(model, data, loss_fn):
    labels = data["labels"].to(torch.float16)

    if isinstance(model, ReaderModel):
        logits = model(
            data["input_ids"], data["attention_mask"], data["prev_first_page_distance"]
        )

        # debugging - start
        true_labels = labels[:1].cpu().numpy()
        pred_probs = torch.sigmoid(logits[:1]).detach().cpu().numpy()
        print(f"[DEBUG] True labels: {true_labels.squeeze(1)}")
        print(f"[DEBUG] LLM pred: {pred_probs.squeeze(1)}")
        # debugging - start

        return loss_fn(logits, labels)

    elif isinstance(model, CNNModel):
        logits = model(data["cnn_input"], data["prev_first_page_distance"])

        # debugging - start
        true_labels = labels[:1].cpu().numpy()
        pred_probs = torch.sigmoid(logits[:1]).detach().cpu().numpy()
        print(f"[DEBUG] True labels: {true_labels.squeeze(1)}")
        print(f"[DEBUG] CNN pred: {pred_probs.squeeze(1)}")
        # debugging - start

        return loss_fn(logits, labels)

    else:
        fused_logits, llm_logits, cnn_logits = model(
            data["input_ids"],
            data["attention_mask"],
            data["cnn_input"],
            data["prev_first_page_distance"],
            return_all_logits=True,
        )

        fused_loss = loss_fn(fused_logits, labels)
        aux_llm_loss = loss_fn(llm_logits, labels)
        aux_cnn_loss = loss_fn(cnn_logits, labels)

        alpha = 0.5
        loss = fused_loss + alpha * (aux_llm_loss + aux_cnn_loss)

        # debugging - start
        true_labels = labels[:1].cpu().numpy()
        fusion_pred_probs = torch.sigmoid(fused_logits[:1]).detach().cpu().numpy()
        cnn_pred_probs = torch.sigmoid(cnn_logits[:1]).detach().cpu().numpy()
        llm_pred_probs = torch.sigmoid(llm_logits[:1]).detach().cpu().numpy()
        print(f"[DEBUG] True labels: {true_labels.squeeze(1)}")
        print(f"[DEBUG] Fusion pred: {fusion_pred_probs.squeeze(1)}")
        print(f"[DEBUG] CNN pred: {cnn_pred_probs.squeeze(1)}")
        print(f"[DEBUG] LLM pred: {llm_pred_probs.squeeze(1)}")
        # debugging - start

        return loss


def get_data_from_loader(batch):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    cnn_input = batch["cnn_input"].to(device)
    labels = batch["labels"].to(device)
    prev_first_page_distance = batch["prev_first_page_distance"].to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "cnn_input": cnn_input,
        "labels": labels,
        "prev_first_page_distance": prev_first_page_distance,
    }


def train_loop(
    train_dataset, test_dataset, N0, N1, model, cnn_warmup_steps, llm_warmup_steps
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_mini_batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=testing_mini_batch_size)

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
            data = get_data_from_loader(batch)
            step += 1

            with torch.amp.autocast_mode.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                if step <= cnn_warmup_steps:
                    opt_cnn.zero_grad()
                    loss = forward_and_get_loss(model.cnn_model, data, loss_fn)
                    scaler.scale(loss).backward()
                    scaler.step(opt_cnn)
                    scaler.update()

                    if step % log_steps == 0:
                        print(
                            f"[CNN Warmup] [STEP {step}] Epoch {epoch+1} | Loss: {loss.item():.4f}"
                        )

                elif step <= llm_warmup_steps + cnn_warmup_steps:
                    opt_llm.zero_grad()
                    loss = forward_and_get_loss(model.reader_model, data, loss_fn)
                    scaler.scale(loss).backward()
                    scaler.step(opt_llm)
                    scaler.update()

                    if step % log_steps == 0:
                        print(
                            f"[LLM Warmup] [STEP {step}] Epoch {epoch+1} | Loss: {loss.item():.4f}"
                        )

                else:
                    opt_fusion.zero_grad()
                    loss = forward_and_get_loss(model, data, loss_fn)
                    scaler.scale(loss).backward()
                    scaler.step(opt_fusion)
                    scaler.update()

                    if step % log_steps == 0:
                        print(
                            f"[STEP {step}] Epoch {epoch+1} | Loss: {loss.item():.4f}"
                        )

                    if step % eval_steps == 0:
                        eval_and_save(
                            model, scheduler, step, loss_fn, test_loader, best_f1
                        )


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
    model.load_state_dict(
        torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
    )

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
