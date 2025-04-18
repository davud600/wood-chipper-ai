import argparse
import json
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
    mlp_warmup_steps,
)
from .utils import count_classes, evaluate
from .model import FusionModel
from .dataset.dataset import DocumentDataset
from config.settings import (
    TRAINING_DATA_CSV,
    TESTING_DATA_CSV,
    SPLITTER_MODEL_DIR,
    image_output_size,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--lr", type=float, default=learning_rate)
    p.add_argument("--wd", type=float, default=weight_decay)
    p.add_argument("--pw_multiplier", type=float, default=0.5)

    return p.parse_args()


def train_loop(
    model, train_dataset, test_dataset, cnn_warmup_steps, llm_warmup_steps, pw, args
):
    train_loader = DataLoader(
        train_dataset, batch_size=training_mini_batch_size, shuffle=False, num_workers=6
    )
    test_loader = DataLoader(
        test_dataset, batch_size=testing_mini_batch_size, shuffle=False, num_workers=6
    )

    # # debug - start
    # batch = next(iter(train_loader))
    # for k, v in batch.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"[check] batch[{k}] on: {v.device}")
    # # debug - end

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    llm_params = model.reader_model.parameters()
    cnn_params = model.cnn_model.parameters()
    fusion_params = model.fusion_mlp.parameters()

    opt_llm = torch.optim.AdamW(llm_params, lr=args.lr, weight_decay=args.wd)
    opt_cnn = torch.optim.AdamW(cnn_params, lr=args.lr, weight_decay=args.wd)
    opt_fusion = torch.optim.AdamW(
        fusion_params, lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt_fusion, patience=patience, factor=factor
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

                    if step % eval_steps == 0:
                        eval_loss, acc, rec, prec, f1, cm = evaluate(
                            model.cnn_model, test_loader, loss_fn
                        )
                        print(
                            f"  Loss: {eval_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}"
                        )
                        print(f"  Confusion Matrix:\n{cm}\n")

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

                    if step % eval_steps == 0:
                        eval_loss, acc, rec, prec, f1, cm = evaluate(
                            model.reader_model, test_loader, loss_fn
                        )
                        print(
                            f"  Loss: {eval_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}"
                        )
                        print(f"  Confusion Matrix:\n{cm}\n")

                elif step <= llm_warmup_steps + cnn_warmup_steps + mlp_warmup_steps:
                    for p in model.reader_model.parameters():
                        p.requires_grad = False
                    for p in model.cnn_model.parameters():
                        p.requires_grad = False

                    opt_fusion.zero_grad()
                    _, loss = model.forward(batch, loss_fn, warmup=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt_fusion)
                    scaler.update()

                    if step % log_steps == 0:
                        print(
                            f"[MLP Warmup] [STEP {step}] Epoch {epoch+1} | Loss: {loss.item():.4f}"
                        )

                    if step % eval_steps == 0:
                        eval_loss, acc, rec, prec, f1, cm = evaluate(
                            model, test_loader, loss_fn
                        )
                        print(
                            f"  Loss: {eval_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}"
                        )
                        print(f"  Confusion Matrix:\n{cm}\n")

                else:
                    for p in model.reader_model.parameters():
                        p.requires_grad = True
                    for p in model.cnn_model.parameters():
                        p.requires_grad = True

                    opt_fusion.zero_grad()
                    opt_cnn.zero_grad()
                    opt_llm.zero_grad()
                    _, loss = model.forward(
                        batch,
                        loss_fn,
                    )
                    scaler.scale(loss).backward()
                    scaler.step(opt_fusion)
                    scaler.step(opt_cnn)
                    scaler.step(opt_llm)
                    scaler.update()

                    if step % log_steps == 0:
                        print(
                            f"[STEP {step}] Epoch {epoch+1} | Loss: {loss.item():.4f}"
                        )

                    if step % eval_steps == 0:
                        eval_loss, acc, rec, prec, f1, cm = evaluate(
                            model, test_loader, loss_fn
                        )
                        print(
                            f"  Loss: {eval_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}"
                        )
                        print(f"  Confusion Matrix:\n{cm}\n")

                        scheduler.step(eval_loss)
                        if f1 > best_f1:
                            best_f1 = f1
                            torch.save(
                                model.state_dict(),
                                f"{SPLITTER_MODEL_DIR}/model_{f1:.4f}.pth",
                            )

                            with open(
                                f"{SPLITTER_MODEL_DIR}/model_{f1:.4f}.json", "w"
                            ) as fp:
                                json.dump(
                                    {
                                        "lr": args.lr,
                                        "wd": args.wd,
                                        "step": step,
                                        "f1": f1,
                                    },
                                    fp,
                                    indent=2,
                                )
                            print(f"  ✅ Saved new best model (F1: {f1:.4f})")


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

    args = parse_args()
    pw = torch.tensor([(n0 / n1) ** args.pw_multiplier], dtype=torch.float32).to(device)

    train_loop(
        model,
        train_dataset,
        test_dataset,
        cnn_warmup_steps,
        llm_warmup_steps,
        pw,
        args,
    )
