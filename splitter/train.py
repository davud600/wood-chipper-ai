import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from textwrap import dedent
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import (
    device,
    training_mini_batch_size,
    testing_mini_batch_size,
    patience,
    factor,
    epochs,
    log_steps,
    eval_steps,
    lr_llm,
    wd_llm,
    isolated_epochs_llm,
    lr_cnn,
    wd_cnn,
    isolated_epochs_cnn,
    lr_mlp,
    wd_mlp,
    pw_multiplier,
    use_fp16,
    use_all_types,
)
from .utils import count_classes, evaluate, load_best_weights
from .model import FusionModel
from .dataset.dataset import DocumentDataset
from config.settings import (
    TRAINING_DATA_CSV,
    TESTING_DATA_CSV,
    SPLITTER_MODEL_DIR,
    BEST_PERF_TYPES,
    image_output_size,
)

session_dirs = [
    int(d)
    for d in os.listdir(SPLITTER_MODEL_DIR)
    if os.path.isdir(os.path.join(SPLITTER_MODEL_DIR, d)) and d.isdigit()
]
session = max(session_dirs, default=-1) + 1

best_f1 = {
    "fusion": 0.0,
    "mlp": 0.0,
    "cnn": 0.0,
    "llm": 0.0,
}


def step_model(
    model,
    optimizer,
    scheduler,
    scaler,
    batch,
    test_loader,
    loss_fn,
    step: int,
    epoch: float,
):
    optimizer.zero_grad()
    logits, loss = model(batch, loss_fn)
    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)  # ??
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

    if step % log_steps == 0:
        print(
            f"[{model.title.upper()}] [STEP {step}] Epoch {epoch:.4f} | Loss: {loss.item():.4f} | true: {int(batch['labels'][0])} | pred: {torch.sigmoid(logits.squeeze(-1)[0]):.2f} | lr: {optimizer.param_groups[-1]['lr']}"
        )

    if step % eval_steps == 0:
        global session
        global best_f1

        eval_loss, acc, rec, prec, f1, cm = evaluate(model, test_loader, loss_fn)
        print(
            f"  Loss: {eval_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}"
        )
        print(f"  Confusion Matrix:\n{cm}\n")

        scheduler.step(eval_loss)
        if f1 >= best_f1[model.title]:
            best_f1[model.title] = float(f1)

            torch.save(
                model.state_dict(),
                f"{SPLITTER_MODEL_DIR}/{session}/{model.title}_model_{f1:.4f}.pth",
            )
            print(f"  ✅ Saved model [{model.title.upper()}] (F1: {f1:.4f})")


def train_loop(model, train_dataset, test_dataset, pw, args):
    global session

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.training_mini_batch_size,
        shuffle=True,
        num_workers=12,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.testing_mini_batch_size,
        shuffle=True,
        num_workers=12,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    # loss_fn = nn.BCEWithLogitsLoss()

    opt_mlp = optim.AdamW(
        [
            {
                "params": model.reader_model.backbone.parameters(),
                "lr": args.lr_llm * 0.0025,
                "weight_decay": args.wd_llm * 0.0025,
            },
            {
                "params": model.reader_model.classifier.parameters(),
                "lr": args.lr_llm,
                "weight_decay": args.wd_llm,
            },
            {
                "params": model.cnn_model.parameters(),
                "lr": args.lr_cnn,
                "weight_decay": args.wd_cnn,
            },
            {
                "params": model.fusion_mlp.parameters(),
                "lr": args.lr_mlp,
                "weight_decay": args.wd_mlp,
            },
        ],
        weight_decay=args.wd_mlp,
    )
    sched_mlp = optim.lr_scheduler.ReduceLROnPlateau(
        opt_mlp, patience=patience, factor=factor
    )

    opt_cnn = optim.AdamW(
        model.cnn_model.parameters(), lr=args.lr_cnn, weight_decay=args.wd_cnn
    )
    sched_cnn = optim.lr_scheduler.ReduceLROnPlateau(
        opt_cnn, patience=patience, factor=factor
    )

    opt_llm = torch.optim.AdamW(
        [
            {
                "params": model.reader_model.backbone.parameters(),
                "lr": args.lr_llm * 0.0025,
                "weight_decay": args.wd_llm * 0.0025,
            },
            {
                "params": model.reader_model.classifier.parameters(),
                "lr": args.lr_llm,
                "weight_decay": args.wd_llm,
            },
        ],
        betas=(0.9, 0.98),
    )
    sched_llm = optim.lr_scheduler.ReduceLROnPlateau(
        opt_llm, patience=patience, factor=factor
    )

    scaler = torch.amp.grad_scaler.GradScaler()

    print("[INFO] Starting training...")
    config_log = dedent(
        f"""
        === Training Configuration - Session #{session} ===

        [LLM Optimizer]
          Learning Rate      : {args.lr_llm}
          Weight Decay       : {args.wd_llm}
          Isolated Epochs    : {args.isolated_epochs_llm}

        [CNN Optimizer]
          Learning Rate      : {args.lr_cnn}
          Weight Decay       : {args.wd_cnn}
          Isolated Epochs    : {args.isolated_epochs_cnn}

        [MLP Optimizer]
          Learning Rate      : {args.lr_mlp}
          Weight Decay       : {args.wd_mlp}

        [General]
          Epochs             : {args.epochs}
          Pos‑Weight         : {pw.item():.4f}
          Train Batch Size   : {args.training_mini_batch_size}
          Test  Batch Size   : {args.testing_mini_batch_size}

        ==============================
    """
    ).strip()
    print(config_log)

    with torch.amp.autocast_mode.autocast(
        device_type=str(device),
        dtype=(torch.float16 if use_fp16 else torch.float32),
    ):

        step = 0
        for epoch in range(args.isolated_epochs_cnn):
            model.train()
            epoch_step = 0
            for batch in train_loader:
                step += 1
                epoch_step += 1
                step_model(
                    model.cnn_model,
                    opt_cnn,
                    sched_cnn,
                    scaler,
                    batch,
                    test_loader,
                    loss_fn,
                    step,
                    epoch + (epoch_step / len(train_loader)),
                )

        step = 0
        for epoch in range(args.isolated_epochs_llm):
            model.train()
            epoch_step = 0
            for batch in train_loader:
                step += 1
                epoch_step += 1
                step_model(
                    model.reader_model,
                    opt_llm,
                    sched_llm,
                    scaler,
                    batch,
                    test_loader,
                    loss_fn,
                    step,
                    epoch + (epoch_step / len(train_loader)),
                )

        step = 0
        load_best_weights(model, session)
        for epoch in range(args.epochs):
            model.train()
            epoch_step = 0
            for batch in train_loader:
                step += 1
                epoch_step += 1
                step_model(
                    model,
                    opt_mlp,
                    sched_mlp,
                    scaler,
                    batch,
                    test_loader,
                    loss_fn,
                    step,
                    epoch + (epoch_step / len(train_loader)),
                )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--lr_llm", type=float, default=lr_llm)
    p.add_argument("--wd_llm", type=float, default=wd_llm)
    p.add_argument("--isolated_epochs_llm", type=int, default=isolated_epochs_llm)

    p.add_argument("--lr_cnn", type=float, default=lr_cnn)
    p.add_argument("--wd_cnn", type=float, default=wd_cnn)
    p.add_argument("--isolated_epochs_cnn", type=int, default=isolated_epochs_cnn)

    p.add_argument("--lr_mlp", type=float, default=lr_mlp)
    p.add_argument("--wd_mlp", type=float, default=wd_mlp)

    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--pw_multiplier", type=float, default=pw_multiplier)
    p.add_argument(
        "--training_mini_batch_size", type=int, default=training_mini_batch_size
    )
    p.add_argument(
        "--testing_mini_batch_size", type=int, default=testing_mini_batch_size
    )

    return p.parse_args()


if __name__ == "__main__":
    """
    Main training loop for the FusionModel.

    Initializes tokenizer, datasets, model, optimizer, scheduler, and AMP scaler.
    Trains main model and branch models individually with
    mixed precision and evaluates it periodically.
    Saves the best model based on F1 score.
    """

    os.makedirs(os.path.join(SPLITTER_MODEL_DIR, str(session)), exist_ok=True)

    print("[INFO] Initializing...")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = FusionModel(image_size=image_output_size, tokenizer_len=len(tokenizer)).to(
        device
    )
    load_best_weights(model, session, True)

    print(f"[check] model on: {next(model.parameters()).device}")

    print("[training]")
    train_dataset = DocumentDataset(
        TRAINING_DATA_CSV,
        tokenizer,
        mode="train",
        image_size=image_output_size,
        doc_types=(None if use_all_types else BEST_PERF_TYPES),
    )
    n1, n0 = count_classes(train_dataset)

    print("[testing]")
    test_dataset = DocumentDataset(
        TESTING_DATA_CSV,
        tokenizer,
        mode="test",
        image_size=image_output_size,
        doc_types=(None if use_all_types else BEST_PERF_TYPES),
    )
    count_classes(test_dataset)

    args = parse_args()
    pw = torch.tensor(
        [(n0 / n1) * args.pw_multiplier],
        dtype=(torch.float16 if use_fp16 else torch.float32),
    ).to(device)

    train_loop(
        model,
        train_dataset,
        test_dataset,
        pw,
        args,
    )
