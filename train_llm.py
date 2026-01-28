import argparse
import dataclasses
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import AppConfig, set_nested_attr
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.nn.optimizer import AdamW, gradient_clipping, lr_cosine_schedule
from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.utils import get_logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def parse_args_to_config() -> AppConfig:
    cfg = AppConfig()
    parser = argparse.ArgumentParser(description="Train a language model.")

    # Data
    parser.add_argument("--train_data", default=cfg.data.train_data)
    parser.add_argument("--val_data", default=cfg.data.val_data)
    parser.add_argument("--dtype", default=cfg.data.dtype)
    parser.add_argument("--device", default=cfg.data.device)

    # Model
    parser.add_argument("--vocab_size", type=int, default=cfg.model.vocab_size)
    parser.add_argument("--context_length", type=int, default=cfg.model.context_length)
    parser.add_argument("--d_model", type=int, default=cfg.model.d_model)
    parser.add_argument("--n_layers", type=int, default=cfg.model.n_layers)
    parser.add_argument("--n_heads", type=int, default=cfg.model.n_heads)
    parser.add_argument("--d_ff", type=int, default=cfg.model.d_ff)
    parser.add_argument("--dropout", type=float, default=cfg.model.dropout)
    tie_group = parser.add_mutually_exclusive_group()
    tie_group.add_argument("--tie_embeddings", action="store_true", default=cfg.model.tie_embeddings)
    tie_group.add_argument("--no_tie_embeddings", action="store_false", dest="tie_embeddings")

    # Optimizer
    parser.add_argument("--lr", type=float, default=cfg.optim.lr)
    parser.add_argument("--betas", type=float, nargs=2, default=list(cfg.optim.betas))
    parser.add_argument("--weight_decay", type=float, default=cfg.optim.weight_decay)
    parser.add_argument("--eps", type=float, default=cfg.optim.eps)
    parser.add_argument("--grad_clip", type=float, default=cfg.optim.grad_clip)

    # Schedule
    parser.add_argument("--warmup_iters", type=int, default=cfg.schedule.warmup_iters)
    parser.add_argument("--cosine_cycle_iters", type=int, default=cfg.schedule.cosine_cycle_iters)
    parser.add_argument("--min_lr", type=float, default=cfg.schedule.min_lr)

    # Train
    parser.add_argument("--batch_size", type=int, default=cfg.train.batch_size)
    parser.add_argument("--max_iters", type=int, default=cfg.train.max_iters)
    parser.add_argument("--eval_every", type=int, default=cfg.train.eval_every)
    parser.add_argument("--eval_iters", type=int, default=cfg.train.eval_iters)
    parser.add_argument("--log_every", type=int, default=cfg.train.log_every)
    parser.add_argument("--seed", type=int, default=cfg.train.seed)
    parser.add_argument("--exp_name", default=cfg.train.experiment_name)

    # Checkpoint
    parser.add_argument("--ckpt_every", type=int, default=cfg.checkpoint.every)
    parser.add_argument("--resume", action="store_true", default=cfg.checkpoint.resume)

    # Logging
    parser.add_argument("--use_wandb", action="store_true", default=cfg.logging.use_wandb)
    parser.add_argument("--wandb_project", default=cfg.logging.wandb_project)
    parser.add_argument("--wandb_run_name", default=cfg.logging.wandb_run_name)

    args = parser.parse_args()

    # Apply CLI overrides to config
    set_nested_attr(cfg, "data.train_data", args.train_data)
    set_nested_attr(cfg, "data.val_data", args.val_data)
    set_nested_attr(cfg, "data.dtype", args.dtype)
    set_nested_attr(cfg, "data.device", args.device)

    set_nested_attr(cfg, "model.vocab_size", args.vocab_size)
    set_nested_attr(cfg, "model.context_length", args.context_length)
    set_nested_attr(cfg, "model.d_model", args.d_model)
    set_nested_attr(cfg, "model.n_layers", args.n_layers)
    set_nested_attr(cfg, "model.n_heads", args.n_heads)
    set_nested_attr(cfg, "model.d_ff", args.d_ff)
    set_nested_attr(cfg, "model.dropout", args.dropout)
    set_nested_attr(cfg, "model.tie_embeddings", args.tie_embeddings)

    set_nested_attr(cfg, "optim.lr", args.lr)
    set_nested_attr(cfg, "optim.betas", tuple(args.betas))
    set_nested_attr(cfg, "optim.weight_decay", args.weight_decay)
    set_nested_attr(cfg, "optim.eps", args.eps)
    set_nested_attr(cfg, "optim.grad_clip", args.grad_clip)

    set_nested_attr(cfg, "schedule.warmup_iters", args.warmup_iters)
    set_nested_attr(cfg, "schedule.cosine_cycle_iters", args.cosine_cycle_iters)
    set_nested_attr(cfg, "schedule.min_lr", args.min_lr)

    set_nested_attr(cfg, "train.batch_size", args.batch_size)
    set_nested_attr(cfg, "train.max_iters", args.max_iters)
    set_nested_attr(cfg, "train.eval_every", args.eval_every)
    set_nested_attr(cfg, "train.eval_iters", args.eval_iters)
    set_nested_attr(cfg, "train.log_every", args.log_every)
    set_nested_attr(cfg, "train.seed", args.seed)
    set_nested_attr(cfg, "train.experiment_name", args.exp_name)

    set_nested_attr(cfg, "checkpoint.every", args.ckpt_every)
    set_nested_attr(cfg, "checkpoint.resume", args.resume)

    set_nested_attr(cfg, "logging.use_wandb", args.use_wandb)
    set_nested_attr(cfg, "logging.wandb_project", args.wandb_project)
    set_nested_attr(cfg, "logging.wandb_run_name", args.wandb_run_name)

    return cfg


def print_config(cfg: AppConfig) -> None:
    console = Console()
    ckpt_root = "checkpoints"
    config_table = Table(title="Configuration", show_header=False, border_style="blue")
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Data.train_data", str(cfg.data.train_data))
    config_table.add_row("Data.val_data", str(cfg.data.val_data))
    config_table.add_row("Data.dtype", str(cfg.data.dtype))
    config_table.add_row("Data.device", str(cfg.data.device))

    config_table.add_row("Model.vocab_size", str(cfg.model.vocab_size))
    config_table.add_row("Model.context_length", str(cfg.model.context_length))
    config_table.add_row("Model.d_model", str(cfg.model.d_model))
    config_table.add_row("Model.n_layers", str(cfg.model.n_layers))
    config_table.add_row("Model.n_heads", str(cfg.model.n_heads))
    config_table.add_row("Model.d_ff", str(cfg.model.d_ff))
    config_table.add_row("Model.dropout", str(cfg.model.dropout))
    config_table.add_row("Model.tie_embeddings", str(cfg.model.tie_embeddings))

    config_table.add_row("Optim.lr", str(cfg.optim.lr))
    config_table.add_row("Optim.betas", str(cfg.optim.betas))
    config_table.add_row("Optim.weight_decay", str(cfg.optim.weight_decay))
    config_table.add_row("Optim.eps", str(cfg.optim.eps))
    config_table.add_row("Optim.grad_clip", str(cfg.optim.grad_clip))

    config_table.add_row("Schedule.warmup_iters", str(cfg.schedule.warmup_iters))
    config_table.add_row("Schedule.cosine_cycle_iters", str(cfg.schedule.cosine_cycle_iters))
    config_table.add_row("Schedule.min_lr", str(cfg.schedule.min_lr))

    config_table.add_row("Train.batch_size", str(cfg.train.batch_size))
    config_table.add_row("Train.max_iters", str(cfg.train.max_iters))
    config_table.add_row("Train.eval_every", str(cfg.train.eval_every))
    config_table.add_row("Train.eval_iters", str(cfg.train.eval_iters))
    config_table.add_row("Train.log_every", str(cfg.train.log_every))
    config_table.add_row("Train.seed", str(cfg.train.seed))
    config_table.add_row("Train.experiment_name", str(cfg.train.experiment_name))

    config_table.add_row("Checkpoint.run_dir", f"{ckpt_root}/{cfg.train.experiment_name}")
    config_table.add_row("Checkpoint.every", str(cfg.checkpoint.every))
    config_table.add_row("Checkpoint.resume", str(cfg.checkpoint.resume))

    config_table.add_row("Logging.use_wandb", str(cfg.logging.use_wandb))
    config_table.add_row("Logging.wandb_project", str(cfg.logging.wandb_project))
    config_table.add_row("Logging.wandb_run_name", str(cfg.logging.wandb_run_name))

    console.print(Panel("[bold blue]Training Configuration[/]", expand=False))
    console.print(config_table)
    console.print()


def _build_model(cfg: AppConfig, device: torch.device) -> torch.nn.Module:
    model = TransformerLM(
        d_model=cfg.model.d_model,
        num_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        num_layers=cfg.model.n_layers,
    ).to(device)
    if cfg.model.tie_embeddings:
        model.linear.weight = model.token_embedding.weight
    return model


def _load_memmap(path: str, dtype: str) -> np.memmap:
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")


def _compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))


@torch.no_grad()
def _estimate_loss(
    model: torch.nn.Module,
    data: np.ndarray,
    cfg: AppConfig,
    device: torch.device,
    eval_iters: int = 10,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, cfg.train.batch_size, cfg.model.context_length, device=str(device))
        logits = model(x)
        loss = _compute_loss(logits, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def _checkpoint_paths(exp_name: str, step: int) -> tuple[Path, Path]:
    run_dir = Path("checkpoints") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    step_path = run_dir / f"step_{step:08d}.pt"
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    return step_path, latest_path, best_path


def main():
    cfg = parse_args_to_config()
    logger = get_logger(__name__)
    print_config(cfg)
    logger.info("Configuration printed via rich console.")

    if not cfg.data.train_data:
        logger.error("Missing --train_data path.")
        return

    device = torch.device(cfg.data.device)
    torch.manual_seed(cfg.train.seed)

    logger.info("Loading datasets with np.memmap...")
    train_data = _load_memmap(cfg.data.train_data, cfg.data.dtype)
    val_data = None
    if cfg.data.val_data:
        val_data = _load_memmap(cfg.data.val_data, cfg.data.dtype)

    model = _build_model(cfg, device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
    )

    use_wandb = cfg.logging.use_wandb
    if use_wandb:
        try:
            import wandb  # type: ignore
        except ModuleNotFoundError:
            logger.warning("wandb is not installed; disabling wandb logging.")
            use_wandb = False
        else:
            wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.logging.wandb_run_name or cfg.train.experiment_name,
                config=dataclasses.asdict(cfg),
            )

    start_it = 0
    if cfg.checkpoint.resume:
        _, latest_path, _ = _checkpoint_paths(cfg.train.experiment_name, step=0)
        if latest_path.exists():
            start_it = load_checkpoint(latest_path, model, optimizer)
            logger.info("Resumed from %s at iteration %s", latest_path, start_it)
        else:
            logger.warning("Resume requested but no checkpoint found at %s", latest_path)

    logger.info("Starting training loop.")
    t0 = time.time()
    best_val_loss = float("inf")
    for it in range(start_it, cfg.train.max_iters):
        lr = lr_cosine_schedule(
            it=it,
            max_learning_rate=cfg.optim.lr,
            min_learning_rate=cfg.schedule.min_lr,
            warmup_iters=cfg.schedule.warmup_iters,
            cosine_cycle_iters=cfg.schedule.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, cfg.train.batch_size, cfg.model.context_length, device=str(device))
        logits = model(x)
        loss = _compute_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        if cfg.optim.grad_clip and cfg.optim.grad_clip > 0:
            gradient_clipping(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()

        if it % cfg.train.log_every == 0:
            elapsed = time.time() - t0
            logger.info("it=%d loss=%.4f lr=%.6g time=%.1fs", it, loss.item(), lr, elapsed)
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "lr": lr, "time": elapsed}, step=it)

        if val_data is not None and it % cfg.train.eval_every == 0:
            val_loss = _estimate_loss(
                model,
                val_data,
                cfg,
                device,
                eval_iters=cfg.train.eval_iters,
            )
            logger.info("it=%d val_loss=%.4f", it, val_loss)
            if use_wandb:
                wandb.log({"val/loss": val_loss}, step=it)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _, _, best_path = _checkpoint_paths(cfg.train.experiment_name, it)
                save_checkpoint(model, optimizer, it, best_path)
                logger.info("Saved best checkpoint to %s", best_path)

        if cfg.checkpoint.every > 0 and it % cfg.checkpoint.every == 0 and it != start_it:
            step_path, latest_path, _ = _checkpoint_paths(cfg.train.experiment_name, it)
            save_checkpoint(model, optimizer, it, step_path)
            save_checkpoint(model, optimizer, it, latest_path)
            logger.info("Saved checkpoint to %s", step_path)

    logger.info("Training complete.")

if __name__ == "__main__":
    main()
