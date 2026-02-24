import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import wandb
import os
import shutil
from datetime import datetime

from model import BaselineModel
from data import get_data, resolve_special_token_ids
from typing import NotRequired, TypedDict
from config import Config

def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_optimizer_state_to_device(
    optimizer: optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def get_uncompiled_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def maybe_compile_model(
    model: torch.nn.Module, device: torch.device, config: Config
) -> tuple[torch.nn.Module, bool, str]:
    if not config.get("use_torch_compile", False):
        return model, False, "disabled"
    if not hasattr(torch, "compile"):
        return model, False, "torch.compile unavailable"
    if device.type != "cuda":
        return model, False, f"skipped on {device.type}"

    try:
        compiled_model = torch.compile(
            model,
            mode=config.get("torch_compile_mode", "default"),
            fullgraph=bool(config.get("torch_compile_fullgraph", False)),
            dynamic=bool(config.get("torch_compile_dynamic", False)),
        )
        return compiled_model, True, "enabled"
    except Exception as exc:
        return model, False, f"failed: {exc}"


def find_latest_artifact_dir_with_checkpoint(
    models_root: str, checkpoint_filename: str
) -> str | None:
    latest_dir = None
    latest_mtime = float("-inf")

    for entry in os.scandir(models_root):
        if not entry.is_dir():
            continue
        checkpoint_path = os.path.join(entry.path, checkpoint_filename)
        if not os.path.exists(checkpoint_path):
            continue

        checkpoint_mtime = os.path.getmtime(checkpoint_path)
        if checkpoint_mtime > latest_mtime:
            latest_mtime = checkpoint_mtime
            latest_dir = entry.path

    return latest_dir


def prepare_run_artifact_paths(config: Config) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configured_models_root = config.get("models_root_dir")
    if configured_models_root:
        models_root = (
            configured_models_root
            if os.path.isabs(configured_models_root)
            else os.path.join(script_dir, configured_models_root)
        )
    else:
        models_root = os.path.join(script_dir, "models")
    os.makedirs(models_root, exist_ok=True)

    original_checkpoint_path = config["checkpoint_path"]
    original_checkpoint_path = (
        original_checkpoint_path
        if os.path.isabs(original_checkpoint_path)
        else os.path.join(script_dir, original_checkpoint_path)
    )

    checkpoint_filename = os.path.basename(config["checkpoint_path"])
    final_model_filename = os.path.basename(config["final_model_path"])

    run_dir = None
    if config.get("run_name"):
        run_dir = os.path.join(models_root, config["run_name"])
        os.makedirs(run_dir, exist_ok=True)
    elif config["resume_from_checkpoint"]:
        run_dir = find_latest_artifact_dir_with_checkpoint(
            models_root=models_root,
            checkpoint_filename=checkpoint_filename,
        )
        if run_dir:
            print(f"Resuming artifacts from: {run_dir}")

    if run_dir is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join(models_root, run_name)
        os.makedirs(run_dir, exist_ok=True)

    config["run_artifact_dir"] = run_dir
    config["checkpoint_path"] = os.path.join(run_dir, checkpoint_filename)
    config["final_model_path"] = os.path.join(run_dir, final_model_filename)
    config["model_diagram_path"] = os.path.join(run_dir, "baseline_model_architecture")
    if (
        config["resume_from_checkpoint"]
        and os.path.exists(original_checkpoint_path)
        and original_checkpoint_path != config["checkpoint_path"]
        and not os.path.exists(config["checkpoint_path"])
    ):
        shutil.copy2(original_checkpoint_path, config["checkpoint_path"])
        print(
            f"Copied checkpoint for resume: {original_checkpoint_path} -> {config['checkpoint_path']}"
        )
    print(f"Run artifacts will be saved to: {run_dir}")

# ==================================================================


def make(
    config: Config,
    pin_memory: bool = False,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    BaselineModel,
    optim.Optimizer,
    CrossEntropyLoss,
]:
    _, _, _, _, pad_id = resolve_special_token_ids(config)
    dataloader, val_dataloader = get_data(config, pin_memory)
    # Create the model
    model = BaselineModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        layers=config["layers"],
        pad_id=pad_id,
    )

    # log model parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Model param names: {[name for name, _ in model.named_parameters()]}")
    # save image of model architecture
    try:
        from torchviz import make_dot
        sample_input, _, sample_key_padding_mask = next(iter(dataloader))
        sample_input = sample_input.to(get_best_device())
        sample_key_padding_mask = sample_key_padding_mask.to(get_best_device())
        model_viz = make_dot(
            model(sample_input, key_padding_mask=sample_key_padding_mask),
            params=dict(model.named_parameters()),
        )
        model_viz.format = "png"
        model_diagram_path = config.get(
            "model_diagram_path", "baseline_model_architecture"
        )
        model_viz.render(model_diagram_path)
        print(f"Saved model architecture visualization to {model_diagram_path}.png")
    except ImportError:
        print("torchviz not installed, skipping model architecture visualization.")
    except Exception as exc:
        print(f"Skipping model architecture visualization: {exc}")
    
    # Create the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    return dataloader, val_dataloader, model, optimizer, loss_fn


def evaluate(
    model: BaselineModel,
    loader: torch.utils.data.DataLoader,
    loss_fn: CrossEntropyLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    pad_id = int(loss_fn.ignore_index)
    total_tokens = 0
    total_loss = 0.0
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for input_seq, target_seq, key_padding_mask in loader:
            input_seq = input_seq.to(device, non_blocking=non_blocking)
            target_seq = target_seq.to(device, non_blocking=non_blocking)
            key_padding_mask = key_padding_mask.to(device, non_blocking=non_blocking)

            output = model(input_seq, key_padding_mask=key_padding_mask)
            loss_sum = loss_fn(
                output.reshape(-1, output.size(-1)),
                target_seq.reshape(-1),
            )

            valid_target_mask = target_seq != pad_id
            tokens = int(valid_target_mask.sum().item())
            if tokens == 0:
                continue

            total_tokens += tokens
            total_loss += loss_sum.item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
    }


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: CrossEntropyLoss,
    optimizer: optim.Optimizer,
    config: Config,
    run,
    device: torch.device,
):
    checkpoint_model = get_uncompiled_model(model)
    pad_id = int(loss_fn.ignore_index)

    def save_checkpoint(epoch: int, next_batch_idx: int, global_step: int):
        checkpoint = {
            "epoch": epoch,
            "batch_idx": next_batch_idx,
            "global_step": global_step,
            "model_state_dict": checkpoint_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(config),
        }
        torch.save(checkpoint, config["checkpoint_path"])

    def load_checkpoint_if_available() -> tuple[int, int, int]:
        if not config["resume_from_checkpoint"]:
            return 0, 0, 0

        checkpoint_path = config["checkpoint_path"]
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}, starting fresh.")
            return 0, 0, 0

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state_dict = dict(checkpoint["model_state_dict"])
        # Backward compatibility: old checkpoints may serialize the positional cache buffer.
        model_state_dict.pop("pos_encoding.pos_enc_cache", None)
        checkpoint_model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)

        start_epoch = checkpoint.get("epoch", 0)
        start_batch_idx = checkpoint.get("batch_idx", 0)
        global_step = checkpoint.get("global_step", 0)
        print(
            f"Resumed from {checkpoint_path} at epoch={start_epoch}, batch={start_batch_idx}, step={global_step}"
        )
        return start_epoch, start_batch_idx, global_step

    model.train()
    non_blocking = device.type == "cuda"

    start_epoch, start_batch_idx, global_step = load_checkpoint_if_available()

    for epoch in tqdm(range(start_epoch, config["epochs"]), desc="Epochs"):
        epoch_train_loss_sum = 0.0
        epoch_token_count = 0

        for batch_idx, (input_seq, target_seq, key_padding_mask) in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            input_seq = input_seq.to(device, non_blocking=non_blocking)
            target_seq = target_seq.to(device, non_blocking=non_blocking)
            key_padding_mask = key_padding_mask.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            output = model(input_seq, key_padding_mask=key_padding_mask)
            loss_sum = loss_fn(
                output.reshape(-1, output.size(-1)),
                target_seq.reshape(-1),
            )
            valid_tokens = int((target_seq != pad_id).sum().item())
            if valid_tokens == 0:
                continue

            loss = loss_sum / valid_tokens
            loss.backward()
            optimizer.step()
            global_step += 1

            epoch_token_count += valid_tokens
            epoch_train_loss_sum += loss_sum.item()

            if batch_idx % 10 == 0:
                run.log(
                    {
                        "train_loss_step": loss.item(),
                        "epoch": epoch + (batch_idx + 1) / max(len(train_loader), 1),
                    },
                    step=global_step,
                )

            if (
                config["checkpoint_every_n_steps"] > 0
                and global_step % config["checkpoint_every_n_steps"] == 0
            ):
                next_epoch = epoch
                next_batch_idx = batch_idx + 1
                if next_batch_idx >= len(train_loader):
                    next_epoch += 1
                    next_batch_idx = 0

                save_checkpoint(next_epoch, next_batch_idx, global_step)
                run.save(config["checkpoint_path"])

        avg_train_loss = epoch_train_loss_sum / max(epoch_token_count, 1)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        model.train()

        run.log(
            {
                "epoch": epoch + 1,
                "train_loss_epoch": avg_train_loss,
                **val_metrics,
            },
            step=global_step,
        )

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_perplexity={val_metrics['val_perplexity']:.4f}"
        )

    # Save a final checkpoint marker at the end of training.
    save_checkpoint(config["epochs"], 0, global_step)
    run.save(config["checkpoint_path"])

    # Save final trained model weights.
    torch.save(checkpoint_model.state_dict(), config["final_model_path"])
    run.save(config["final_model_path"])

def model_pipeline(config: Config, project_name: str):
    device = get_best_device()
    print(f"Using device: {device}")
    prepare_run_artifact_paths(config)

    train_loader, val_loader, model, optimizer, loss_fn = make(
        config, pin_memory=device.type == "cuda"
    )
    model = model.to(device)

    with wandb.init(project=project_name, config=dict(config)) as run:
        model, compile_enabled, compile_status = maybe_compile_model(
            model, device, config
        )
        run.config.update(
            {
                "device": str(device),
                "torch_compile_enabled": compile_enabled,
                "torch_compile_status": compile_status,
            },
            allow_val_change=True,
        )
        print(f"torch.compile: {compile_status}")

        run.watch(get_uncompiled_model(model), loss_fn, log="all", log_freq=10)
        train(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            config,
            run,
            device,
        )

    return model

