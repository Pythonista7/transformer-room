import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb
import os

from model import BaselineModel
from tokenizer import BPETokenizer
from typing import NotRequired, TypedDict

# ===

class Config(TypedDict):
    vocab_size: int
    d_model: int
    n_heads: int
    layers: int
    learning_rate: float
    epochs: int
    training_seq_len: int
    batch_size: int
    checkpoint_every_n_steps: int
    checkpoint_path: str
    final_model_path: str
    resume_from_checkpoint: bool
    use_torch_compile: NotRequired[bool]
    torch_compile_mode: NotRequired[str]
    torch_compile_fullgraph: NotRequired[bool]
    torch_compile_dynamic: NotRequired[bool]


config: Config = {
    "vocab_size": 10_000,
    "d_model": 128,
    "n_heads": 8,
    "layers": 2,
    "learning_rate": 1e-3,
    "epochs": 3,
    "training_seq_len": 128,
    "batch_size": 256,
    "checkpoint_every_n_steps": 250,
    "checkpoint_path": "baseline_checkpoint.pt",
    "final_model_path": "baseline_model.pt",
    "resume_from_checkpoint": True,
    "use_torch_compile": True,
    "torch_compile_mode": "default",
    "torch_compile_fullgraph": False,
    "torch_compile_dynamic": False,
}

# ===


class CustomShakespeareDataset(Dataset):
    """
    IMPORTANT: REMEBER TO USE JAGGED/NESTED TENSORS AS KEY PADDING IS NOT IMPLEMENTED IN THE ATTENTION LAYER!
    """

    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        input_seq = self.tokens[idx : idx + self.seq_len]
        target_seq = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(
            target_seq, dtype=torch.long
        )


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
    # Load up the data and tokenize it
    corpus = None
    with open("../datasets/tiny_shakespeare.txt") as f:
        corpus = f.read()

    if corpus is None:
        raise Exception("Invalid dataset read!")

    tokenizer = BPETokenizer(
        corpus=corpus,
        max_vocab_count=config["vocab_size"],
        path="tiny_shakespeare_bpe_vocab.txt",
    )

    # Create a vocab and encode the corpus
    VOCAB = tokenizer.vocab
    VOCAB_SIZE = len(VOCAB)
    assert (
        VOCAB_SIZE == config["vocab_size"]
    ), f"Tokenizer vocab size {VOCAB_SIZE} does not match config vocab size {config['vocab_size']}"

    print(f"Vocabulary size: {VOCAB_SIZE}")

    # Create a mapping from token to ID and encode the corpus
    token_to_id = {token: idx for idx, token in enumerate(VOCAB)}
    tokens = [token_to_id[token] for token in tokenizer.encode(corpus)]
    print(f"Encoded {len(tokens):,} tokens")

    # Create the dataset and dataloader
    dataset = CustomShakespeareDataset(tokens, seq_len=config["training_seq_len"])

    train_set = torch.utils.data.Subset(dataset, range(0, int(0.9 * len(dataset))))
    val_set = torch.utils.data.Subset(
        dataset, range(int(0.9 * len(dataset)), len(dataset))
    )

    dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=pin_memory,
    )

    # Create the model
    # Note that the embedding class should create nested tensors in the d_model dim
    model = BaselineModel(
        vocab_size=VOCAB_SIZE,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        layers=config["layers"],
    )

    # log model parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Create the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = CrossEntropyLoss()

    return dataloader, val_dataloader, model, optimizer, loss_fn


def evaluate(
    model: BaselineModel,
    loader: torch.utils.data.DataLoader,
    loss_fn: CrossEntropyLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_tokens = 0
    total_correct = 0
    total_loss = 0.0
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for input_seq, target_seq in loader:
            input_seq = input_seq.to(device, non_blocking=non_blocking)
            target_seq = target_seq.to(device, non_blocking=non_blocking)

            output = model(input_seq)
            loss = loss_fn(
                output.reshape(-1, output.size(-1)),
                target_seq.reshape(-1),
            )

            tokens = target_seq.numel()
            total_tokens += tokens
            total_loss += loss.item() * tokens
            total_correct += (output.argmax(dim=-1) == target_seq).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    avg_accuracy = total_correct / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "val_loss": avg_loss,
        "val_token_accuracy": avg_accuracy,
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
        checkpoint_model.load_state_dict(checkpoint["model_state_dict"])
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

        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            input_seq = input_seq.to(device, non_blocking=non_blocking)
            target_seq = target_seq.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            output = model(input_seq)
            loss = loss_fn(
                output.reshape(-1, output.size(-1)),
                target_seq.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            global_step += 1

            tokens = target_seq.numel()
            epoch_token_count += tokens
            epoch_train_loss_sum += loss.item() * tokens

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
            f"val_token_accuracy={val_metrics['val_token_accuracy']:.4f}"
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

if __name__ == "__main__":
    model_pipeline(
        config,
        project_name="transformer-room-baseline",
    )
