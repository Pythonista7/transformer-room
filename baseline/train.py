import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb
import os

from model import BaselineModel
from tokenizer import BPETokenizer
from typing import TypedDict

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


config: Config = {
    "vocab_size": 10_000,
    "d_model": 128,
    "n_heads": 8,
    "layers": 2,
    "learning_rate": 2.5e-4,  # from the GPT-1 paper
    "epochs": 10,
    "training_seq_len": 128,
    "batch_size": 256,
    "checkpoint_every_n_steps": 250,
    "checkpoint_path": "baseline_checkpoint.pt",
    "final_model_path": "baseline_model.pt",
    "resume_from_checkpoint": True,
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


def make(
    config: Config,
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
        train_set, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False
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

def train(
    model: BaselineModel,
    loader: torch.utils.data.DataLoader,
    loss_fn: CrossEntropyLoss,
    optimizer: optim.Optimizer,
    config: Config,
    project_name: str,
):
    def save_checkpoint(epoch: int, next_batch_idx: int, global_step: int):
        checkpoint = {
            "epoch": epoch,
            "batch_idx": next_batch_idx,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
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

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint.get("epoch", 0)
        start_batch_idx = checkpoint.get("batch_idx", 0)
        global_step = checkpoint.get("global_step", 0)
        print(
            f"Resumed from {checkpoint_path} at epoch={start_epoch}, batch={start_batch_idx}, step={global_step}"
        )
        return start_epoch, start_batch_idx, global_step

    run = wandb.init(project=project_name, config=config)
    run.watch(model, loss_fn, log="all", log_freq=10)
    model.train()

    start_epoch, start_batch_idx, global_step = load_checkpoint_if_available()
    total_batches = len(loader) * config["epochs"]
    
    for epoch in tqdm(range(start_epoch, config["epochs"]), desc="Epochs"):
        for batch_idx, (input_seq, target_seq) in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            optimizer.zero_grad()
            output = model(input_seq)
            loss = loss_fn(output.view(-1, output.size(-1)), target_seq.view(-1))
            loss.backward()
            optimizer.step()
            global_step += 1

            # Log the loss to wandb
            if batch_idx % 10 == 0:
                wandb.log({"train_loss": loss.item(), "epoch": epoch + batch_idx / total_batches})

            if (
                config["checkpoint_every_n_steps"] > 0
                and global_step % config["checkpoint_every_n_steps"] == 0
            ):
                next_epoch = epoch
                next_batch_idx = batch_idx + 1
                if next_batch_idx >= len(loader):
                    next_epoch += 1
                    next_batch_idx = 0

                save_checkpoint(next_epoch, next_batch_idx, global_step)
                run.save(config["checkpoint_path"])

    # Save a final checkpoint marker at the end of training.
    save_checkpoint(config["epochs"], 0, global_step)
    run.save(config["checkpoint_path"])

    # Save final trained model weights.
    torch.save(model.state_dict(), config["final_model_path"])
    run.save(config["final_model_path"])

def test(model, test_loader, project_name):
    model.eval()
    with wandb.init(project=project_name+"_testing") as run:
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (input_seq, target_seq) in enumerate(test_loader):
                output = model(input_seq)
                
                total += target_seq.numel()
                correct += (output.argmax(dim=-1) == target_seq).sum().item()
                
                # Log the output and target sequences to wandb for qualitative analysis
                if batch_idx % 100 == 0:
                    run.log({
                        "input_seq": wandb.Table(data=input_seq.cpu().numpy()),
                        "target_seq": wandb.Table(data=target_seq.cpu().numpy()),
                        "output_seq": wandb.Table(data=output.argmax(dim=-1).cpu().numpy()),
                    })

def model_pipeline(config: Config, project_name: str):
    
    with wandb.init(project=project_name, config=config) as run:
        config = run.config
        train_loader, val_loader, model, optimizer, loss_fn = make(config)
        train(model, train_loader, loss_fn, optimizer, config, project_name)
        test(model, val_loader, project_name)
    
    return model

if __name__ == "__main__":
    model_pipeline(
        config,
        project_name="transformer-room-baseline",
    )
