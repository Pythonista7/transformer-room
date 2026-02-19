import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

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

config: Config = {
    "vocab_size": 10_000,
    "d_model": 128,
    "n_heads": 8,
    "layers": 2,
    "learning_rate": 1e-3,
    "epochs": 10,
    "training_seq_len": 128,
    "batch_size": 256
}

#===

corpus = None
with open("../datasets/tiny_shakespeare.txt") as f:
    corpus = f.read()

if corpus is None:
    raise Exception("Invalid dataset read!")

tokenizer = BPETokenizer(
    corpus=corpus, 
    max_vocab_count=config["vocab_size"],
    path="tiny_shakespeare_bpe_vocab.txt"
)

VOCAB = tokenizer.vocab
VOCAB_SIZE = len(VOCAB)
assert VOCAB_SIZE == config["vocab_size"], f"Tokenizer vocab size {VOCAB_SIZE} does not match config vocab size {config['vocab_size']}"

print(f"Vocabulary size: {VOCAB_SIZE}")

token_to_id = {token: idx for idx, token in enumerate(VOCAB)}
tokens = [token_to_id[token] for token in tokenizer.encode(corpus)]
print(f"Encoded {len(tokens):,} tokens")


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
        input_seq = self.tokens[idx:idx+self.seq_len]
        target_seq = self.tokens[idx+1:idx+self.seq_len+1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

dataset = CustomShakespeareDataset(tokens, seq_len=config["training_seq_len"])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# print a sample shape from the dataloader
for batch in dataloader:
    input_batch, target_batch = batch
    print(f"Input batch shape: {input_batch.shape}")
    print(f"Target batch shape: {target_batch.shape}")
    break

# Note that the embedding class should create nested tensors in the d_model dim
model = BaselineModel(
    vocab_size=VOCAB_SIZE,
    d_model=config["d_model"],
    n_heads=config["n_heads"],
    layers=config["layers"]
)

# log model parameter count
param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")


def train():
    
    pass




if __name__ == "__main__":
    train()