import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from model import BaselineModel
from tokenizer import BPETokenizer

corpus = None
with open("../datasets/tiny_shakespeare.txt") as f:
    corpus = f.read()

if corpus is None:
    raise Exception("Invalid dataset read!")

tokenizer = BPETokenizer(
    corpus=corpus, 
    max_vocab_count=10_000,
    path="tiny_shakespeare_bpe_vocab.txt"
)

VOCAB = tokenizer.vocab
VOCAB_SIZE = len(VOCAB)

print(f"Vocabulary size: {VOCAB_SIZE}")

tokens = tokenizer.encode(corpus)
print(f"Encoded {len(tokens):,} tokens")


class CustomShakespeareDataset(Dataset):
    """
    IMPORTANT: REMEBER TO USE JAGGED/NESTED TENSORS ARE KEY PADDING IS NOT IMPLEMENTED IN THE ATTENTION LAYER!
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

dataset = CustomShakespeareDataset(tokens, seq_len=128)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)


model = BaselineModel(
    vocab_size=VOCAB_SIZE,
    d_model=128,
    n_heads=6,
    layers=2
)

# log model parameter count
param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")


def train():
    
    pass




if __name__ == "__main__":
    train()