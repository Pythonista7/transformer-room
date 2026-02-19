# Byte-Pair Encoding (BPE)
import ast
import os
import numpy as np
import tqdm

class BPETokenizer:
    def __init__(self, max_vocab_count: int = None, corpus: str = None, path: str = None, max_itr: int = None, target_vocab_count: int = None):
        # Backward compatibility: allow old argument name `target_vocab_count`.
        if max_vocab_count is not None and target_vocab_count is not None and max_vocab_count != target_vocab_count:
            raise ValueError("Provide only one of `max_vocab_count` or `target_vocab_count` (legacy alias), or make them equal.")

        if max_vocab_count is None:
            max_vocab_count = target_vocab_count

        # Exactly one stopping criterion must be provided.
        if (max_vocab_count is None and max_itr is None) or (max_vocab_count is not None and max_itr is not None):
            raise ValueError("Provide exactly one stopping criterion: either `max_vocab_count` or `max_itr`.")

        self.max_vocab_count = max_vocab_count
        self.corpus = corpus
        self.vocab = None
        self.max_itr = max_itr
        self._encode_trie = None
        # Prefer loading from an existing vocab file.
        if path is not None and os.path.exists(path):
            self.vocab = self.load(path)
            if self.max_vocab_count is not None and len(self.vocab) != self.max_vocab_count:
                raise ValueError(f"Loaded vocab size {len(self.vocab)} does not match target {self.max_vocab_count}, consider retraining with the correct `max_vocab_count` or updating the vocab file.")
            self._build_encode_trie()
            return

        # If no vocab file exists, train from corpus and optionally persist it.
        if corpus is not None:
            self.vocab = self._train(corpus, self.max_vocab_count, self.max_itr)
            if path is not None:
                self.save(path)
            self._build_encode_trie()
            return

        raise ValueError("Vocab file not found. Provide `corpus` to train a new tokenizer.")

    def _token_to_bytes(self, token):
        if isinstance(token, int):
            return [token]
        if isinstance(token, tuple):
            out = []
            for t in token:
                out.extend(self._token_to_bytes(t))
            return out
        raise TypeError(f"Unsupported token type: {type(token)}")

    def _merge_token(self, left, right):
        return tuple(self._token_to_bytes(left) + self._token_to_bytes(right))

    def _build_encode_trie(self):
        """Build a byte-level trie for fast longest-prefix token matching during encode."""
        trie = {}
        terminal_key = "_token"

        for token in self.vocab:
            token_bytes = [token] if isinstance(token, int) else self._token_to_bytes(token)
            node = trie
            for b in token_bytes:
                if b not in node:
                    node[b] = {}
                node = node[b]
            node[terminal_key] = token

        self._encode_trie = trie

    def _pair_count(self, byte_arr):
        count = {}
        for i in range(len(byte_arr)-1):
            count[(byte_arr[i],byte_arr[i+1])] = count.get((byte_arr[i],byte_arr[i+1]) , 0) + 1
        return count
    
    def _merge(self, pair, corpus):
        new_corpus = []
        i = 0
        merged_token = self._merge_token(pair[0], pair[1])
        while i < len(corpus):
            if i+1 < len(corpus) and corpus[i] == pair[0] and corpus[i+1] == pair[1]:
                new_corpus.append(merged_token)
                i += 2
            else:
                new_corpus.append(corpus[i])
                i += 1
        return new_corpus
    
    def _train(self, text, max_vocab_count=None, max_itr=None):
        # Initial vocab = individual characters from the corpus.
        corpus = list(text.encode("utf-8")) # convert text to list of byte values.
        vocab = np.unique(corpus).tolist() # get unique byte values as initial vocab.

        total = max_vocab_count if max_vocab_count is not None else max_itr
        itr = tqdm.tqdm(total=total, desc="Training BPE Tokenizer")
        i = 0
        with itr:
            while True:
                if max_vocab_count is not None and len(vocab) >= max_vocab_count:
                    break
                if max_itr is not None and i >= max_itr:
                    break

                # get most freq pair that is adjacent
                counts = self._pair_count(corpus)
                if not counts:
                    break
                
                max_freq_pair = max(counts,key=counts.get)
                merged_token = self._merge_token(max_freq_pair[0], max_freq_pair[1])
                
                # merge the pair into one token and add to vocab
                if merged_token not in vocab:
                    vocab.append(merged_token)
                
                # update corpus
                corpus = self._merge(max_freq_pair, corpus)
                
                itr.update(1)
                if max_vocab_count is not None:
                    itr.set_description(f"iteration:{i} - VOCAB {len(vocab)}/{max_vocab_count}")
                else:
                    itr.set_description(f"iteration:{i} - ITR {i + 1}/{max_itr}")
                i+=1
                
        return vocab

    def encode(self, text):
        # Greedy longest-match encoding using a prebuilt trie.
        byte_arr = list(text.encode("utf-8"))
        tokens = []

        if self._encode_trie is None:
            self._build_encode_trie()

        trie = self._encode_trie
        terminal_key = "_token"
        i = 0
        while i < len(byte_arr):
            node = trie
            j = i
            best_match = None
            best_end = i

            while j < len(byte_arr) and byte_arr[j] in node:
                node = node[byte_arr[j]]
                j += 1
                if terminal_key in node:
                    best_match = node[terminal_key]
                    best_end = j

            if best_match is None:
                print(f"No matching token found for byte sequence starting at position {i}: {byte_arr[i:i+10]}... (consider increasing vocab size or checking corpus)")
                # use <UNK> token 
                tokens.append()
            tokens.append(best_match)
            i = best_end

        return tokens
    
    def decode(self, tokens):
        byte_arr = []
        for token in tokens:
            byte_arr.extend(self._token_to_bytes(token))
        return bytes(byte_arr).decode("utf-8")

    def save(self, path):
        with open(path, "w") as f:
            for token in self.vocab:
                f.write(f"{token}\n")
        print(f"Tokenizer vocab saved to {path}")

    def load(self, path):
        vocab = []
        with open(path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line == "":
                    continue
                token = ast.literal_eval(line)
                if not isinstance(token, (int, tuple)):
                    raise ValueError(f"Invalid token in vocab file: {line}")
                vocab.append(token)
        print(f"Tokenizer vocab loaded from {path}")
        return vocab

if __name__ == "__main__":
    lines = []
    with open("../datasets/tiny_shakespeare.txt") as f:
        lines = f.readlines()
    tk = BPETokenizer(10_000, ''.join(lines[:20]), path="tiny_shakespeare_bpe_vocab.txt")
    # print("Decoded all vocab tokens:", [tk.decode([token]) for token in tk.vocab])
    # Decode sample text
    sample_text = ''.join(lines[30:45])
    encoded = tk.encode(sample_text)
    print(f"Encoded: {encoded} , Len = {len(encoded)}")
    decoded = tk.decode(encoded)
    print(f"Decoded: {decoded}")
