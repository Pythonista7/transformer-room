from config import Config
from train import model_pipeline

if __name__ == "__main__":

    config: Config = {
        "base_vocab_size": 10_000,
        "num_special_tokens": 2,  # EOS and PAD
        "vocab_size": 10_002,
        "d_model": 128,
        "n_heads": 8,
        "layers": 2,
        "learning_rate": 0.001,
        "epochs": 3,
        "training_seq_len": 128,
        "training_stride": 128,
        "data_fraction": 1,
        "batch_size": 256,
        "checkpoint_every_n_steps": 250,
        "checkpoint_path": "baseline_checkpoint.pt",
        "final_model_path": "baseline_model.pt",
        "dataset_source": "local",
        "dataset_path": "../datasets/tiny_shakespeare.txt",
        # Hugging Face dataset config example:
        # "dataset_source": "huggingface",
        # "hf_dataset_name": "roneneldan/TinyStories",
        # "hf_dataset_split": "train",
        # "hf_text_field": "text",
        # "hf_max_rows": 50_000,
        "tokenizer_vocab_path": "tiny_shakespeare_bpe_vocab.txt",
        "resume_from_checkpoint": True,
        "use_torch_compile": True,
        "torch_compile_mode": "default",
        "torch_compile_fullgraph": False,
        "torch_compile_dynamic": False,
    }
    model_pipeline(
        config,
        project_name="transformer-room-baseline",
    )
