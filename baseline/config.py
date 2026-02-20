
from typing import NotRequired, TypedDict


class Config(TypedDict):
    base_vocab_size: int
    num_special_tokens: int
    vocab_size: int
    d_model: int
    n_heads: int
    layers: int
    learning_rate: float
    epochs: int
    training_seq_len: int
    training_stride: int
    data_fraction: float
    batch_size: int
    checkpoint_every_n_steps: int
    checkpoint_path: str
    final_model_path: str
    dataset_path: str
    tokenizer_vocab_path: str
    resume_from_checkpoint: bool
    run_name: NotRequired[str]
    models_root_dir: NotRequired[str]
    run_artifact_dir: NotRequired[str]
    model_diagram_path: NotRequired[str]
    use_torch_compile: NotRequired[bool]
    torch_compile_mode: NotRequired[str]
    torch_compile_fullgraph: NotRequired[bool]
    torch_compile_dynamic: NotRequired[bool]
