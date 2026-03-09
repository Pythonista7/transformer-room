"""Built-in adapter registration utilities."""


def register_builtin_adapters() -> None:
    """Register all built-in adapters with the runtime registry."""
    from .datasets import register_dataset_adapters
    from .loggers import register_logger_adapters
    from .models import register_model_adapters
    from .splits import register_split_adapters
    from .tokenizers import register_tokenizer_adapters

    register_dataset_adapters()
    register_tokenizer_adapters()
    register_model_adapters()
    register_split_adapters()
    register_logger_adapters()


__all__ = ["register_builtin_adapters"]
