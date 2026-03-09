"""Built-in adapter registrations.

Importing this package registers all built-in dataset/tokenizer/model/split/logger adapters.
"""

from . import datasets  # noqa: F401
from . import tokenizers  # noqa: F401
from . import models  # noqa: F401
from . import splits  # noqa: F401
from . import loggers  # noqa: F401
