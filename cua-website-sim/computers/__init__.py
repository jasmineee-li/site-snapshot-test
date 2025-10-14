from . import default
from . import contrib
from .computer import Computer
from .config import computers_config
from .default.local_playwright import LocalPlaywrightBrowser as LocalPlaywrightComputer

__all__ = [
    "default",
    "contrib",
    "Computer",
    "computers_config",
    "LocalPlaywrightComputer",
]
