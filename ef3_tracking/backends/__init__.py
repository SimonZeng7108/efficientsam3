"""Backend adapters for the trackers.

Real backend lazily imports the SAM3 stack so unit tests can avoid pulling in
the model weights / heavy dependencies. The mock backend lives next to it for
testing and demo purposes.
"""

from .mock import MockBackend
from .sam3_backend import Sam3EdgeBackend, build_edge_backend

__all__ = ["MockBackend", "Sam3EdgeBackend", "build_edge_backend"]
