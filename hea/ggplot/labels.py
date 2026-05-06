"""Label stubs. Real :func:`labs`, :func:`xlab`, :func:`ylab`, :func:`ggtitle`
land in Phase 6.1."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Labels:
    labels: dict = field(default_factory=dict)
