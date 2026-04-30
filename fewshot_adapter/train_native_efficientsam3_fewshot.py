"""兼容入口：`python -m fewshot_adapter.train_native_efficientsam3_fewshot`。"""

from __future__ import annotations

from .cli.train_native import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
