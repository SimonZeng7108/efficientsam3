"""兼容入口：`python -m fewshot_adapter.convert_datatrain`。"""

from __future__ import annotations

from .cli.convert_datatrain import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
