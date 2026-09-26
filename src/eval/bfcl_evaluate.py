"""Invoke the official BFCL v4 evaluator for locally generated Qwen output.

Run this module with the Python environment in which ``bfcl_eval`` is
installed. It registers a local model label in memory and leaves the upstream
BFCL checkout/package untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.bfcl_common import (
    BFCLAdapterError,
    DEFAULT_CATEGORIES,
    DEFAULT_DISPLAY_NAME,
    DEFAULT_REGISTRY_NAME,
    DEFAULT_RESULT_DIR,
    DEFAULT_SCORE_DIR,
    official_result_path,
    parse_categories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score local Qwen3 result files with the official BFCL v4 evaluator."
    )
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--registry-name", default=DEFAULT_REGISTRY_NAME)
    parser.add_argument("--display-name", default=DEFAULT_DISPLAY_NAME)
    parser.add_argument("--partial-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = parse_categories(args.categories)
    if not args.registry_name or any(char in args.registry_name for char in "_/\\"):
        raise BFCLAdapterError(
            "registry-name must be non-empty and contain no underscore or slash"
        )
    result_dir = args.result_dir.resolve()
    score_dir = args.score_dir.resolve()
    missing_files = [
        official_result_path(result_dir, args.registry_name, category)
        for category in categories
        if not official_result_path(result_dir, args.registry_name, category).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "official BFCL result files are missing: "
            + ", ".join(str(path) for path in missing_files)
        )

    try:
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
        from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
        from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    except ImportError as error:
        raise RuntimeError(
            "bfcl_eval is not importable. Run this script with the BFCL Python "
            "environment, not the LLaMA-Factory/PyTorch environment."
        ) from error

    MODEL_CONFIG_MAPPING[args.registry_name] = ModelConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        display_name=args.display_name,
        url="local",
        org="ToolAlign",
        license="apache-2.0",
        model_handler=QwenFCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    )
    score_dir.mkdir(parents=True, exist_ok=True)
    evaluation_main(
        [args.registry_name],
        list(categories),
        str(result_dir),
        str(score_dir),
        partial_eval=args.partial_eval,
    )


if __name__ == "__main__":
    main()
