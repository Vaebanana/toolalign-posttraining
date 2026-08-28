import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATA_FILES = [
    RAW_DIR / "xlam" / "xlam_function_calling_60k.json",
    RAW_DIR / "hermes" / "func-calling-singleturn.json",
    RAW_DIR / "hermes" / "func-calling.json",
    RAW_DIR / "hermes" / "glaive-function-calling-5k.json",
    RAW_DIR / "hermes" / "json-mode-agentic.json",
    RAW_DIR / "hermes" / "json-mode-singleturn.json",
]

SAMPLE_OUTPUT_DIR = RAW_DIR / "_first_samples"


def read_first_json_value(path: Path) -> tuple[Any, str]:
    """
    不把整个数据集加载进内存，只读取第一个JSON值。

    支持：
    1. JSON数组：[{}, {}, ...]
    2. JSONL：每行一个JSON对象
    3. 单个JSON对象
    """
    decoder = json.JSONDecoder()
    buffer = ""

    with path.open("r", encoding="utf-8-sig") as file:
        # 找到第一个非空字符
        while True:
            char = file.read(1)

            if char == "":
                raise ValueError("文件为空")

            if not char.isspace():
                first_char = char
                break

        if first_char == "[":
            file_format = "JSON array"
        elif first_char == "{":
            file_format = "JSON object or JSONL"
            buffer = first_char
        else:
            raise ValueError(
                f"无法识别的文件开头：{first_char!r}"
            )

        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            buffer += chunk

            candidate = buffer.lstrip()

            # JSON数组时，跳过最外层左中括号
            if first_char == "[":
                candidate = candidate.lstrip("[").lstrip()

            try:
                value, _ = decoder.raw_decode(candidate)
                return value, file_format
            except json.JSONDecodeError:
                # 第一条样本尚未读取完整，继续读取
                continue

    raise ValueError("无法解析第一条JSON数据")


def maybe_parse_nested_json(value: Any) -> Any:
    """
    某些数据集把tools、answers等内容保存成JSON字符串。
    遇到这种情况时尝试再解析一层。
    """
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if not stripped.startswith(("[", "{")):
        return value

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def describe_value(
    name: str,
    value: Any,
    indent: int = 0,
    max_depth: int = 2,
) -> None:
    """简要展示字段类型和嵌套结构。"""
    prefix = " " * indent
    parsed_value = maybe_parse_nested_json(value)

    nested_note = ""

    if parsed_value is not value:
        nested_note = "（原始值是JSON字符串，已二次解析）"

    if isinstance(parsed_value, dict):
        keys = list(parsed_value.keys())
        print(
            f"{prefix}- {name}: dict，"
            f"字段={keys}{nested_note}"
        )

        if max_depth > 0:
            for key, child in list(parsed_value.items())[:8]:
                describe_value(
                    key,
                    child,
                    indent=indent + 4,
                    max_depth=max_depth - 1,
                )

    elif isinstance(parsed_value, list):
        print(
            f"{prefix}- {name}: list，"
            f"长度={len(parsed_value)}{nested_note}"
        )

        if parsed_value and max_depth > 0:
            describe_value(
                "[0]",
                parsed_value[0],
                indent=indent + 4,
                max_depth=max_depth - 1,
            )

    elif isinstance(parsed_value, str):
        preview = parsed_value.replace("\n", "\\n")

        if len(preview) > 120:
            preview = preview[:120] + "..."

        print(f"{prefix}- {name}: str，示例={preview!r}")

    else:
        print(
            f"{prefix}- {name}: "
            f"{type(parsed_value).__name__}，值={parsed_value!r}"
        )


def save_first_sample(path: Path, sample: Any) -> Path:
    """将第一条样本单独保存，方便在VS Code中查看。"""
    SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SAMPLE_OUTPUT_DIR / f"{path.stem}_first.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            sample,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def inspect_file(path: Path) -> None:
    print("\n" + "=" * 90)
    print(f"文件：{path.relative_to(PROJECT_ROOT)}")
    print("=" * 90)

    if not path.exists():
        print("[跳过] 文件不存在")
        return

    file_size_mb = path.stat().st_size / 1024 / 1024
    print(f"文件大小：{file_size_mb:.2f} MB")

    try:
        sample, file_format = read_first_json_value(path)
    except Exception as exc:
        print(f"[失败] 无法读取第一条样本：{exc}")
        return

    print(f"检测格式：{file_format}")
    print(f"第一条样本类型：{type(sample).__name__}")

    if isinstance(sample, dict):
        print(f"顶层字段：{list(sample.keys())}")

        print("\n字段结构：")

        for key, value in sample.items():
            describe_value(
                key,
                value,
                indent=0,
                max_depth=2,
            )
    else:
        describe_value(
            "sample",
            sample,
            indent=0,
            max_depth=2,
        )

    output_path = save_first_sample(path, sample)
    print(f"\n第一条样本已保存到：{output_path}")


def main() -> None:
    for path in DATA_FILES:
        inspect_file(path)


if __name__ == "__main__":
    main()