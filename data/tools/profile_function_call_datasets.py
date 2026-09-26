import ast
import json
import math
import re
from collections import Counter
from json import JSONDecodeError
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

XLAM_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xlam"
    / "xlam_function_calling_60k.json"
)

HERMES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hermes"
    / "func-calling-singleturn.json"
)

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "dataset_profile.json"
ERROR_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "dataset_profile_errors.jsonl"

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.DOTALL,
)


def iter_json_array(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> Iterator[Any]:
    """
    流式读取最外层为JSON数组的大文件。

    不使用json.load一次性把整个文件加载进内存。
    """
    decoder = json.JSONDecoder()

    with path.open("r", encoding="utf-8-sig") as file:
        buffer = ""
        array_started = False
        reached_end = False

        while True:
            chunk = file.read(chunk_size)

            if chunk:
                buffer += chunk
            else:
                reached_end = True

            position = 0

            while True:
                # 跳过空白
                while (
                    position < len(buffer)
                    and buffer[position].isspace()
                ):
                    position += 1

                if not array_started:
                    if position >= len(buffer):
                        break

                    if buffer[position] != "[":
                        raise ValueError(
                            f"{path}最外层不是JSON数组"
                        )

                    array_started = True
                    position += 1
                    continue

                # 跳过数组元素之间的逗号和空白
                while position < len(buffer):
                    char = buffer[position]

                    if char.isspace() or char == ",":
                        position += 1
                    else:
                        break

                if position >= len(buffer):
                    break

                if buffer[position] == "]":
                    return

                try:
                    value, end_position = decoder.raw_decode(
                        buffer,
                        position,
                    )
                except JSONDecodeError:
                    # 当前样本可能还没读完整，保留剩余内容，
                    # 下一次读取更多文件内容后继续解析。
                    break

                yield value
                position = end_position

            buffer = buffer[position:]

            if reached_end:
                remaining = buffer.strip()

                if remaining in {"", "]"}:
                    return

                raise ValueError(
                    f"{path}末尾存在无法解析的内容："
                    f"{remaining[:200]!r}"
                )


def parse_nested_json(value: Any) -> Any:
    """
    xLAM和Hermes的部分字段虽然语义上是list/dict，
    但原文件中保存成了JSON字符串。
    """
    if not isinstance(value, str):
        return value

    value = value.strip()

    if not value.startswith(("[", "{")):
        return value

    return json.loads(value)


def percentile(values: list[int], ratio: float) -> int | None:
    """计算离散数据的最近秩百分位数。"""
    if not values:
        return None

    sorted_values = sorted(values)
    index = max(
        0,
        math.ceil(len(sorted_values) * ratio) - 1,
    )

    return sorted_values[index]


def summarize_numbers(values: list[int]) -> dict[str, Any]:
    """生成一组整数的描述性统计。"""
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "mean": round(mean(values), 3),
        "median": median(values),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def add_error(
    errors: list[dict[str, Any]],
    dataset: str,
    sample_id: Any,
    error_type: str,
    detail: str,
    max_errors: int = 200,
) -> None:
    """只保存少量错误示例，防止错误文件过大。"""
    if len(errors) >= max_errors:
        return

    errors.append(
        {
            "dataset": dataset,
            "sample_id": sample_id,
            "error_type": error_type,
            "detail": detail,
        }
    )


def profile_xlam(
    path: Path,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    total_samples = 0

    tools_per_sample: list[int] = []
    calls_per_sample: list[int] = []
    query_lengths: list[int] = []
    parameters_per_tool: list[int] = []

    unique_tool_names: set[str] = set()
    seen_queries: set[str] = set()

    duplicate_queries = 0
    invalid_tools_json = 0
    invalid_answers_json = 0
    invalid_tool_structures = 0
    invalid_answer_structures = 0

    samples_with_one_candidate_tool = 0
    samples_with_multiple_candidate_tools = 0

    samples_with_no_calls = 0
    samples_with_one_call = 0
    samples_with_multiple_calls = 0

    same_tool_parallel_samples = 0
    multiple_different_tool_samples = 0

    samples_using_unavailable_tool = 0

    total_parameters = 0
    parameters_with_default = 0
    parameter_type_counter: Counter[str] = Counter()

    for sample in iter_json_array(path):
        total_samples += 1
        sample_id = sample.get("id", total_samples - 1)

        query = sample.get("query", "")

        if not isinstance(query, str):
            query = str(query)

        query_lengths.append(len(query))

        if query in seen_queries:
            duplicate_queries += 1
        else:
            seen_queries.add(query)

        try:
            tools = parse_nested_json(sample.get("tools", []))
        except Exception as exc:
            invalid_tools_json += 1
            add_error(
                errors,
                "xlam",
                sample_id,
                "invalid_tools_json",
                str(exc),
            )
            tools = []

        try:
            answers = parse_nested_json(sample.get("answers", []))
        except Exception as exc:
            invalid_answers_json += 1
            add_error(
                errors,
                "xlam",
                sample_id,
                "invalid_answers_json",
                str(exc),
            )
            answers = []

        if not isinstance(tools, list):
            invalid_tool_structures += 1
            tools = []

        if not isinstance(answers, list):
            invalid_answer_structures += 1
            answers = []

        tool_count = len(tools)
        call_count = len(answers)

        tools_per_sample.append(tool_count)
        calls_per_sample.append(call_count)

        if tool_count == 1:
            samples_with_one_candidate_tool += 1
        elif tool_count > 1:
            samples_with_multiple_candidate_tools += 1

        if call_count == 0:
            samples_with_no_calls += 1
        elif call_count == 1:
            samples_with_one_call += 1
        else:
            samples_with_multiple_calls += 1

        available_tool_names: set[str] = set()

        for tool in tools:
            if not isinstance(tool, dict):
                invalid_tool_structures += 1
                continue

            tool_name = tool.get("name")

            if isinstance(tool_name, str):
                available_tool_names.add(tool_name)
                unique_tool_names.add(tool_name)

            parameters = tool.get("parameters", {})

            if not isinstance(parameters, dict):
                invalid_tool_structures += 1
                continue

            parameters_per_tool.append(len(parameters))

            for parameter_schema in parameters.values():
                total_parameters += 1

                if not isinstance(parameter_schema, dict):
                    continue

                if "default" in parameter_schema:
                    parameters_with_default += 1

                parameter_type = parameter_schema.get(
                    "type",
                    "<missing>",
                )
                parameter_type_counter[str(parameter_type)] += 1

        called_tool_names: list[str] = []
        unavailable_tool_found = False

        for answer in answers:
            if not isinstance(answer, dict):
                invalid_answer_structures += 1
                continue

            call_name = answer.get("name")

            if not isinstance(call_name, str):
                invalid_answer_structures += 1
                continue

            called_tool_names.append(call_name)

            if call_name not in available_tool_names:
                unavailable_tool_found = True
                add_error(
                    errors,
                    "xlam",
                    sample_id,
                    "unavailable_tool_call",
                    (
                        f"调用了{call_name!r}，"
                        f"候选工具为{sorted(available_tool_names)}"
                    ),
                )

            arguments = answer.get("arguments")

            if not isinstance(arguments, dict):
                invalid_answer_structures += 1

        if unavailable_tool_found:
            samples_using_unavailable_tool += 1

        unique_called_names = set(called_tool_names)

        if call_count > 1 and len(unique_called_names) == 1:
            same_tool_parallel_samples += 1

        if len(unique_called_names) > 1:
            multiple_different_tool_samples += 1

    return {
        "file": str(path),
        "total_samples": total_samples,
        "unique_tool_names": len(unique_tool_names),
        "duplicate_queries": duplicate_queries,
        "tools_per_sample": summarize_numbers(
            tools_per_sample
        ),
        "calls_per_sample": summarize_numbers(
            calls_per_sample
        ),
        "query_length_characters": summarize_numbers(
            query_lengths
        ),
        "parameters_per_tool": summarize_numbers(
            parameters_per_tool
        ),
        "candidate_tool_counts": {
            "one_tool": samples_with_one_candidate_tool,
            "multiple_tools": (
                samples_with_multiple_candidate_tools
            ),
        },
        "call_counts": {
            "zero_calls": samples_with_no_calls,
            "one_call": samples_with_one_call,
            "multiple_calls": samples_with_multiple_calls,
        },
        "multi_call_types": {
            "same_tool_parallel": (
                same_tool_parallel_samples
            ),
            "multiple_different_tools": (
                multiple_different_tool_samples
            ),
        },
        "parameter_summary": {
            "total_parameters": total_parameters,
            "parameters_with_default": (
                parameters_with_default
            ),
            "type_counts": dict(
                parameter_type_counter.most_common()
            ),
        },
        "quality_checks": {
            "invalid_tools_json": invalid_tools_json,
            "invalid_answers_json": invalid_answers_json,
            "invalid_tool_structures": (
                invalid_tool_structures
            ),
            "invalid_answer_structures": (
                invalid_answer_structures
            ),
            "samples_using_unavailable_tool": (
                samples_using_unavailable_tool
            ),
        },
    }

def parse_single_tool_call(
    content: str,
) -> tuple[dict[str, Any] | None, str]:
    """
    解析一段<tool_call>标签内部的内容。

    返回：
    - 解析后的dict，失败时返回None
    - 解析方式：strict_json / escaped_boundary_json /
      python_literal / failed
    """
    content = content.strip()
    boundary_escape_repaired = False

    # 部分Hermes样本在标签正文边界保存的是字面量"\\n"，
    # 而不是真实换行。只清理边界，避免破坏参数值中的合法转义。
    escaped_whitespace = (r"\r", r"\n", r"\t")

    while True:
        original = content

        for token in escaped_whitespace:
            if content.startswith(token):
                content = content[len(token):].lstrip()

            if content.endswith(token):
                content = content[:-len(token)].rstrip()

        if content == original:
            break

        boundary_escape_repaired = True

    # 优先使用标准JSON
    try:
        result = json.loads(content)

        if isinstance(result, dict):
            if boundary_escape_repaired:
                return result, "escaped_boundary_json"

            return result, "strict_json"

        return None, "failed"

    except JSONDecodeError:
        pass

    # 抢救Python单引号格式
    try:
        result = ast.literal_eval(content)

        if not isinstance(result, dict):
            return None, "failed"

        # 再经过一次JSON序列化，转成标准JSON可表示的数据结构
        normalized = json.loads(
            json.dumps(
                result,
                ensure_ascii=False,
            )
        )

        if isinstance(normalized, dict):
            return normalized, "python_literal"

    except (
        ValueError,
        SyntaxError,
        TypeError,
        JSONDecodeError,
    ):
        pass

    return None, "failed"

def parse_hermes_tool_calls(
    assistant_text: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
    str,
]:
    """
    从Hermes的<tool_call>...</tool_call>中提取调用。

    解析顺序：
    1. 优先使用json.loads解析严格JSON；
    2. 失败后使用ast.literal_eval解析Python字面量；
    3. 两种方式都失败时，记录为failed。

    返回：
    - 成功解析并规范化后的调用列表；
    - 各种解析方式的数量统计；
    - 删除tool_call标签后剩余的普通文本。
    """
    matches = TOOL_CALL_PATTERN.findall(assistant_text)

    calls: list[dict[str, Any]] = []

    parse_stats = {
        "total_tags": len(matches),
        "strict_json": 0,
        "escaped_boundary_json": 0,
        "python_literal": 0,
        "failed": 0,
    }

    for content in matches:
        parsed_call, parse_mode = parse_single_tool_call(
            content
        )

        if parsed_call is None:
            parse_stats["failed"] += 1
            continue

        calls.append(parsed_call)
        parse_stats[parse_mode] += 1

    remaining_text = TOOL_CALL_PATTERN.sub(
        "",
        assistant_text,
    ).strip()

    return calls, parse_stats, remaining_text


def normalize_hermes_tool_call_structure(
    call: dict[str, Any],
    available_tool_names: set[str],
) -> tuple[dict[str, Any], str]:
    """检查工具调用结构，并修复可由候选工具确认的嵌套名称。"""
    call_name = call.get("name")
    arguments = call.get("arguments")

    if isinstance(call_name, str) and isinstance(arguments, dict):
        return call, "normal"

    if isinstance(arguments, dict):
        nested_name = arguments.get("name")

        if isinstance(nested_name, str):
            if nested_name in available_tool_names:
                normalized_arguments = dict(arguments)
                normalized_arguments.pop("name")

                normalized_call = dict(call)
                normalized_call["name"] = nested_name
                normalized_call["arguments"] = normalized_arguments

                return normalized_call, "nested_name_repaired"

            if not available_tool_names:
                return call, "nested_name_unverifiable"

            return call, "nested_name_not_in_candidates"

    return call, "invalid"


def profile_hermes_singleturn(
    path: Path,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    total_samples = 0

    tools_per_sample: list[int] = []
    calls_per_sample: list[int] = []
    query_lengths: list[int] = []
    parameters_per_tool: list[int] = []

    unique_tool_names: set[str] = set()
    seen_queries: set[str] = set()

    role_sequence_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    subcategory_counter: Counter[str] = Counter()

    duplicate_queries = 0
    invalid_tools_json = 0
    invalid_tool_structures = 0
    invalid_conversation_structures = 0

    samples_with_no_calls = 0
    samples_with_one_call = 0
    samples_with_multiple_calls = 0

    same_tool_parallel_samples = 0
    multiple_different_tool_samples = 0

    strict_json_tool_calls = 0
    escaped_boundary_json_repaired_tool_calls = 0
    python_literal_repaired_tool_calls = 0
    failed_tool_call_tags = 0

    samples_without_tool_call_tags = 0
    samples_with_parse_failures = 0
    samples_with_only_failed_calls = 0
    samples_with_non_tool_assistant_text = 0
    samples_using_unavailable_tool = 0
    samples_with_nonstandard_schema = 0

    normal_tool_call_structures = 0
    repaired_nested_tool_call_names = 0
    unverifiable_nested_tool_call_names = 0
    invalid_tool_call_structures = 0

    for sample in iter_json_array(path):
        total_samples += 1
        sample_id = sample.get("id", total_samples - 1)

        category_counter[str(sample.get("category", "<missing>"))] += 1
        subcategory_counter[
            str(sample.get("subcategory", "<missing>"))
        ] += 1

        try:
            tools = parse_nested_json(sample.get("tools", []))
        except Exception as exc:
            invalid_tools_json += 1
            add_error(
                errors,
                "hermes_singleturn",
                sample_id,
                "invalid_tools_json",
                str(exc),
            )
            tools = []

        if not isinstance(tools, list):
            invalid_tool_structures += 1
            tools = []

        tools_per_sample.append(len(tools))

        available_tool_names: set[str] = set()
        schema_invalid_for_sample = False

        for tool_wrapper in tools:
            if not isinstance(tool_wrapper, dict):
                invalid_tool_structures += 1
                continue

            if tool_wrapper.get("type") != "function":
                schema_invalid_for_sample = True

            function = tool_wrapper.get("function")

            if not isinstance(function, dict):
                invalid_tool_structures += 1
                continue

            tool_name = function.get("name")

            if isinstance(tool_name, str):
                available_tool_names.add(tool_name)
                unique_tool_names.add(tool_name)

            parameters = function.get("parameters")

            if not isinstance(parameters, dict):
                schema_invalid_for_sample = True
                continue

            if parameters.get("type") != "object":
                schema_invalid_for_sample = True

            properties = parameters.get("properties", {})
            required = parameters.get("required", [])

            if not isinstance(properties, dict):
                schema_invalid_for_sample = True
                properties = {}

            if not isinstance(required, list):
                schema_invalid_for_sample = True

            parameters_per_tool.append(len(properties))

        if schema_invalid_for_sample:
            samples_with_nonstandard_schema += 1

        conversations = sample.get("conversations", [])

        if not isinstance(conversations, list):
            invalid_conversation_structures += 1
            conversations = []

        roles = []
        human_messages: list[str] = []
        assistant_messages: list[str] = []

        for message in conversations:
            if not isinstance(message, dict):
                invalid_conversation_structures += 1
                continue

            role = message.get("from")
            value = message.get("value")

            roles.append(str(role))

            if not isinstance(value, str):
                invalid_conversation_structures += 1
                continue

            if role == "human":
                human_messages.append(value)
            elif role == "gpt":
                assistant_messages.append(value)

        role_sequence_counter[" -> ".join(roles)] += 1

        query = "\n".join(human_messages)

        query_lengths.append(len(query))

        if query in seen_queries:
            duplicate_queries += 1
        else:
            seen_queries.add(query)

        all_calls: list[dict[str, Any]] = []
        non_tool_text_found = False

        sample_total_tool_call_tags = 0
        sample_failed_tool_call_tags = 0

        for assistant_text in assistant_messages:
            calls, parse_stats, remaining_text = (
                parse_hermes_tool_calls(assistant_text)
            )

            normalized_calls: list[dict[str, Any]] = []

            for call in calls:
                normalized_call, structure_mode = (
                    normalize_hermes_tool_call_structure(
                        call,
                        available_tool_names,
                    )
                )

                if structure_mode == "normal":
                    normal_tool_call_structures += 1
                elif structure_mode == "nested_name_repaired":
                    repaired_nested_tool_call_names += 1
                elif structure_mode == "nested_name_unverifiable":
                    unverifiable_nested_tool_call_names += 1
                    add_error(
                        errors,
                        "hermes_singleturn",
                        sample_id,
                        "unverifiable_nested_tool_name",
                        json.dumps(call, ensure_ascii=False)[:500],
                    )
                else:
                    invalid_tool_call_structures += 1
                    add_error(
                        errors,
                        "hermes_singleturn",
                        sample_id,
                        "invalid_tool_call_structure",
                        json.dumps(call, ensure_ascii=False)[:500],
                    )

                normalized_calls.append(normalized_call)

            all_calls.extend(normalized_calls)

            strict_json_tool_calls += parse_stats["strict_json"]
            escaped_boundary_json_repaired_tool_calls += (
                parse_stats["escaped_boundary_json"]
            )
            python_literal_repaired_tool_calls += (
                parse_stats["python_literal"]
            )
            failed_tool_call_tags += parse_stats["failed"]

            sample_total_tool_call_tags += (
                parse_stats["total_tags"]
            )
            sample_failed_tool_call_tags += (
                parse_stats["failed"]
            )

            if parse_stats["failed"] > 0:
                add_error(
                    errors,
                    "hermes_singleturn",
                    sample_id,
                    "tool_call_parse_error",
                    assistant_text[:500],
                )

            if remaining_text:
                non_tool_text_found = True

        if sample_total_tool_call_tags == 0:
            samples_without_tool_call_tags += 1

        if sample_failed_tool_call_tags > 0:
            samples_with_parse_failures += 1

        if (
            sample_total_tool_call_tags > 0
            and not all_calls
        ):
            samples_with_only_failed_calls += 1

        if non_tool_text_found:
            samples_with_non_tool_assistant_text += 1

        call_count = len(all_calls)
        calls_per_sample.append(call_count)

        if sample_total_tool_call_tags == 0:
            # assistant中确实没有出现<tool_call>标签
            samples_with_no_calls += 1

        elif call_count == 1:
            samples_with_one_call += 1

        elif call_count > 1:
            samples_with_multiple_calls += 1

        called_tool_names: list[str] = []
        unavailable_tool_found = False

        for call in all_calls:
            call_name = call.get("name")

            if not isinstance(call_name, str):
                continue

            called_tool_names.append(call_name)

            if call_name not in available_tool_names:
                unavailable_tool_found = True
                add_error(
                    errors,
                    "hermes_singleturn",
                    sample_id,
                    "unavailable_tool_call",
                    (
                        f"调用了{call_name!r}，"
                        f"候选工具为{sorted(available_tool_names)}"
                    ),
                )

        if unavailable_tool_found:
            samples_using_unavailable_tool += 1

        unique_called_names = set(called_tool_names)

        if call_count > 1 and len(unique_called_names) == 1:
            same_tool_parallel_samples += 1

        if len(unique_called_names) > 1:
            multiple_different_tool_samples += 1

    return {
        "file": str(path),
        "total_samples": total_samples,
        "unique_tool_names": len(unique_tool_names),
        "duplicate_queries": duplicate_queries,
        "tools_per_sample": summarize_numbers(
            tools_per_sample
        ),
        "calls_per_sample": summarize_numbers(
            calls_per_sample
        ),
        "query_length_characters": summarize_numbers(
            query_lengths
        ),
        "parameters_per_tool": summarize_numbers(
            parameters_per_tool
        ),
        "call_counts": {
            "no_tool_call_tags": samples_with_no_calls,
            "one_parsed_call": samples_with_one_call,
            "multiple_parsed_calls": samples_with_multiple_calls,
            "samples_with_only_failed_calls": (
                samples_with_only_failed_calls
            ),
        },
        "multi_call_types": {
            "same_tool_parallel": (
                same_tool_parallel_samples
            ),
            "multiple_different_tools": (
                multiple_different_tool_samples
            ),
        },
        "conversation_summary": {
            "role_sequences": dict(
                role_sequence_counter.most_common()
            ),
            "samples_with_non_tool_assistant_text": (
                samples_with_non_tool_assistant_text
            ),
        },
        "category_summary": {
            "unique_categories": len(category_counter),
            "unique_subcategories": len(
                subcategory_counter
            ),
            "top_10_categories": dict(
                category_counter.most_common(10)
            ),
            "top_10_subcategories": dict(
                subcategory_counter.most_common(10)
            ),
        },
        "quality_checks": {
            "invalid_tools_json": invalid_tools_json,
            "invalid_tool_structures": (
                invalid_tool_structures
            ),
            "invalid_conversation_structures": (
                invalid_conversation_structures
            ),

            "strict_json_tool_calls": (
                strict_json_tool_calls
            ),
            "escaped_boundary_json_repaired_tool_calls": (
                escaped_boundary_json_repaired_tool_calls
            ),
            "python_literal_repaired_tool_calls": (
                python_literal_repaired_tool_calls
            ),
            "failed_tool_call_tags": (
                failed_tool_call_tags
            ),
            "samples_with_parse_failures": (
                samples_with_parse_failures
            ),
            "samples_with_only_failed_calls": (
                samples_with_only_failed_calls
            ),
            "samples_without_tool_call_tags": (
                samples_without_tool_call_tags
            ),

            "samples_using_unavailable_tool": (
                samples_using_unavailable_tool
            ),
            "samples_with_nonstandard_schema": (
                samples_with_nonstandard_schema
            ),
            "normal_tool_call_structures": (
                normal_tool_call_structures
            ),
            "repaired_nested_tool_call_names": (
                repaired_nested_tool_call_names
            ),
            "unverifiable_nested_tool_call_names": (
                unverifiable_nested_tool_call_names
            ),
            "invalid_tool_call_structures": (
                invalid_tool_call_structures
            ),
        },
    }


def print_number_summary(
    name: str,
    summary: dict[str, Any],
) -> None:
    print(
        f"{name}: "
        f"min={summary['min']}, "
        f"mean={summary['mean']}, "
        f"median={summary['median']}, "
        f"p90={summary['p90']}, "
        f"max={summary['max']}"
    )


def print_xlam_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("xLAM FUNCTION CALLING 60K")
    print("=" * 80)

    print(f"样本总数：{report['total_samples']}")
    print(f"去重工具名数量：{report['unique_tool_names']}")
    print(f"完全重复query数量：{report['duplicate_queries']}")

    print_number_summary(
        "每条样本候选工具数",
        report["tools_per_sample"],
    )
    print_number_summary(
        "每条样本标准调用数",
        report["calls_per_sample"],
    )
    print_number_summary(
        "query字符长度",
        report["query_length_characters"],
    )
    print_number_summary(
        "每个工具参数数量",
        report["parameters_per_tool"],
    )

    print("\n候选工具数量：")
    print(json.dumps(
        report["candidate_tool_counts"],
        ensure_ascii=False,
        indent=2,
    ))

    print("\n调用数量：")
    print(json.dumps(
        report["call_counts"],
        ensure_ascii=False,
        indent=2,
    ))

    print("\n多调用类型：")
    print(json.dumps(
        report["multi_call_types"],
        ensure_ascii=False,
        indent=2,
    ))

    print("\n数据质量检查：")
    print(json.dumps(
        report["quality_checks"],
        ensure_ascii=False,
        indent=2,
    ))


def print_hermes_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("HERMES FUNCTION CALLING SINGLE-TURN")
    print("=" * 80)

    print(f"样本总数：{report['total_samples']}")
    print(f"去重工具名数量：{report['unique_tool_names']}")
    print(f"完全重复query数量：{report['duplicate_queries']}")

    print_number_summary(
        "每条样本候选工具数",
        report["tools_per_sample"],
    )
    print_number_summary(
        "每条样本标准调用数",
        report["calls_per_sample"],
    )
    print_number_summary(
        "query字符长度",
        report["query_length_characters"],
    )
    print_number_summary(
        "每个工具参数数量",
        report["parameters_per_tool"],
    )

    print("\n调用数量：")
    print(json.dumps(
        report["call_counts"],
        ensure_ascii=False,
        indent=2,
    ))

    print("\n多调用类型：")
    print(json.dumps(
        report["multi_call_types"],
        ensure_ascii=False,
        indent=2,
    ))

    print("\n角色序列：")
    print(json.dumps(
        report["conversation_summary"]["role_sequences"],
        ensure_ascii=False,
        indent=2,
    ))

    print("\n数据质量检查：")
    print(json.dumps(
        report["quality_checks"],
        ensure_ascii=False,
        indent=2,
    ))

    print("\n数量最多的10个类别：")
    print(json.dumps(
        report["category_summary"]["top_10_categories"],
        ensure_ascii=False,
        indent=2,
    ))


def main() -> None:
    test_assistant_text = r"""
<tool_call>\n{"arguments": {"queries": ['question 1', 'question 2'], "name": "ExpertQAExtractor"}}\n</tool_call>
"""

    test_calls, test_stats, test_remaining = (
        parse_hermes_tool_calls(test_assistant_text)
    )

    if (
        test_stats["total_tags"] != 1
        or test_stats["python_literal"] != 1
        or test_stats["failed"] != 0
        or len(test_calls) != 1
        or test_remaining
    ):
        raise RuntimeError(
            "Hermes完整解析链路自检失败："
            f"stats={test_stats}, "
            f"calls={test_calls}, "
            f"remaining={test_remaining!r}"
        )

    test_call, test_structure_mode = (
        normalize_hermes_tool_call_structure(
            test_calls[0],
            {"ExpertQAExtractor"},
        )
    )

    if (
        test_structure_mode != "nested_name_repaired"
        or test_call.get("name") != "ExpertQAExtractor"
        or "name" in test_call.get("arguments", {})
    ):
        raise RuntimeError(
            "Hermes工具调用结构修复自检失败："
            f"mode={test_structure_mode}, call={test_call}"
        )

    print(
        "Hermes完整解析与结构修复链路自检通过："
        f"{test_stats}"
    )


    for path in (XLAM_PATH, HERMES_PATH):
        if not path.exists():
            raise FileNotFoundError(f"找不到数据文件：{path}")

    errors: list[dict[str, Any]] = []

    print("正在统计xLAM……")
    xlam_report = profile_xlam(XLAM_PATH, errors)

    print("正在统计Hermes single-turn……")
    hermes_report = profile_hermes_singleturn(
        HERMES_PATH,
        errors,
    )

    report = {
        "xlam": xlam_report,
        "hermes_singleturn": hermes_report,
        "saved_error_examples": len(errors),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with ERROR_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for error in errors:
            file.write(
                json.dumps(
                    error,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print_xlam_report(xlam_report)
    print_hermes_report(hermes_report)

    print("\n" + "=" * 80)
    print(f"完整统计已保存：{OUTPUT_PATH}")
    print(f"错误示例已保存：{ERROR_OUTPUT_PATH}")
    print(f"保存的错误示例数量：{len(errors)}")


if __name__ == "__main__":
    main()
