"""将 xLAM 与 repaired Hermes 适配为统一、模型无关的 canonical schema。

本模块只改变数据表示，不做数据修复、质量判断、样本筛选或训练协议转换。
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, Iterable


UNIFIED_SYSTEM_PROMPT = """You are a tool-calling assistant.
Use only the provided tools.
Ensure that all arguments conform to the provided tool schemas."""

TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "List": "array",
    "array": "array",
    "dict": "object",
    "Dict": "object",
    "object": "object",
}


class NormalizationError(ValueError):
    """单条样本无法无歧义转换为 canonical schema。"""


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NormalizationError(f"{field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise NormalizationError(f"{field} must be an array")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise NormalizationError(f"{field} must be a string")
    return value


def _canonical_sample_id(source: str, raw_id: Any) -> str:
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
        raise NormalizationError("sample id must be a string or integer")
    return f"{source}_{raw_id}"


def _strip_xlam_type_qualifiers(type_name: str) -> str:
    """去除 xLAM 写在 type 文本中的 optional/default 限定。"""
    normalized = re.sub(
        r",\s*optional\b",
        "",
        type_name,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r",\s*default(?:\s*=\s*|\s+).*$",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized.strip()


def _split_generic_arguments(value: str) -> list[str]:
    arguments: list[str] = []
    start = 0
    depth = 0

    for index, character in enumerate(value):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth < 0:
                raise NormalizationError(
                    f"malformed xLAM parameter type: {value}"
                )
        elif character == "," and depth == 0:
            arguments.append(value[start:index].strip())
            start = index + 1

    if depth != 0:
        raise NormalizationError(
            f"malformed xLAM parameter type: {value}"
        )

    arguments.append(value[start:].strip())
    return arguments


def normalize_xlam_type(type_name: Any) -> dict[str, Any]:
    """把数据中实际出现的 xLAM/Python 风格类型映射到 JSON Schema。"""
    if not isinstance(type_name, str) or not type_name.strip():
        raise NormalizationError(
            f"unsupported parameter type: {type_name!r}"
        )

    normalized = _strip_xlam_type_qualifiers(type_name)

    if normalized in TYPE_MAP:
        return {"type": TYPE_MAP[normalized]}

    if normalized == "set":
        return {"type": "array", "uniqueItems": True}

    # xLAM 将 callable argument 序列化成字符串（如 lambda expression）。
    if normalized == "Callable[[float], float]":
        return {"type": "string"}

    if normalized.startswith("List[") and normalized.endswith("]"):
        item_type = normalized[5:-1].strip()
        if item_type == "Union[int, float]":
            item_schema = {"type": "number"}
        else:
            item_schema = normalize_xlam_type(item_type)
        return {"type": "array", "items": item_schema}

    if normalized.startswith("Tuple[") and normalized.endswith("]"):
        item_types = _split_generic_arguments(normalized[6:-1])
        item_schemas = [normalize_xlam_type(item) for item in item_types]

        if not item_schemas or any(
            schema != item_schemas[0] for schema in item_schemas[1:]
        ):
            raise NormalizationError(
                f"unsupported parameter type: {type_name}"
            )

        return {
            "type": "array",
            "items": item_schemas[0],
            "minItems": len(item_schemas),
            "maxItems": len(item_schemas),
        }

    raise NormalizationError(
        f"unsupported parameter type: {type_name}"
    )


def normalize_xlam_parameters(parameters: Any) -> dict[str, Any]:
    """把 xLAM 参数字典转换成标准 JSON Schema object。"""
    raw_parameters = _require_dict(parameters, "tool.parameters")
    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter_name, raw_parameter in raw_parameters.items():
        name = _require_string(parameter_name, "parameter name")
        parameter = _require_dict(
            raw_parameter,
            f"tool.parameters.{name}",
        )
        raw_type = parameter.get("type")
        property_schema = normalize_xlam_type(raw_type)

        if "description" in parameter:
            property_schema["description"] = _require_string(
                parameter["description"],
                f"tool.parameters.{name}.description",
            )
        if "default" in parameter:
            property_schema["default"] = copy.deepcopy(
                parameter["default"]
            )

        properties[name] = property_schema
        is_optional = (
            isinstance(raw_type, str)
            and re.search(r",\s*optional\b", raw_type, re.IGNORECASE)
            is not None
        )
        if not is_optional and "default" not in parameter:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def normalize_xlam_tool(tool: Any) -> dict[str, Any]:
    raw_tool = _require_dict(tool, "tool")
    name = _require_string(raw_tool.get("name"), "tool.name")
    description = _require_string(
        raw_tool.get("description"),
        f"tool[{name}].description",
    )

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": normalize_xlam_parameters(
                raw_tool.get("parameters")
            ),
        },
    }


def normalize_xlam_tool_call(answer: Any) -> dict[str, Any]:
    raw_answer = _require_dict(answer, "answer")
    name = _require_string(raw_answer.get("name"), "answer.name")
    arguments = _require_dict(
        raw_answer.get("arguments"),
        f"answer[{name}].arguments",
    )

    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": copy.deepcopy(arguments),
        },
    }


def normalize_xlam_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """将一条 parse 后的 xLAM 样本转换成 canonical sample。"""
    query = _require_string(sample.get("query"), "query")
    tools_parse = _require_dict(sample.get("tools_parse"), "tools_parse")
    answers_parse = _require_dict(
        sample.get("answers_parse"),
        "answers_parse",
    )

    if not tools_parse.get("success"):
        raise NormalizationError("xLAM tools were not parsed successfully")
    if not answers_parse.get("success"):
        raise NormalizationError("xLAM answers were not parsed successfully")

    tools = _require_list(tools_parse.get("value"), "tools")
    answers = _require_list(answers_parse.get("value"), "answers")

    return {
        "sample_id": _canonical_sample_id("xlam", sample.get("id")),
        "source": "xlam",
        "messages": [
            {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "tools": [normalize_xlam_tool(tool) for tool in tools],
        "assistant": {
            "content": None,
            "tool_calls": [
                normalize_xlam_tool_call(answer) for answer in answers
            ],
        },
        "metadata": {
            "category": None,
            "subcategory": None,
            "task": None,
        },
    }


def normalize_hermes_tool(tool: Any) -> dict[str, Any]:
    """统一 Hermes direct/wrapped tool definition，Schema 原样保留。"""
    raw_tool = _require_dict(tool, "tool")
    if isinstance(raw_tool.get("function"), dict):
        if raw_tool.get("type") != "function":
            raise NormalizationError(
                "wrapped Hermes tool.type must be 'function'"
            )
        raw_function = raw_tool["function"]
    else:
        raw_function = raw_tool

    name = _require_string(raw_function.get("name"), "tool.name")
    description = _require_string(
        raw_function.get("description"),
        f"tool[{name}].description",
    )
    parameters = _require_dict(
        raw_function.get("parameters"),
        f"tool[{name}].parameters",
    )

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": copy.deepcopy(parameters),
        },
    }


def normalize_hermes_tool_call(tool_call: Any) -> dict[str, Any]:
    """把 repaired Hermes call object 包成 canonical function call。"""
    repaired_call = _require_dict(tool_call, "tool_call")
    name = _require_string(
        repaired_call.get("name"),
        "tool_call.name",
    )
    arguments = _require_dict(
        repaired_call.get("arguments"),
        f"tool_call[{name}].arguments",
    )

    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": copy.deepcopy(arguments),
        },
    }


def normalize_hermes_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """将一条 repaired Hermes 样本转换成 canonical sample。"""
    conversations = _require_list(
        sample.get("conversations"),
        "conversations",
    )
    repaired_tools = _require_list(sample.get("tools"), "tools")
    parsed_calls = _require_list(
        sample.get("tool_calls"),
        "tool_calls",
    )
    user_messages: list[dict[str, str]] = []
    assistant_texts: list[str] = []

    for message in conversations:
        raw_message = _require_dict(message, "conversation message")
        role = raw_message.get("from")
        content = raw_message.get("value")

        if role == "human":
            user_messages.append(
                {
                    "role": "user",
                    "content": _require_string(
                        content,
                        "human message.value",
                    ),
                }
            )
        elif role == "gpt":
            assistant_texts.append(
                _require_string(content, "gpt message.value")
            )
        elif role != "system":
            raise NormalizationError(
                f"unsupported Hermes conversation role: {role!r}"
            )

    if not user_messages:
        raise NormalizationError("Hermes sample has no human message")
    if len(assistant_texts) != 1:
        raise NormalizationError(
            "Hermes singleturn sample must have one gpt message"
        )

    canonical_calls = [
        normalize_hermes_tool_call(call.get("parsed_value"))
        for call in parsed_calls
        if isinstance(call, dict)
    ]
    if len(canonical_calls) != len(parsed_calls):
        raise NormalizationError("Hermes tool_call record must be an object")

    # 无 tool call 的 10 条澄清回复保留为普通 assistant content；有调用时
    # 原文本只承载 Hermes XML 协议，因此不进入 canonical content。
    assistant_content = assistant_texts[0] if not canonical_calls else None

    return {
        "sample_id": _canonical_sample_id(
            "hermes",
            sample.get("id"),
        ),
        "source": "hermes",
        "messages": [
            {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
            *user_messages,
        ],
        "tools": [
            normalize_hermes_tool(tool) for tool in repaired_tools
        ],
        "assistant": {
            "content": assistant_content,
            "tool_calls": canonical_calls,
        },
        "metadata": {
            "category": sample.get("category"),
            "subcategory": sample.get("subcategory"),
            "task": sample.get("task"),
        },
    }


def normalize_dataset(
    samples: Iterable[dict[str, Any]],
    source: str,
    adapter: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """逐条运行 adapter；单条失败只记日志，不阻断其他样本。"""
    normalized: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for sample in samples:
        try:
            normalized.append(adapter(sample))
        except (KeyError, NormalizationError, TypeError, ValueError) as error:
            errors.append(
                {
                    "sample_id": sample.get("id"),
                    "source": source,
                    "error": str(error),
                }
            )

    return normalized, errors
