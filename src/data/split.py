"""对最终 canonical 数据执行 group-aware、tool-aware 的可复现切分。"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from sklearn.model_selection import GroupShuffleSplit


SEED = 42
MIN_UNSEEN_TOOL_SAMPLES = 10
UNSEEN_TARGET_RATIO = 0.05
SEEN_TRAIN_RATIO = 0.90

TRAIN = "train"
DEV_SEEN = "dev_seen"
TEST_SEEN = "test_seen"
TEST_UNSEEN_TOOLS = "test_unseen_tools"
SPLIT_NAMES = (TRAIN, DEV_SEEN, TEST_SEEN, TEST_UNSEEN_TOOLS)


@dataclass
class SplitResult:
    splits: dict[str, list[dict[str, Any]]]
    heldout_tools: set[str]
    tool_frequency: Counter[str]
    target_unseen_samples: int


def normalize_text(text: str) -> str:
    """只统一首尾空白、大小写和连续空白。"""
    return " ".join(text.strip().lower().split())


def build_group_key(sample: dict[str, Any]) -> str:
    """基于 canonical user content 生成稳定 query group。"""
    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"messages 不是 array：{sample.get('sample_id')!r}")

    user_contents = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not user_contents or not all(
        isinstance(content, str) for content in user_contents
    ):
        raise ValueError(
            f"缺少合法 user content：{sample.get('sample_id')!r}"
        )

    normalized = normalize_text("\n".join(user_contents))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def collect_tool_names(sample: dict[str, Any]) -> set[str]:
    """取得样本上下文中暴露给模型的全部候选工具名。"""
    tools = sample.get("tools")
    if not isinstance(tools, list):
        raise ValueError(f"tools 不是 array：{sample.get('sample_id')!r}")

    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError(f"tool 不是 object：{sample.get('sample_id')!r}")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(
                f"tool.function 不是 object：{sample.get('sample_id')!r}"
            )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tool name 非法：{sample.get('sample_id')!r}")
        names.add(name)

    return names


def count_tool_frequency(
    samples: Iterable[dict[str, Any]],
) -> Counter[str]:
    """统计每个工具出现在多少条样本的候选 tools 中。"""
    frequency: Counter[str] = Counter()
    for sample in samples:
        frequency.update(collect_tool_names(sample))
    return frequency


def select_unseen_tools(
    samples: list[dict[str, Any]],
    group_keys: list[str],
    tool_names: list[set[str]],
    tool_frequency: Counter[str],
    seed: int = SEED,
    min_samples: int = MIN_UNSEEN_TOOL_SAMPLES,
    target_ratio: float = UNSEEN_TARGET_RATIO,
) -> tuple[set[str], set[str], int]:
    """随机排序 eligible tools，再贪心逼近目标 unseen 样本数。

    样本按 group 闭包计入：任何命中 held-out tool 的 query group 都整体进入
    unseen split，从而同时满足工具隔离与 query 防泄漏。
    """
    if not (
        len(samples) == len(group_keys) == len(tool_names)
    ):
        raise ValueError("samples/group_keys/tool_names 长度不一致")

    group_sizes = Counter(group_keys)
    tool_groups: dict[str, set[str]] = defaultdict(set)
    for group_key, names in zip(group_keys, tool_names):
        for name in names:
            tool_groups[name].add(group_key)

    eligible_tools = sorted(
        name for name, count in tool_frequency.items() if count >= min_samples
    )
    random.Random(seed).shuffle(eligible_tools)
    target_samples = round(len(samples) * target_ratio)
    selected: set[str] = set()
    unseen_groups: set[str] = set()
    unseen_count = 0

    for name in eligible_tools:
        added_groups = tool_groups[name] - unseen_groups
        if not added_groups:
            continue

        candidate_count = unseen_count + sum(
            group_sizes[group] for group in added_groups
        )
        if abs(target_samples - candidate_count) >= abs(
            target_samples - unseen_count
        ):
            continue

        selected.add(name)
        unseen_groups.update(added_groups)
        unseen_count = candidate_count
        if unseen_count >= target_samples:
            break

    if not selected:
        raise RuntimeError("没有选出任何 held-out tool")

    return selected, unseen_groups, target_samples


def _group_shuffle_90_5_5(
    indices: list[int],
    group_keys: list[str],
    seed: int,
) -> tuple[set[int], set[int], set[int]]:
    """用两次 GroupShuffleSplit 得到约 90/5/5。"""
    groups = [group_keys[index] for index in indices]
    if len(set(groups)) < 3:
        raise ValueError("seen source 至少需要 3 个 query group")

    first = GroupShuffleSplit(
        n_splits=1,
        train_size=SEEN_TRAIN_RATIO,
        random_state=seed,
    )
    train_local, temp_local = next(
        first.split(indices, groups=groups)
    )
    train_indices = {indices[position] for position in train_local}
    temp_indices = [indices[position] for position in temp_local]
    temp_groups = [group_keys[index] for index in temp_indices]

    second = GroupShuffleSplit(
        n_splits=1,
        train_size=0.5,
        random_state=seed + 1,
    )
    dev_local, test_local = next(
        second.split(temp_indices, groups=temp_groups)
    )
    dev_indices = {temp_indices[position] for position in dev_local}
    test_indices = {temp_indices[position] for position in test_local}
    return train_indices, dev_indices, test_indices


def split_seen_pool(
    samples: list[dict[str, Any]],
    seen_indices: list[int],
    group_keys: list[str],
    seed: int = SEED,
) -> dict[int, str]:
    """按 source 分别执行 seen pool 的 group-aware 90/5/5 split。"""
    indices_by_source: dict[str, list[int]] = defaultdict(list)
    group_sources: dict[str, set[str]] = defaultdict(set)

    for index in seen_indices:
        source = samples[index].get("source")
        if not isinstance(source, str):
            raise ValueError(
                f"source 非法：{samples[index].get('sample_id')!r}"
            )
        indices_by_source[source].append(index)
        group_sources[group_keys[index]].add(source)

    cross_source_groups = [
        group for group, sources in group_sources.items() if len(sources) > 1
    ]
    if cross_source_groups:
        raise ValueError(
            "存在跨 source 的相同 query group，不能安全地按 source 分切："
            f"{len(cross_source_groups)} groups"
        )

    assignments: dict[int, str] = {}
    for source in sorted(indices_by_source):
        train_indices, dev_indices, test_indices = _group_shuffle_90_5_5(
            indices_by_source[source],
            group_keys,
            seed,
        )
        assignments.update({index: TRAIN for index in train_indices})
        assignments.update({index: DEV_SEEN for index in dev_indices})
        assignments.update({index: TEST_SEEN for index in test_indices})

    return assignments


def split_dataset(
    samples: list[dict[str, Any]],
    seed: int = SEED,
) -> SplitResult:
    """执行 unseen tool holdout 与 seen 90/5/5 group split。"""
    group_keys = [build_group_key(sample) for sample in samples]
    tool_names = [collect_tool_names(sample) for sample in samples]
    tool_frequency = Counter()
    for names in tool_names:
        tool_frequency.update(names)

    heldout_tools, unseen_groups, target_unseen = select_unseen_tools(
        samples,
        group_keys,
        tool_names,
        tool_frequency,
        seed=seed,
    )
    unseen_indices = {
        index
        for index, group_key in enumerate(group_keys)
        if group_key in unseen_groups
    }
    seen_indices = [
        index for index in range(len(samples)) if index not in unseen_indices
    ]
    assignments = split_seen_pool(
        samples,
        seen_indices,
        group_keys,
        seed=seed,
    )
    assignments.update(
        {index: TEST_UNSEEN_TOOLS for index in unseen_indices}
    )

    splits = {name: [] for name in SPLIT_NAMES}
    for index, sample in enumerate(samples):
        splits[assignments[index]].append(sample)

    return SplitResult(
        splits=splits,
        heldout_tools=heldout_tools,
        tool_frequency=tool_frequency,
        target_unseen_samples=target_unseen,
    )
