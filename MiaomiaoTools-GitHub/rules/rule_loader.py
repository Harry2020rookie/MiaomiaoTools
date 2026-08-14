from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from models.node_rule import FLOOR_KEYS, FloorKey, NodeDistanceRule, parse_rule_value


@dataclass(frozen=True, slots=True)
class GenerationLimit:
    minimum: int
    maximum: int
    allowed_counts: frozenset[int] | None = None


@dataclass(frozen=True, slots=True)
class RuleSet:
    distance_rules: dict[str, NodeDistanceRule]
    categories: dict[str, list[str]]
    generation_limits: dict[str, dict[FloorKey, GenerationLimit]]

    def generation_limit(self, node_type: str, floor: int) -> GenerationLimit | None:
        floor_key = FLOOR_KEYS.get(floor)
        if floor_key is None:
            raise ValueError(f"不支持的地图层数: {floor}")
        limits = self.generation_limits.get(node_type)
        return None if limits is None else limits[floor_key]

    def generation_minimum(self, node_type: str, floor: int) -> int | None:
        limit = self.generation_limit(node_type, floor)
        return None if limit is None else limit.minimum

    def generation_maximum(self, node_type: str, floor: int) -> int | None:
        limit = self.generation_limit(node_type, floor)
        return None if limit is None else limit.maximum


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取规则文件 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"规则文件 {path} 顶层必须是对象")
    return value


def _parse_generation_limit(value: object, *, context: str) -> GenerationLimit:
    # null in the source table means the type cannot be generated on this
    # floor, so its effective upper bound is zero.
    if value is None:
        return GenerationLimit(0, 0)
    if isinstance(value, int) and value >= 0:
        return GenerationLimit(value, value)
    if not isinstance(value, str):
        raise ValueError(f"{context} 的生成数量必须是数字、范围、'a 或 b' 或 null")
    compact = re.sub(r"\s+", "", value)
    if compact.isdigit():
        exact = int(compact)
        return GenerationLimit(exact, exact)
    range_match = re.fullmatch(r"(\d+)-(\d+)", compact)
    choice_match = re.fullmatch(r"(\d+)或(\d+)", compact)
    match = range_match or choice_match
    if match is None:
        raise ValueError(f"{context} 的生成数量格式无效: {value}")
    low, high = (int(item) for item in match.groups())
    if high < low:
        raise ValueError(f"{context} 的生成数量上限小于下限: {value}")
    allowed_counts = frozenset((low, high)) if choice_match is not None else None
    return GenerationLimit(low, high, allowed_counts)


def load_rules(rule_dir: str | Path) -> RuleSet:
    rule_dir = Path(rule_dir)
    raw_rules = _read_json(rule_dir / "node_distance_rules.json")
    categories = _read_json(rule_dir / "node_categories.json")
    raw_generation_limits = _read_json(rule_dir / "node_generation_limits.json")
    rules: dict[str, NodeDistanceRule] = {}
    required_keys = set(FLOOR_KEYS.values())
    for node_type, raw_floors in raw_rules.items():
        if not isinstance(raw_floors, dict) or set(raw_floors) != required_keys:
            raise ValueError(f"节点类型 {node_type} 必须完整配置 I、II、III、IV、V")
        parsed: dict[FloorKey, object] = {}
        for floor_key in FLOOR_KEYS.values():
            parsed[floor_key] = parse_rule_value(
                raw_floors[floor_key], context=f"{node_type}.{floor_key}"
            )
        rules[node_type] = NodeDistanceRule(node_type=node_type, floors=parsed)  # type: ignore[arg-type]

    for category, node_types in categories.items():
        if not isinstance(node_types, list) or not all(isinstance(item, str) for item in node_types):
            raise ValueError(f"分类 {category} 必须是字符串数组")
        missing = [node_type for node_type in node_types if node_type not in rules]
        if missing:
            raise ValueError(f"分类 {category} 引用了未配置距离规则的类型: {', '.join(missing)}")

    generation_limits: dict[str, dict[FloorKey, GenerationLimit]] = {}
    for node_type, raw_floors in raw_generation_limits.items():
        if node_type.startswith("_"):
            continue
        if not isinstance(raw_floors, dict) or set(raw_floors) != required_keys:
            raise ValueError(f"节点类型 {node_type} 必须完整配置 I、II、III、IV、V 的生成数量")
        generation_limits[node_type] = {
            floor_key: _parse_generation_limit(
                raw_floors[floor_key], context=f"{node_type}.{floor_key}"
            )
            for floor_key in FLOOR_KEYS.values()
        }
    missing_limits = sorted(
        {
            node_type
            for node_types in categories.values()
            for node_type in node_types
            if node_type not in generation_limits
        }
    )
    if missing_limits:
        raise ValueError(f"候选类型缺少生成数量上限: {', '.join(missing_limits)}")
    return RuleSet(
        distance_rules=rules,
        categories=categories,
        generation_limits=generation_limits,
    )
