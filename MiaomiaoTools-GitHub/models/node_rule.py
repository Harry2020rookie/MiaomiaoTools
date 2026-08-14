from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


FloorKey = Literal["I", "II", "III", "IV", "V"]
RuleValue = tuple[int, int] | None | Literal["undetermined"]
FLOOR_KEYS: dict[int, FloorKey] = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}


@dataclass(frozen=True, slots=True)
class NodeDistanceRule:
    node_type: str
    floors: dict[FloorKey, RuleValue]

    def allows(self, floor: int, distance: int) -> bool:
        key = FLOOR_KEYS.get(floor)
        if key is None:
            raise ValueError(f"不支持的地图层数: {floor}")
        rule = self.floors[key]
        if rule is None:
            return False
        if rule == "undetermined":
            return True
        return rule[0] <= distance <= rule[1]


def parse_rule_value(value: Any, *, context: str) -> RuleValue:
    if value is None or value == "undetermined":
        return value
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{context} 必须是 [min, max]、null 或 undetermined")
    low, high = value
    if not isinstance(low, int) or not isinstance(high, int) or low < 0 or high < low:
        raise ValueError(f"{context} 的距离范围无效: {value}")
    return (low, high)
