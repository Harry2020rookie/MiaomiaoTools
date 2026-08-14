from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

from .rule_loader import RuleSet


def filter_candidates(
    rule_set: RuleSet,
    *,
    category: str,
    floor: int,
    distance_from_start: int,
    appeared_counts: Mapping[str, int] | None = None,
) -> list[str]:
    if category not in rule_set.categories:
        raise ValueError(f"未知节点分类: {category}")
    if distance_from_start < 0:
        raise ValueError("节点图距离不能为负数")
    appeared_counts = appeared_counts or {}
    candidates: list[str] = []
    for node_type in rule_set.categories[category]:
        if not rule_set.distance_rules[node_type].allows(floor, distance_from_start):
            continue
        maximum = rule_set.generation_maximum(node_type, floor)
        if maximum is not None and appeared_counts.get(node_type, 0) >= maximum:
            continue
        candidates.append(node_type)
    return candidates


@dataclass(slots=True)
class _FlowEdge:
    target: int
    reverse: int
    capacity: int


class _Dinic:
    def __init__(self, size: int) -> None:
        self.graph: list[list[_FlowEdge]] = [[] for _ in range(size)]

    def add_edge(self, source: int, target: int, capacity: int) -> None:
        forward = _FlowEdge(target, len(self.graph[target]), capacity)
        backward = _FlowEdge(source, len(self.graph[source]), 0)
        self.graph[source].append(forward)
        self.graph[target].append(backward)

    def max_flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            levels = [-1] * len(self.graph)
            levels[source] = 0
            queue = [source]
            for vertex in queue:
                for edge in self.graph[vertex]:
                    if edge.capacity > 0 and levels[edge.target] < 0:
                        levels[edge.target] = levels[vertex] + 1
                        queue.append(edge.target)
            if levels[sink] < 0:
                return total
            positions = [0] * len(self.graph)

            def send(vertex: int, amount: int) -> int:
                if vertex == sink:
                    return amount
                while positions[vertex] < len(self.graph[vertex]):
                    edge = self.graph[vertex][positions[vertex]]
                    if edge.capacity > 0 and levels[edge.target] == levels[vertex] + 1:
                        pushed = send(edge.target, min(amount, edge.capacity))
                        if pushed:
                            edge.capacity -= pushed
                            self.graph[edge.target][edge.reverse].capacity += pushed
                            return pushed
                    positions[vertex] += 1
                return 0

            while pushed := send(source, 1 << 30):
                total += pushed


def _has_bounded_assignment(
    predictions: Mapping[str, list[str]],
    *,
    minimums: Mapping[str, int],
    maximums: Mapping[str, int],
) -> bool:
    """Return whether every unknown node can be assigned within all quotas."""

    if any(not candidates for candidates in predictions.values()):
        return False
    node_ids = list(predictions)
    node_types = sorted(
        set(minimums) | {item for candidates in predictions.values() for item in candidates}
    )
    node_count = len(node_ids)
    if sum(minimums.get(item, 0) for item in node_types) > node_count:
        return False
    if sum(maximums.get(item, node_count) for item in node_types) < node_count:
        return False
    if any(minimums.get(item, 0) > maximums.get(item, node_count) for item in node_types):
        return False

    source = 0
    node_offset = 1
    type_offset = node_offset + node_count
    sink = type_offset + len(node_types)
    super_source = sink + 1
    super_sink = sink + 2
    flow = _Dinic(super_sink + 1)
    balances = [0] * (super_sink + 1)

    def add_bounded_edge(start: int, end: int, lower: int, upper: int) -> None:
        if lower > upper:
            raise ValueError("流量下界不能超过上界")
        flow.add_edge(start, end, upper - lower)
        balances[start] -= lower
        balances[end] += lower

    type_vertices = {
        node_type: type_offset + index for index, node_type in enumerate(node_types)
    }
    for index, node_id in enumerate(node_ids):
        node_vertex = node_offset + index
        add_bounded_edge(source, node_vertex, 1, 1)
        for node_type in predictions[node_id]:
            add_bounded_edge(node_vertex, type_vertices[node_type], 0, 1)
    for node_type, vertex in type_vertices.items():
        add_bounded_edge(
            vertex,
            sink,
            minimums.get(node_type, 0),
            maximums.get(node_type, node_count),
        )
    add_bounded_edge(sink, source, 0, node_count)

    required = 0
    for vertex, balance in enumerate(balances[: super_source]):
        if balance > 0:
            flow.add_edge(super_source, vertex, balance)
            required += balance
        elif balance < 0:
            flow.add_edge(vertex, super_sink, -balance)
    return flow.max_flow(super_source, super_sink) == required


def _has_feasible_assignment(
    predictions: Mapping[str, list[str]],
    *,
    minimums: Mapping[str, int],
    maximums: Mapping[str, int],
    allowed_totals: Mapping[str, frozenset[int]] | None = None,
) -> bool:
    """Check min/max quotas plus optional non-contiguous allowed totals."""

    choices = {
        node_type: tuple(sorted(values))
        for node_type, values in (allowed_totals or {}).items()
    }
    if any(not values for values in choices.values()):
        return False
    if not choices:
        return _has_bounded_assignment(
            predictions, minimums=minimums, maximums=maximums
        )
    node_types = tuple(choices)
    for totals in product(*(choices[node_type] for node_type in node_types)):
        exact_minimums = dict(minimums)
        exact_maximums = dict(maximums)
        for node_type, total in zip(node_types, totals):
            exact_minimums[node_type] = total
            exact_maximums[node_type] = total
        if _has_bounded_assignment(
            predictions,
            minimums=exact_minimums,
            maximums=exact_maximums,
        ):
            return True
    return False


def propagate_global_capacities(
    rule_set: RuleSet,
    *,
    floor: int,
    predictions: Mapping[str, list[str]],
    appeared_counts: Mapping[str, int] | None = None,
) -> dict[str, list[str]]:
    """Prune candidates using one global min/max constrained assignment.

    Every unknown node must receive exactly one type.  Visible named nodes
    consume that type's capacity first.  A candidate remains only if at least
    one complete assignment containing it satisfies every floor-level minimum
    and maximum.  If the observed screenshot is already inconsistent with the
    table, the independent candidate lists are retained instead of erasing the
    prediction result.
    """

    appeared_counts = appeared_counts or {}
    independent = {
        node_id: [
            node_type
            for node_type in candidates
            if (
                (maximum := rule_set.generation_maximum(node_type, floor)) is None
                or appeared_counts.get(node_type, 0) < maximum
            )
        ]
        for node_id, candidates in predictions.items()
    }
    category_types = rule_set.categories["unknown_mystery"]
    minimums: dict[str, int] = {}
    maximums: dict[str, int] = {}
    allowed_totals: dict[str, frozenset[int]] = {}
    node_count = len(independent)
    for node_type in category_types:
        limit = rule_set.generation_limit(node_type, floor)
        minimum = None if limit is None else limit.minimum
        maximum = None if limit is None else limit.maximum
        visible = appeared_counts.get(node_type, 0)
        minimums[node_type] = max(0, (minimum or 0) - visible)
        maximums[node_type] = (
            node_count if maximum is None else max(0, maximum - visible)
        )
        if limit is not None and limit.allowed_counts is not None:
            allowed_totals[node_type] = frozenset(
                total - visible
                for total in limit.allowed_counts
                if visible <= total <= visible + node_count
            )

    if not _has_feasible_assignment(
        independent,
        minimums=minimums,
        maximums=maximums,
        allowed_totals=allowed_totals,
    ):
        return independent

    reduced: dict[str, list[str]] = {}
    for node_id, candidates in independent.items():
        retained: list[str] = []
        for candidate in candidates:
            fixed = {key: list(value) for key, value in independent.items()}
            fixed[node_id] = [candidate]
            if _has_feasible_assignment(
                fixed,
                minimums=minimums,
                maximums=maximums,
                allowed_totals=allowed_totals,
            ):
                retained.append(candidate)
        reduced[node_id] = retained
    return reduced
