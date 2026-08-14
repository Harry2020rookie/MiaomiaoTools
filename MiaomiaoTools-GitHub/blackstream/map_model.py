from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from rules.candidate_filter import filter_candidates, propagate_global_capacities
from rules.rule_loader import RuleSet


Cell = tuple[int, int]
EdgeKey = tuple[Cell, Cell]


def normalized_edge(first: Cell, second: Cell) -> EdgeKey:
    return tuple(sorted((first, second)))  # type: ignore[return-value]


@dataclass(slots=True)
class SlotState:
    cell: Cell
    screen_center: tuple[float, float] | None = None
    present: bool = False
    node_type: str | None = None
    confidence: float = 0.0
    fixed_end: bool = False
    ideal_source: bool = False
    flow_resident: bool = False
    settlement_candidate: bool = False
    settlement_distance: int | None = None
    inferred_type: str | None = None
    visited: bool = False
    candidates: list[str] = field(default_factory=list)
    distance: int | None = None

    @property
    def label(self) -> str:
        if not self.present:
            return "未生成"
        if self.ideal_source:
            return "紧急作战 · 理想源"
        if self.flow_resident:
            return "流窜“居民”"
        if self.settlement_candidate:
            return "未知的凶戾 · 据点可能位置"
        if self.inferred_type:
            return f"{self.inferred_type} · 推断"
        if self.node_type:
            return self.node_type
        return "空连接点"

    @property
    def display_type(self) -> str | None:
        return self.inferred_type or self.node_type


@dataclass(slots=True)
class EdgeState:
    first: Cell
    second: Cell
    present: bool = False
    confidence: float = 0.0
    score: float = 0.0

    @property
    def key(self) -> EdgeKey:
        return normalized_edge(self.first, self.second)


@dataclass(slots=True)
class FloorMapState:
    floor: int
    columns: int
    rows: int
    slots: dict[Cell, SlotState]
    edges: dict[EdgeKey, EdgeState]
    start_cell: Cell | None = None
    source_path: str | None = None
    reference_template_id: str | None = None
    reference_template_needs_review: bool = False
    current_cell: Cell | None = None
    map_region: tuple[int, int, int, int] | None = None
    status: str = "等待上传截图"

    @classmethod
    def empty(cls, floor: int, columns: int, rows: int) -> "FloorMapState":
        slots = {
            (column, row): SlotState((column, row))
            for row in range(rows)
            for column in range(columns)
        }
        edges: dict[EdgeKey, EdgeState] = {}
        for row in range(rows):
            for column in range(columns):
                cell = (column, row)
                if column + 1 < columns:
                    edge = EdgeState(cell, (column + 1, row))
                    edges[edge.key] = edge
                if row + 1 < rows:
                    edge = EdgeState(cell, (column, row + 1))
                    edges[edge.key] = edge
        return cls(floor, columns, rows, slots, edges)

    def neighbors(self, cell: Cell) -> list[Cell]:
        result: list[Cell] = []
        for edge in self.edges.values():
            if not edge.present:
                continue
            if edge.first == cell:
                result.append(edge.second)
            elif edge.second == cell:
                result.append(edge.first)
        return result

    def trim_trailing_columns(self, columns: int) -> None:
        """Remove unused rightmost lattice columns after a template is known."""

        if columns >= self.columns:
            return
        retained = {cell for cell in self.slots if cell[0] < columns}
        self.slots = {cell: slot for cell, slot in self.slots.items() if cell in retained}
        self.edges = {
            key: edge
            for key, edge in self.edges.items()
            if edge.first in retained and edge.second in retained
        }
        self.columns = columns
        if self.start_cell not in retained:
            self.start_cell = None
        if self.current_cell not in retained:
            self.current_cell = None

    def recompute(self, rules: RuleSet) -> None:
        self._compute_distances()
        self._mark_fixed_ends()
        self._enforce_single_settlement()
        self._mark_resident_candidates()
        self._compute_candidates(rules)

    def _enforce_single_settlement(self) -> None:
        settlements = [
            slot
            for slot in self.slots.values()
            if slot.present and slot.node_type == "“居民”据点"
        ]
        if len(settlements) <= 1:
            return
        keep = max(settlements, key=lambda slot: slot.confidence)
        for slot in settlements:
            if slot is keep:
                continue
            slot.node_type = "未知的凶戾"
            slot.inferred_type = None
            slot.confidence = min(slot.confidence, 0.90)

    def _mark_resident_candidates(self) -> None:
        for slot in self.slots.values():
            slot.settlement_candidate = False
            slot.settlement_distance = None
        flow_cells = [
            cell
            for cell, slot in self.slots.items()
            if slot.present and slot.flow_resident
        ]
        settlement_exists = any(
            slot.present and slot.node_type == "“居民”据点"
            for slot in self.slots.values()
        )
        if self.floor not in {2, 4, 5} or not flow_cells or settlement_exists:
            return
        possible: set[Cell] | None = None
        distances_by_flow: dict[Cell, dict[Cell, int]] = {}
        for origin in flow_cells:
            distances = {origin: 0}
            queue: deque[Cell] = deque([origin])
            while queue:
                cell = queue.popleft()
                if distances[cell] >= 5:
                    continue
                for neighbor in self.neighbors(cell):
                    if neighbor not in distances:
                        distances[neighbor] = distances[cell] + 1
                        queue.append(neighbor)
            distances_by_flow[origin] = distances
            reachable = {
                cell
                for cell, distance in distances.items()
                if distance <= 5
                and cell not in flow_cells
                and self.slots[cell].present
                and self.slots[cell].node_type == "未知的凶戾"
            }
            possible = reachable if possible is None else possible & reachable
        for cell in possible or set():
            slot = self.slots[cell]
            slot.settlement_candidate = True
            slot.settlement_distance = max(
                distances_by_flow[origin].get(cell, 99) for origin in flow_cells
            )

    def _compute_distances(self) -> None:
        for slot in self.slots.values():
            slot.distance = None
            slot.fixed_end = False
        if self.start_cell is None or not self.slots[self.start_cell].present:
            return
        distances = {self.start_cell: 0}
        queue: deque[Cell] = deque([self.start_cell])
        while queue:
            cell = queue.popleft()
            for neighbor in self.neighbors(cell):
                if not self.slots[neighbor].present or neighbor in distances:
                    continue
                distances[neighbor] = distances[cell] + 1
                queue.append(neighbor)
        for cell, distance in distances.items():
            self.slots[cell].distance = distance

    def _mark_fixed_ends(self) -> None:
        for slot in self.slots.values():
            if slot.present and slot.node_type == "险路尽头" and slot.confidence >= 0.90:
                slot.fixed_end = True
        if self.floor != 1:
            return
        for cell, slot in self.slots.items():
            if (
                slot.present
                and cell != self.start_cell
                and slot.distance == 5
                and len(self.neighbors(cell)) == 1
            ):
                slot.fixed_end = True
                slot.node_type = "险路尽头"
                slot.confidence = max(slot.confidence, 0.92)

    def _compute_candidates(self, rules: RuleSet) -> None:
        for slot in self.slots.values():
            slot.candidates = []
            slot.inferred_type = None
        fixed_traders = self._fixed_trader_cells()
        appeared = Counter(
            slot.node_type
            for slot in self.slots.values()
            if slot.present
            and slot.node_type
            and slot.node_type not in {"未知的诡秘", "未知的凶戾", "起点"}
        )
        appeared["诡意行商"] += sum(
            self.slots[cell].present
            and self.slots[cell].node_type == "未知的诡秘"
            for cell in fixed_traders
        )
        mystery_predictions: dict[str, list[str]] = {}
        mystery_cells: dict[str, Cell] = {}
        for cell, slot in self.slots.items():
            if slot.distance is None:
                continue
            key = f"C{cell[0] + 1}R{cell[1] + 1}"
            if slot.node_type == "未知的诡秘":
                if cell in fixed_traders:
                    slot.candidates = ["诡意行商"]
                    slot.inferred_type = "诡意行商"
                    continue
                mystery_predictions[key] = filter_candidates(
                    rules,
                    category="unknown_mystery",
                    floor=self.floor,
                    distance_from_start=slot.distance,
                    appeared_counts=appeared,
                )
                mystery_cells[key] = cell
            elif slot.node_type == "未知的凶戾":
                if slot.ideal_source:
                    slot.candidates = ["紧急作战"]
                    slot.inferred_type = "紧急作战"
                    continue
                slot.candidates = [
                    node_type
                    for node_type in ("作战", "紧急作战")
                    if rules.distance_rules[node_type].allows(self.floor, slot.distance)
                    and (
                        rules.generation_maximum(node_type, self.floor) is None
                        or appeared[node_type]
                        < rules.generation_maximum(node_type, self.floor)
                    )
                ]
                if len(slot.candidates) == 1:
                    slot.inferred_type = slot.candidates[0]
        if mystery_predictions:
            reduced = propagate_global_capacities(
                rules,
                floor=self.floor,
                predictions=mystery_predictions,
                appeared_counts=appeared,
            )
            for key, candidates in reduced.items():
                slot = self.slots[mystery_cells[key]]
                slot.candidates = candidates
                if self.floor == 1:
                    slot.candidates = [
                        candidate
                        for candidate in candidates
                        if candidate != "诡意行商"
                    ]
                if len(slot.candidates) == 1:
                    slot.inferred_type = slot.candidates[0]

    def _fixed_trader_cells(self) -> set[Cell]:
        if self.floor != 1:
            return set()
        return {
            "floor_1_template_01": {(3, 1)},
            "floor_1_template_02": {(3, 1)},
            "floor_1_template_03": {(1, 0), (4, 1)},
        }.get(self.reference_template_id or "", set())
