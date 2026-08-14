from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from .grid_geometry import GRID_SHAPES, GridGeometryError, locate_template_points


class TemplateValidationError(ValueError):
    """Raised when a structured map template is unsafe to use."""


@dataclass(slots=True)
class MapNode:
    id: str
    x: float
    y: float
    distance_from_start: int
    is_start: bool = False
    is_end: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MapNode":
        return cls(
            id=str(value["id"]),
            x=float(value["x"]),
            y=float(value["y"]),
            distance_from_start=int(value["distance_from_start"]),
            is_start=bool(value.get("is_start", False)),
            is_end=bool(value.get("is_end", False)),
        )


@dataclass(slots=True)
class MapTemplate:
    template_id: str
    floor: int
    image_file: str
    start_node_id: str
    nodes: list[MapNode]
    edges: list[tuple[str, str]]
    end_node_ids: list[str]
    image_width: int | None = None
    image_height: int | None = None
    annotation_source: str = "manual"
    source_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MapTemplate":
        known = {
            "template_id",
            "floor",
            "image_file",
            "start_node_id",
            "nodes",
            "edges",
            "end_node_ids",
            "image_width",
            "image_height",
            "annotation_source",
            "source_sha256",
            "metadata",
        }
        metadata = dict(value.get("metadata", {}))
        metadata.update({key: val for key, val in value.items() if key not in known})
        return cls(
            template_id=str(value["template_id"]),
            floor=int(value["floor"]),
            image_file=str(value["image_file"]),
            start_node_id=str(value["start_node_id"]),
            nodes=[MapNode.from_dict(node) for node in value["nodes"]],
            edges=[(str(edge[0]), str(edge[1])) for edge in value["edges"]],
            end_node_ids=[str(node_id) for node_id in value["end_node_ids"]],
            image_width=(int(value["image_width"]) if value.get("image_width") else None),
            image_height=(int(value["image_height"]) if value.get("image_height") else None),
            annotation_source=str(value.get("annotation_source", "manual")),
            source_sha256=(str(value["source_sha256"]) if value.get("source_sha256") else None),
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: str | Path, *, validate: bool = True) -> "MapTemplate":
        path = Path(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TemplateValidationError(f"无法读取模板 {path}: {exc}") from exc
        try:
            template = cls.from_dict(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise TemplateValidationError(f"模板 {path} 数据格式错误: {exc}") from exc
        if validate:
            template.validate()
        return template

    @classmethod
    def create(
        cls,
        *,
        template_id: str,
        floor: int,
        image_file: str,
        points: Iterable[tuple[str, float, float]],
        edges: Iterable[tuple[str, str]],
        start_node_id: str,
        end_node_ids: Iterable[str],
        **kwargs: Any,
    ) -> "MapTemplate":
        point_list = list(points)
        edge_list = [tuple(edge) for edge in edges]
        end_ids = list(end_node_ids)
        graph = nx.Graph()
        graph.add_nodes_from(node_id for node_id, _, _ in point_list)
        graph.add_edges_from(edge_list)
        if start_node_id not in graph:
            raise TemplateValidationError(f"出生点 {start_node_id} 不存在")
        try:
            columns, rows = GRID_SHAPES[floor]
            locate_template_points(
                point_list,
                columns=columns,
                rows=rows,
            )
        except (KeyError, GridGeometryError) as exc:
            raise TemplateValidationError(f"模板固定网格无效: {exc}") from exc
        if not nx.is_connected(graph):
            raise TemplateValidationError("Template graph is disconnected")
        distances = dict(nx.single_source_shortest_path_length(graph, start_node_id))
        nodes = [
            MapNode(
                id=node_id,
                x=float(x),
                y=float(y),
                distance_from_start=distances[node_id],
                is_start=node_id == start_node_id,
                is_end=node_id in end_ids,
            )
            for node_id, x, y in point_list
        ]
        template = cls(
            template_id=template_id,
            floor=floor,
            image_file=image_file,
            start_node_id=start_node_id,
            nodes=nodes,
            edges=edge_list,
            end_node_ids=end_ids,
            **kwargs,
        )
        template.validate()
        return template

    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(node.id for node in self.nodes)
        graph.add_edges_from(self.edges)
        return graph

    def validate(self) -> None:
        errors: list[str] = []
        if not self.template_id:
            errors.append("template_id 不能为空")
        if self.floor not in range(1, 6):
            errors.append(f"floor 必须在 1..5，当前为 {self.floor}")

        ids = [node.id for node in self.nodes]
        id_set = set(ids)
        if len(ids) != len(id_set):
            duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
            errors.append(f"节点 ID 不唯一: {', '.join(duplicates)}")
        if self.start_node_id not in id_set:
            errors.append(f"出生点 {self.start_node_id} 不存在")
        if any(node_id not in id_set for node_id in self.end_node_ids):
            invalid = sorted(set(self.end_node_ids) - id_set)
            errors.append(f"尽头节点 ID 无效: {', '.join(invalid)}")
        for node in self.nodes:
            if not 0.0 <= node.x <= 1.0 or not 0.0 <= node.y <= 1.0:
                errors.append(f"节点 {node.id} 的归一化坐标越界: ({node.x}, {node.y})")

        edge_set: set[tuple[str, str]] = set()
        for left, right in self.edges:
            if left not in id_set or right not in id_set:
                errors.append(f"边 {left}-{right} 引用了不存在的节点")
            if left == right:
                errors.append(f"节点 {left} 存在自环")
            canonical = tuple(sorted((left, right)))
            if canonical in edge_set:
                errors.append(f"边 {left}-{right} 重复")
            edge_set.add(canonical)

        graph = self.graph()
        if graph.number_of_nodes() and not nx.is_connected(graph):
            components = [sorted(component) for component in nx.connected_components(graph)]
            errors.append(f"地图图不连通: {components}")
        if self.start_node_id in graph:
            try:
                columns, rows = GRID_SHAPES[self.floor]
                locate_template_points(
                    ((node.id, node.x, node.y) for node in self.nodes),
                    columns=columns,
                    rows=rows,
                )
            except (KeyError, GridGeometryError) as exc:
                errors.append(f"固定网格无效: {exc}")
            distances = dict(
                nx.single_source_shortest_path_length(graph, self.start_node_id)
            )
            for node in self.nodes:
                expected = distances.get(node.id)
                if expected is None:
                    errors.append(f"无法计算节点 {node.id} 的沿路线最短距离")
                elif node.distance_from_start != expected:
                    errors.append(
                        f"节点 {node.id} 沿路线最短距离应为 {expected}，"
                        f"JSON 中为 {node.distance_from_start}"
                    )
                if node.is_start != (node.id == self.start_node_id):
                    errors.append(f"节点 {node.id} 的 is_start 标记错误")
                if node.is_end != (node.id in self.end_node_ids):
                    errors.append(f"节点 {node.id} 的 is_end 标记错误")

        if errors:
            raise TemplateValidationError(f"模板 {self.template_id} 校验失败：" + "；".join(errors))

    def node(self, node_id: str) -> MapNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "template_id": self.template_id,
            "floor": self.floor,
            "image_file": self.image_file,
            "start_node_id": self.start_node_id,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
            "end_node_ids": self.end_node_ids,
            "annotation_source": self.annotation_source,
        }
        if self.image_width:
            value["image_width"] = self.image_width
        if self.image_height:
            value["image_height"] = self.image_height
        if self.source_sha256:
            value["source_sha256"] = self.source_sha256
        if self.metadata:
            value["metadata"] = self.metadata
        return value

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
