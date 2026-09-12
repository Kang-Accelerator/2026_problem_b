"""问题3：全向干扰源的保证发现、稳健定位与清除。

默认只执行离线检查。只有显式提供 ``--robot-id`` 时才会连接官方模拟器；
本模块本身不实现模拟器，也不在导入时产生网络请求。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

try:
    from .problem1_geometry import build_regular_ngon, minimum_enclosing_circle
    from .problem2_model import minimum_outer_radius, posterior_outer_polygon
    from ..action_logger import ActionLogger
    from ..simulator_client import (
        RuntimeBudgetExceeded,
        SimulatorClient,
        SimulatorClientError,
        UncertainActionError,
    )
except ImportError:  # 支持直接执行本文件。
    project_dir = Path(__file__).resolve().parents[1]
    if str(project_dir.parent) not in sys.path:
        sys.path.insert(0, str(project_dir.parent))
    from competition_b.code.problem1_geometry import build_regular_ngon, minimum_enclosing_circle
    from competition_b.code.problem2_model import minimum_outer_radius, posterior_outer_polygon
    from competition_b.action_logger import ActionLogger
    from competition_b.simulator_client import (
        RuntimeBudgetExceeded,
        SimulatorClient,
        SimulatorClientError,
        UncertainActionError,
    )


Point = tuple[float, float]
TERMINAL_STATES = {"CLEARED", "ABSENT"}


class SimulatorLike(Protocol):
    def enter(self) -> dict[str, Any]: ...
    def measure(self, x: float, y: float, channel: int) -> dict[str, Any]: ...
    def clear(self, x: float, y: float, channel: int) -> dict[str, Any]: ...
    def exit(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Config:
    target_radius_m: float = 1800.0
    guarantee_receive_m: float = 1000.0
    max_receive_m: float = 1500.0
    bearing_error_deg: float = 1.0
    near_radius_m: float = 5.0
    clear_radius_m: float = 20.0
    # MEC 半径不超过清除半径时，MEC 中心到区域内任一点均不超过清除半径；
    # 这里使用20m而不是人为收紧到18m，减少无必要的精定位动作，仍是几何安全判据。
    safe_clear_radius_m: float = 20.0
    speed_mps: float = 5.0
    switch_time_s: float = 1.0
    measure_time_s: float = 5.0
    clear_success_time_s: float = 5.0
    clear_fail_time_s: float = 3.0
    channel_count: int = 20
    source_count_min: int = 10
    source_count_max: int = 16
    skeleton_ring_radius_m: float = 1200.0
    second_baseline_m: float = 500.0
    second_offset_deg: float = 45.0
    refinement_baseline_m: float = 60.0
    max_refinements: int = 3
    fallback_cell_m: float = 25.0
    max_fallback_clear_attempts: int = 500
    reserve_real_s: float = 90.0
    polygon_edges: int = 256
    opportunistic_max_per_skeleton: int = 4
    opportunistic_min_baseline_m: float = 300.0
    opportunistic_min_cross_angle_deg: float = 20.0
    opportunistic_max_center_distance_m: float = 1000.0
    inline_clear_detour_ratio_max: float = 0.20
    inline_clear_detour_abs_max_m: float = 250.0
    route_two_opt_passes: int = 8

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Config":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"未知配置字段：{unknown}")
        cfg = cls(**{key: raw[key] for key in raw})
        cfg.validate()
        return cfg

    def validate(self) -> None:
        positive = {
            "target_radius_m": self.target_radius_m,
            "guarantee_receive_m": self.guarantee_receive_m,
            "max_receive_m": self.max_receive_m,
            "clear_radius_m": self.clear_radius_m,
            "speed_mps": self.speed_mps,
            "fallback_cell_m": self.fallback_cell_m,
        }
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive.values()):
            raise ValueError("半径、速度和网格尺寸必须是正有限数")
        if self.guarantee_receive_m > self.max_receive_m:
            raise ValueError("guarantee_receive_m 不得大于 max_receive_m")
        if not 0.0 < self.safe_clear_radius_m <= self.clear_radius_m:
            raise ValueError("safe_clear_radius_m 必须位于 (0, clear_radius_m]")
        if self.fallback_cell_m / math.sqrt(2.0) > self.clear_radius_m + 1e-12:
            raise ValueError("fallback_cell_m 太大，不能保证每个网格单元被清除半径覆盖")
        if not 1 <= self.source_count_min <= self.source_count_max <= self.channel_count:
            raise ValueError("源数量上下界必须位于频道范围内")
        if self.max_refinements < 0 or self.max_fallback_clear_attempts < 1:
            raise ValueError("补测次数和兜底尝试上限非法")
        if self.opportunistic_max_per_skeleton < 0:
            raise ValueError("opportunistic_max_per_skeleton 不得为负")
        if self.opportunistic_min_baseline_m < 0.0:
            raise ValueError("opportunistic_min_baseline_m 不得为负")
        if not 0.0 <= self.opportunistic_min_cross_angle_deg <= 90.0:
            raise ValueError("opportunistic_min_cross_angle_deg 必须位于 [0, 90]")
        if self.opportunistic_max_center_distance_m <= 0.0:
            raise ValueError("opportunistic_max_center_distance_m 必须为正")
        if not 0.0 <= self.inline_clear_detour_ratio_max <= 1.0:
            raise ValueError("inline_clear_detour_ratio_max 必须位于 [0, 1]")
        if self.inline_clear_detour_abs_max_m < 0.0 or self.route_two_opt_passes < 0:
            raise ValueError("绕行距离和2-opt轮数不得为负")
        guarantee = guaranteed_second_measurement(self)
        if not guarantee["passed"]:
            raise ValueError(f"第二测点参数不满足保证接收：{guarantee}")


@dataclass
class BearingObservation:
    position: Point
    bearing_deg: float
    skeleton_index: int | None = None


@dataclass
class ChannelState:
    channel: int
    status: str = "UNKNOWN"
    no_signal_skeleton_indices: set[int] = field(default_factory=set)
    observations: list[BearingObservation] = field(default_factory=list)
    near_positions: list[Point] = field(default_factory=list)
    clear_attempts: int = 0
    fallback_used: bool = False
    cleared_at: Point | None = None
    flags: list[str] = field(default_factory=list)
    localization_center: Point | None = None
    localization_radius_m: float | None = None
    opportunistic_attempts: int = 0
    opportunistic_directions: int = 0
    localized_during_scan: bool = False
    scheduled_clear_edge: int | None = None


@dataclass
class Counters:
    L_m: float = 0.0
    N_s: int = 0
    N_m: int = 0
    N_c: int = 0
    N_f: int = 0


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def unit(angle_deg: float) -> Point:
    angle = math.radians(float(angle_deg))
    return math.cos(angle), math.sin(angle)


def project_to_target_disk(point: Sequence[float], target_radius_m: float) -> Point:
    """把动作点投影到目标圆盘；对圆内任意源，此操作不会增大距离。"""

    x, y = float(point[0]), float(point[1])
    radius = math.hypot(x, y)
    if radius <= target_radius_m or radius == 0.0:
        return x, y
    scale = target_radius_m / radius
    return x * scale, y * scale


def build_skeleton(ring_radius_m: float = 1200.0) -> list[Point]:
    """原点出发后沿正六边形依次访问的7点骨架。"""

    return [(0.0, 0.0)] + [
        (ring_radius_m * math.cos(math.radians(60.0 * index)),
         ring_radius_m * math.sin(math.radians(60.0 * index)))
        for index in range(6)
    ]


def skeleton_route_length(points: Sequence[Point]) -> float:
    return sum(distance(first, second) for first, second in zip(points, points[1:]))


def open_route_length(start: Point, order: Sequence[int], targets: Mapping[int, Point]) -> float:
    """从 start 出发、无需返回起点的访问路线长度。"""

    total = 0.0
    current = start
    for channel in order:
        total += distance(current, targets[channel])
        current = targets[channel]
    return total


def plan_open_route(
    start: Point,
    targets: Mapping[int, Point],
    two_opt_passes: int = 8,
) -> list[int]:
    """确定性的最近邻初解加开放路径2-opt；用于最多20个服务点。"""

    remaining = set(targets)
    route: list[int] = []
    current = start
    while remaining:
        channel = min(remaining, key=lambda item: (distance(current, targets[item]), item))
        route.append(channel)
        remaining.remove(channel)
        current = targets[channel]

    best_length = open_route_length(start, route, targets)
    for _ in range(two_opt_passes):
        improved = False
        for left in range(len(route) - 1):
            for right in range(left + 1, len(route)):
                candidate = route[:left] + list(reversed(route[left : right + 1])) + route[right + 1 :]
                candidate_length = open_route_length(start, candidate, targets)
                if candidate_length + 1e-9 < best_length:
                    route = candidate
                    best_length = candidate_length
                    improved = True
        if not improved:
            break
    return route


def plan_open_route_with_options(
    start: Point,
    options: Mapping[int, Sequence[Point]],
    improvement_passes: int = 8,
) -> tuple[list[int], dict[int, Point]]:
    """开放广义TSP启发式：每个频道访问一个服务点，并联合选择候选侧。"""

    normalized = {channel: list(points) for channel, points in options.items() if points}
    remaining = set(normalized)
    route: list[int] = []
    selected: dict[int, Point] = {}
    current = start
    while remaining:
        _, channel, option_index = min(
            (distance(current, point), channel, option_index)
            for channel in remaining
            for option_index, point in enumerate(normalized[channel])
        )
        selected[channel] = normalized[channel][option_index]
        route.append(channel)
        remaining.remove(channel)
        current = selected[channel]

    def length(order: Sequence[int], chosen: Mapping[int, Point]) -> float:
        return open_route_length(start, order, chosen)

    best_length = length(route, selected)
    for _ in range(improvement_passes):
        improved = False
        for channel in route:
            old_point = selected[channel]
            for point in normalized[channel]:
                if point == old_point:
                    continue
                candidate_selected = dict(selected)
                candidate_selected[channel] = point
                candidate_length = length(route, candidate_selected)
                if candidate_length + 1e-9 < best_length:
                    selected = candidate_selected
                    best_length = candidate_length
                    improved = True
        for left in range(len(route) - 1):
            for right in range(left + 1, len(route)):
                candidate_route = route[:left] + list(reversed(route[left : right + 1])) + route[right + 1 :]
                candidate_length = length(candidate_route, selected)
                if candidate_length + 1e-9 < best_length:
                    route = candidate_route
                    best_length = candidate_length
                    improved = True
        if not improved:
            break
    return route, selected


def plan_edge_clear_insertions(
    start: Point,
    end: Point,
    targets: Mapping[int, Point],
    ratio_limit: float,
    absolute_limit_m: float,
) -> tuple[list[int], float]:
    """用最便宜插入法选择可在一条骨架边上顺路清除的目标。"""

    direct = distance(start, end)
    route: list[int] = []
    remaining = set(targets)
    route_length = direct
    while remaining:
        points = [start] + [targets[channel] for channel in route] + [end]
        best: tuple[float, int, int] | None = None
        for channel in remaining:
            point = targets[channel]
            for index in range(len(points) - 1):
                extra = distance(points[index], point) + distance(point, points[index + 1]) - distance(points[index], points[index + 1])
                candidate = (extra, channel, index)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        extra, channel, index = best
        proposed = route_length + extra
        detour = proposed - direct
        ratio = detour / direct if direct > 0.0 else math.inf
        if detour > absolute_limit_m + 1e-9 or ratio > ratio_limit + 1e-12:
            break
        route.insert(index, channel)
        remaining.remove(channel)
        route_length = proposed
    return route, route_length - direct


def exact_skeleton_covering_radius(target_radius_m: float, ring_radius_m: float) -> float:
    """中心+正六边形布局在目标圆盘内的解析最坏最近距离。"""

    sector_half = math.pi / 6.0
    transition = ring_radius_m / (2.0 * math.cos(sector_half))
    boundary = math.sqrt(
        target_radius_m**2 + ring_radius_m**2
        - 2.0 * target_radius_m * ring_radius_m * math.cos(sector_half)
    )
    return max(transition, boundary)


def guaranteed_second_measurement(cfg: Config) -> dict[str, Any]:
    """解析检查侧前方第二点在所有首次可接收距离下仍保证接收。"""

    alpha = math.radians(abs(cfg.second_offset_deg) + cfg.bearing_error_deg)
    cos_alpha = math.cos(alpha)
    b = cfg.second_baseline_m
    if cos_alpha <= 0.0:
        return {"passed": False, "reason": "offset plus bearing error must be below 90 degrees"}

    def r2(d1: float) -> float:
        return math.sqrt(max(0.0, d1 * d1 + b * b - 2.0 * d1 * b * cos_alpha))

    low_candidates = (cfg.near_radius_m, min(cfg.guarantee_receive_m, cfg.max_receive_m))
    low_worst = max(r2(value) for value in low_candidates)
    high_ok = b <= 2.0 * cfg.guarantee_receive_m * cos_alpha + 1e-12
    passed = low_worst <= cfg.guarantee_receive_m + 1e-12 and high_ok
    return {
        "passed": passed,
        "baseline_m": b,
        "worst_relative_angle_deg": math.degrees(alpha),
        "max_distance_when_d1_le_guarantee_m": low_worst,
        "high_distance_condition": "b <= 2*guarantee_receive*cos(alpha)",
    }


def localization_polygon(cfg: Config, observations: Sequence[BearingObservation]) -> np.ndarray:
    """用目标圆、每次±误差扇形和1500米上界构造保守凸外包。"""

    polygon = build_regular_ngon(cfg.target_radius_m, cfg.polygon_edges)
    for observation in observations:
        polygon = posterior_outer_polygon(
            polygon,
            observation.position,
            observation.bearing_deg,
            cfg.bearing_error_deg,
            cfg.max_receive_m,
            cfg.polygon_edges,
        )
        if len(polygon) == 0:
            break
    return polygon


def grid_cover_rectangle(bounds: Sequence[float], cell_m: float) -> list[Point]:
    """返回覆盖闭矩形的格心；每个位置距某格心不超过 cell/sqrt(2)。"""

    x0, x1, y0, y1 = map(float, bounds)
    nx = max(1, int(math.ceil(max(0.0, x1 - x0) / cell_m)))
    ny = max(1, int(math.ceil(max(0.0, y1 - y0) / cell_m)))
    dx = (x1 - x0) / nx if x1 > x0 else 0.0
    dy = (y1 - y0) / ny if y1 > y0 else 0.0
    rows: list[Point] = []
    for j in range(ny):
        row = [(x0 + (i + 0.5) * dx, y0 + (j + 0.5) * dy) for i in range(nx)]
        rows.extend(row if j % 2 == 0 else reversed(row))
    return rows


def wedge_fallback_grid(cfg: Config, observation: BearingObservation) -> list[Point]:
    """覆盖首次±1°、距离0..1500米扇形的保守旋转矩形格心。"""

    along_count = max(1, int(math.ceil(cfg.max_receive_m / cfg.fallback_cell_m)))
    half_width = cfg.max_receive_m * math.sin(math.radians(cfg.bearing_error_deg))
    lateral_count = max(1, int(math.ceil(2.0 * half_width / cfg.fallback_cell_m)))
    along_step = cfg.max_receive_m / along_count
    lateral_step = 2.0 * half_width / lateral_count if half_width > 0.0 else 0.0
    forward = np.asarray(unit(observation.bearing_deg), dtype=float)
    normal = np.array([-forward[1], forward[0]], dtype=float)
    origin = np.asarray(observation.position, dtype=float)
    points: list[Point] = []
    for i in range(along_count):
        lateral_indices = range(lateral_count) if i % 2 == 0 else range(lateral_count - 1, -1, -1)
        for j in lateral_indices:
            along = (i + 0.5) * along_step
            lateral = -half_width + (j + 0.5) * lateral_step
            point = origin + along * forward + lateral * normal
            points.append((float(point[0]), float(point[1])))
    return points


class Problem3Solver:
    """保证覆盖为底线、机会观测与滚动路线优化为效率层的自动策略。"""

    def __init__(self, cfg: Config, simulator: SimulatorLike, event_logger: ActionLogger | None = None):
        cfg.validate()
        self.cfg = cfg
        self.simulator = simulator
        self.logger = event_logger
        self.skeleton = build_skeleton(cfg.skeleton_ring_radius_m)
        self.channels = {channel: ChannelState(channel) for channel in range(1, cfg.channel_count + 1)}
        self.position: Point = (0.0, 0.0)
        self.current_measure_channel = 1
        self.counters = Counters()
        self.virtual_start_s = 0.0
        self.virtual_time_s = 0.0
        self.real_started = time.monotonic()
        self.real_deadline: float | None = None
        self.flags: list[str] = []
        self.trajectory: list[dict[str, Any]] = []
        self.phase = "initial"
        self.phase_distance_m = {"coverage_scan": 0.0, "post_scan": 0.0, "other": 0.0}
        self.strategy_diagnostics = {
            "opportunistic_attempts": 0,
            "opportunistic_directions": 0,
            "opportunistic_no_signal": 0,
            "inline_clear_count": 0,
            "inline_clear_detour_m": 0.0,
            "route_replans": 0,
            "fusion_decisions": 0,
            "fused_skeleton_visits": 0,
            "fused_service_actions": 0,
            "fused_service_distance_m": 0.0,
        }

    def _record(self, kind: str, **values: Any) -> None:
        row = {"kind": kind, "phase": self.phase, **values}
        self.trajectory.append(row)
        if self.logger is not None:
            self.logger.record({"action": "strategy", **row})

    def _guard_time(self) -> None:
        if self.real_deadline is not None and time.monotonic() >= self.real_deadline - self.cfg.reserve_real_s:
            raise RuntimeBudgetExceeded("已进入收尾预留时间")

    def _accept_motion(self, point: Point) -> None:
        leg = distance(self.position, point)
        self.counters.L_m += leg
        key = self.phase if self.phase in self.phase_distance_m else "other"
        self.phase_distance_m[key] += leg
        self.position = point

    def measure(
        self,
        point: Point,
        channel: int,
        skeleton_index: int | None = None,
        role: str = "standard",
    ) -> dict[str, Any]:
        self._guard_time()
        point = project_to_target_disk(point, self.cfg.target_radius_m)
        response = self.simulator.measure(float(point[0]), float(point[1]), int(channel))
        if response.get("accepted") is not True:
            raise SimulatorClientError(f"measure 未被接受：{response}")
        self._accept_motion(point)
        if channel != self.current_measure_channel:
            self.counters.N_s += 1
            self.current_measure_channel = channel
        self.counters.N_m += 1
        self.virtual_time_s = float(response.get("virtual_time_s", self.virtual_time_s))
        result = response.get("measure_result")
        state = self.channels[channel]
        if role == "opportunistic":
            state.opportunistic_attempts += 1
            self.strategy_diagnostics["opportunistic_attempts"] += 1
        if result == "no_signal":
            if skeleton_index is not None:
                state.no_signal_skeleton_indices.add(skeleton_index)
            # 已经获得 direction/near 的频道已被证明存在。后续在别处出现
            # no_signal 只能表示该点收不到，不能把 DETECTED 降级为 PARTIAL/ABSENT。
            if state.status not in {"DETECTED", "CLEARED"}:
                state.status = "ABSENT" if len(state.no_signal_skeleton_indices) == len(self.skeleton) else "PARTIAL"
            if role == "opportunistic":
                self.strategy_diagnostics["opportunistic_no_signal"] += 1
        elif result == "direction":
            bearing = response.get("svd_deg")
            if not isinstance(bearing, (int, float)) or isinstance(bearing, bool):
                raise SimulatorClientError("direction 响应缺少数值型 svd_deg")
            state.observations.append(BearingObservation(point, float(bearing), skeleton_index))
            state.status = "DETECTED"
            if role == "opportunistic":
                state.opportunistic_directions += 1
                self.strategy_diagnostics["opportunistic_directions"] += 1
        elif result == "near":
            state.near_positions.append(point)
            state.status = "DETECTED"
        else:
            raise SimulatorClientError(f"未知 measure_result：{result!r}")
        self._record("measure", position=point, channel=channel, result=result, status=state.status, role=role)
        return response

    def clear(self, point: Point, channel: int, method: str) -> bool:
        self._guard_time()
        point = project_to_target_disk(point, self.cfg.target_radius_m)
        response = self.simulator.clear(float(point[0]), float(point[1]), int(channel))
        if response.get("accepted") is not True:
            raise SimulatorClientError(f"clear 未被接受：{response}")
        self._accept_motion(point)
        self.channels[channel].clear_attempts += 1
        success = response.get("clear_result") == "success"
        if success:
            self.counters.N_c += 1
            self.channels[channel].status = "CLEARED"
            self.channels[channel].cleared_at = point
        else:
            self.counters.N_f += 1
        self.virtual_time_s = float(response.get("virtual_time_s", self.virtual_time_s))
        self._record("clear", position=point, channel=channel, method=method, success=success)
        return success

    def _channel_order(self, candidates: Sequence[int]) -> list[int]:
        values = sorted(candidates)
        if self.current_measure_channel in values:
            start = values.index(self.current_measure_channel)
            return values[start:] + values[:start]
        ascending = values
        descending = list(reversed(values))
        return ascending if abs(ascending[0] - self.current_measure_channel) <= abs(descending[0] - self.current_measure_channel) else descending

    def _known_source_count(self) -> int:
        return sum(state.status in {"DETECTED", "CLEARED"} for state in self.channels.values())

    def _refresh_localization(self, state: ChannelState) -> tuple[Point, float] | None:
        if len(state.observations) < 2:
            state.localization_center = None
            state.localization_radius_m = None
            return None
        polygon = localization_polygon(self.cfg, state.observations)
        if len(polygon) == 0:
            state.localization_center = None
            state.localization_radius_m = None
            if "EMPTY_LOCALIZATION_REGION" not in state.flags:
                state.flags.append("EMPTY_LOCALIZATION_REGION")
            return None
        center_array, radius = minimum_enclosing_circle(polygon)
        center = (float(center_array[0]), float(center_array[1]))
        state.localization_center = center
        state.localization_radius_m = float(radius)
        if self.phase == "coverage_scan" and radius <= self.cfg.safe_clear_radius_m:
            state.localized_during_scan = True
        return center, float(radius)

    def _safe_clear_point(self, state: ChannelState) -> Point | None:
        solution = self._refresh_localization(state)
        if solution is None:
            return None
        center, radius = solution
        return center if radius <= self.cfg.safe_clear_radius_m else None

    @staticmethod
    def _acute_cross_angle(center: Point, first: Point, second: Point) -> float:
        first_vector = (first[0] - center[0], first[1] - center[1])
        second_vector = (second[0] - center[0], second[1] - center[1])
        first_norm = math.hypot(*first_vector)
        second_norm = math.hypot(*second_vector)
        if first_norm <= 1e-12 or second_norm <= 1e-12:
            return 90.0
        cosine = abs((first_vector[0] * second_vector[0] + first_vector[1] * second_vector[1]) / (first_norm * second_norm))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def _opportunistic_score(self, state: ChannelState, point: Point, skeleton_index: int) -> float | None:
        if state.status != "DETECTED" or not state.observations or state.near_positions:
            return None
        if any(observation.skeleton_index == skeleton_index for observation in state.observations):
            return None
        if self._safe_clear_point(state) is not None:
            return None
        polygon = localization_polygon(self.cfg, state.observations)
        if len(polygon) == 0:
            return None
        center_array, radius = minimum_enclosing_circle(polygon)
        center = (float(center_array[0]), float(center_array[1]))
        predicted_distance = distance(point, center)
        if predicted_distance > self.cfg.opportunistic_max_center_distance_m:
            return None
        baselines = [distance(point, observation.position) for observation in state.observations]
        if max(baselines) < self.cfg.opportunistic_min_baseline_m:
            return None
        cross_angle = max(self._acute_cross_angle(center, observation.position, point) for observation in state.observations)
        if cross_angle < self.cfg.opportunistic_min_cross_angle_deg:
            return None
        guaranteed_here = predicted_distance + float(radius) <= self.cfg.guarantee_receive_m
        range_quality = max(0.0, 1.0 - predicted_distance / self.cfg.opportunistic_max_center_distance_m)
        angle_quality = cross_angle / 90.0
        unresolved_scale = min(2.0, float(radius) / max(1e-12, self.cfg.safe_clear_radius_m))
        return (2.0 if guaranteed_here else 0.0) + 2.0 * angle_quality + range_quality + 0.1 * unresolved_scale

    def _take_opportunistic_measurements(self, point: Point, skeleton_index: int) -> None:
        ranked: list[tuple[float, int]] = []
        for channel, state in self.channels.items():
            score = self._opportunistic_score(state, point, skeleton_index)
            if score is not None:
                ranked.append((score, channel))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        for score, channel in ranked[: self.cfg.opportunistic_max_per_skeleton]:
            response = self.measure(point, channel, skeleton_index, role="opportunistic")
            self._record("opportunistic_selection", channel=channel, skeleton_index=skeleton_index, score=score)
            if response.get("measure_result") == "near":
                if not self.clear(point, channel, "near_at_opportunistic_skeleton"):
                    self.channels[channel].flags.append("NEAR_CLEAR_CONTRADICTION")
            elif response.get("measure_result") == "direction":
                self._refresh_localization(self.channels[channel])
            if self.counters.N_c >= self.cfg.source_count_max:
                break

    def _clear_ready_on_edge(self, start: Point, end: Point, edge_index: int) -> None:
        targets: dict[int, Point] = {}
        for channel, state in self.channels.items():
            if state.status != "DETECTED" or state.clear_attempts != 0:
                continue
            clear_point = self._safe_clear_point(state)
            if clear_point is None:
                continue
            remaining_edges = range(edge_index, len(self.skeleton) - 1)
            best_edge = min(
                remaining_edges,
                key=lambda index: (
                    distance(self.skeleton[index], clear_point)
                    + distance(clear_point, self.skeleton[index + 1])
                    - distance(self.skeleton[index], self.skeleton[index + 1]),
                    index,
                ),
            )
            state.scheduled_clear_edge = best_edge
            if best_edge == edge_index:
                targets[channel] = clear_point
        if not targets:
            return
        route, detour = plan_edge_clear_insertions(
            start,
            end,
            targets,
            self.cfg.inline_clear_detour_ratio_max,
            self.cfg.inline_clear_detour_abs_max_m,
        )
        if not route:
            return
        self._record(
            "inline_clear_plan",
            edge_index=edge_index,
            channels=route,
            planned_detour_m=detour,
            direct_edge_m=distance(start, end),
        )
        self.strategy_diagnostics["inline_clear_detour_m"] += detour
        for channel in route:
            point = targets[channel]
            if self.clear(point, channel, "safe_mec_inline_detour"):
                self.strategy_diagnostics["inline_clear_count"] += 1
            else:
                self.channels[channel].flags.append("SAFE_MEC_INLINE_CLEAR_FAILED")

    def scan_covering_skeleton(self) -> None:
        self.phase = "coverage_scan"
        for skeleton_index, point in enumerate(self.skeleton):
            candidates = [channel for channel, state in self.channels.items() if state.status in {"UNKNOWN", "PARTIAL"}]
            for channel in self._channel_order(candidates):
                response = self.measure(point, channel, skeleton_index, role="coverage")
                if response.get("measure_result") == "near":
                    if not self.clear(point, channel, "near_at_skeleton"):
                        self.channels[channel].flags.append("NEAR_CLEAR_CONTRADICTION")
                if self.counters.N_c >= self.cfg.source_count_max:
                    return
            self._take_opportunistic_measurements(point, skeleton_index)
            if self.counters.N_c >= self.cfg.source_count_max:
                return
            if self._known_source_count() >= self.cfg.source_count_max:
                self._record("coverage_early_stop", reason="source_upper_bound_detected", skeleton_index=skeleton_index)
                return
            if skeleton_index + 1 < len(self.skeleton):
                self._clear_ready_on_edge(point, self.skeleton[skeleton_index + 1], skeleton_index)

    def _fusion_service_target(self, channel: int) -> Point | None:
        options = self._next_service_options(channel)
        if not options:
            return None
        return min(options, key=lambda point: (distance(self.position, point), point[0], point[1]))

    def _fusion_route_length(self, start: Point, points: Sequence[Point]) -> float:
        """估计执行候选动作后的剩余开放路线长度；不把点插值成已访问。"""

        if not points:
            return 0.0
        remaining = list(points)
        current = start
        total = 0.0
        while remaining:
            index = min(range(len(remaining)), key=lambda item: (distance(current, remaining[item]), item))
            target = remaining.pop(index)
            total += distance(current, target)
            current = target
        return total

    def _fusion_skeleton_scan_cost(self, skeleton_index: int) -> float:
        pending = [
            channel
            for channel, state in self.channels.items()
            if state.status in {"UNKNOWN", "PARTIAL"}
        ]
        ordered = self._channel_order(pending)
        switches = sum(channel != previous for previous, channel in zip([self.current_measure_channel] + ordered, ordered))
        return len(ordered) * self.cfg.measure_time_s + switches * self.cfg.switch_time_s

    def _choose_fused_action(
        self,
        visited_skeleton: set[int],
        stalled: set[int],
    ) -> tuple[str, int, Point, float] | None:
        """在覆盖节点和服务节点之间做一次滚动选择。

        评分是“本动作代价 + 执行动作后的剩余开放路线代价”。覆盖节点始终进入
        剩余路线，因此服务动作不会把尚未覆盖的骨架点永久挤出计划。
        """

        actions: list[tuple[str, int, Point, float]] = []
        service_points = {
            channel: self._fusion_service_target(channel)
            for channel, state in self.channels.items()
            if state.status == "DETECTED" and channel not in stalled
        }
        service_points = {channel: point for channel, point in service_points.items() if point is not None}

        for index, point in enumerate(self.skeleton):
            if index in visited_skeleton:
                continue
            pending_count = sum(
                state.status in {"UNKNOWN", "PARTIAL"} for state in self.channels.values()
            )
            immediate = distance(self.position, point) / self.cfg.speed_mps + self._fusion_skeleton_scan_cost(index)
            remaining = [
                target
                for other, target in enumerate(self.skeleton)
                if other not in visited_skeleton and other != index
            ] + list(service_points.values())
            score = immediate + self._fusion_route_length(point, remaining) / self.cfg.speed_mps
            # 没有待测频道时，骨架点只承担移动覆盖义务，不额外奖励空测量。
            if pending_count == 0:
                score += 0.1
            actions.append(("skeleton", index, point, score))

        for channel, point in service_points.items():
            immediate = distance(self.position, point) / self.cfg.speed_mps
            immediate += self.cfg.measure_time_s if self.channels[channel].near_positions == [] else self.cfg.clear_success_time_s
            if self.channels[channel].near_positions == [] and channel != self.current_measure_channel:
                immediate += self.cfg.switch_time_s
            remaining = [
                target for other, target in enumerate(self.skeleton) if other not in visited_skeleton
            ] + [target for other, target in service_points.items() if other != channel]
            score = immediate + self._fusion_route_length(point, remaining) / self.cfg.speed_mps
            actions.append(("service", channel, point, score))

        return min(actions, key=lambda item: (item[3], item[0], item[1])) if actions else None

    def _visit_fused_skeleton(self, skeleton_index: int) -> None:
        point = self.skeleton[skeleton_index]
        candidates = [
            channel
            for channel, state in self.channels.items()
            if state.status in {"UNKNOWN", "PARTIAL"}
        ]
        for channel in self._channel_order(candidates):
            response = self.measure(point, channel, skeleton_index, role="coverage")
            if response.get("measure_result") == "near":
                if not self.clear(point, channel, "near_at_fused_skeleton"):
                    self.channels[channel].flags.append("NEAR_CLEAR_CONTRADICTION")
            if self.counters.N_c >= self.cfg.source_count_max:
                break
        if self.counters.N_c < self.cfg.source_count_max:
            self._take_opportunistic_measurements(point, skeleton_index)
        self.strategy_diagnostics["fused_skeleton_visits"] += 1
        self._record(
            "fused_skeleton_visit",
            skeleton_index=skeleton_index,
            position=point,
            measured_channels=self._channel_order(candidates),
        )

    def fused_rolling_plan(self) -> None:
        """覆盖与服务统一进入滚动规划，每次只执行一个可解释动作。"""

        self.phase = "coverage_scan"
        visited_skeleton: set[int] = set()
        stalled: set[int] = set()
        while self.counters.N_c < self.cfg.source_count_max:
            if len(visited_skeleton) == len(self.skeleton):
                self.phase = "post_scan"
            action = self._choose_fused_action(visited_skeleton, stalled)
            if action is None:
                break
            kind, identifier, point, score = action
            self.strategy_diagnostics["fusion_decisions"] += 1
            self._record(
                "fusion_decision",
                action=kind,
                identifier=identifier,
                target=point,
                objective_s=score,
                visited_skeleton=sorted(visited_skeleton),
            )
            if kind == "skeleton":
                self._visit_fused_skeleton(identifier)
                visited_skeleton.add(identifier)
                continue
            before = self.counters.L_m
            if not self._advance_channel_once(identifier, point):
                self.channels[identifier].flags.append("UNRESOLVED_AFTER_FUSED_ROLLING_ACTION")
                stalled.add(identifier)
            self.strategy_diagnostics["fused_service_actions"] += 1
            self.strategy_diagnostics["fused_service_distance_m"] += self.counters.L_m - before

        if len(visited_skeleton) == len(self.skeleton):
            self.phase = "post_scan"

    def _second_measurement_choices(self, observation: BearingObservation) -> list[Point]:
        choices: list[Point] = []
        for sign in (1.0, -1.0):
            direction = unit(observation.bearing_deg + sign * self.cfg.second_offset_deg)
            point = (
                observation.position[0] + self.cfg.second_baseline_m * direction[0],
                observation.position[1] + self.cfg.second_baseline_m * direction[1],
            )
            choices.append(project_to_target_disk(point, self.cfg.target_radius_m))
        return choices

    def _second_measurement_point(self, observation: BearingObservation) -> Point:
        choices = self._second_measurement_choices(observation)
        return min(choices, key=lambda point: distance(self.position, point))

    def _refinement_choices(self, state: ChannelState, center: Point) -> list[Point]:
        latest = state.observations[-1]
        normal = unit(latest.bearing_deg + 90.0)
        first = (center[0] + self.cfg.refinement_baseline_m * normal[0], center[1] + self.cfg.refinement_baseline_m * normal[1])
        second = (center[0] - self.cfg.refinement_baseline_m * normal[0], center[1] - self.cfg.refinement_baseline_m * normal[1])
        return [
            project_to_target_disk(first, self.cfg.target_radius_m),
            project_to_target_disk(second, self.cfg.target_radius_m),
        ]

    def _refinement_point(self, state: ChannelState, center: Point) -> Point:
        first, second = self._refinement_choices(state, center)
        return first if distance(self.position, first) <= distance(self.position, second) else second

    def _fallback_polygon(self, channel: int, polygon: np.ndarray) -> bool:
        state = self.channels[channel]
        if len(polygon) == 0:
            return False
        low = polygon.min(axis=0)
        high = polygon.max(axis=0)
        raw_points = grid_cover_rectangle((low[0], high[0], low[1], high[1]), self.cfg.fallback_cell_m)
        points = list(dict.fromkeys(project_to_target_disk(point, self.cfg.target_radius_m) for point in raw_points))
        if len(points) > self.cfg.max_fallback_clear_attempts:
            return False
        state.fallback_used = True
        for point in points:
            if self.clear(point, channel, "localized_box_cover"):
                return True
        return False

    def _fallback_wedge(self, channel: int) -> bool:
        state = self.channels[channel]
        if not state.observations:
            return False
        raw_points = wedge_fallback_grid(self.cfg, state.observations[0])
        points = list(dict.fromkeys(project_to_target_disk(point, self.cfg.target_radius_m) for point in raw_points))
        if len(points) > self.cfg.max_fallback_clear_attempts:
            state.flags.append("WEDGE_FALLBACK_EXCEEDS_CONFIGURED_LIMIT")
            return False
        state.fallback_used = True
        for point in points:
            if self.clear(point, channel, "first_wedge_cover"):
                return True
        return False

    def locate_and_clear(self, channel: int, preferred_service_point: Point | None = None) -> bool:
        state = self.channels[channel]
        if state.status == "CLEARED":
            return True
        if state.near_positions:
            if self.clear(state.near_positions[-1], channel, "near_retry"):
                return True
        if not state.observations:
            state.flags.append("DETECTED_WITHOUT_DIRECTION")
            return False

        if len(state.observations) == 1:
            second = preferred_service_point or self._second_measurement_point(state.observations[0])
            preferred_service_point = None
            response = self.measure(second, channel)
            if state.status == "CLEARED":
                return True
            if response.get("measure_result") == "near":
                return self.clear(second, channel, "near_at_second_point")
            if response.get("measure_result") == "no_signal":
                state.flags.append("GUARANTEED_SECOND_POINT_RETURNED_NO_SIGNAL")
                state.status = "DETECTED"
                return self._fallback_wedge(channel)

        for _ in range(self.cfg.max_refinements + 1):
            polygon = localization_polygon(self.cfg, state.observations)
            if len(polygon) == 0:
                state.flags.append("EMPTY_LOCALIZATION_REGION")
                break
            center_array, radius = minimum_enclosing_circle(polygon)
            center = (float(center_array[0]), float(center_array[1]))
            if radius <= self.cfg.safe_clear_radius_m:
                if self.clear(center, channel, "mec_safe"):
                    return True
                state.flags.append("SAFE_MEC_CLEAR_FAILED")
            if len(state.observations) - 2 >= self.cfg.max_refinements:
                return self._fallback_polygon(channel, polygon) or self._fallback_wedge(channel)
            point = preferred_service_point or self._refinement_point(state, center)
            preferred_service_point = None
            response = self.measure(point, channel)
            if response.get("measure_result") == "near":
                return self.clear(point, channel, "near_at_refinement")
            if response.get("measure_result") == "no_signal":
                state.flags.append("REFINEMENT_NO_SIGNAL")
                break
        return self._fallback_wedge(channel)

    def _advance_channel_once(self, channel: int, preferred_service_point: Point) -> bool:
        """执行一个滚动服务动作；返回是否取得可继续规划的确定进展。"""

        state = self.channels[channel]
        if state.status == "CLEARED":
            return True
        if state.near_positions:
            if self.clear(state.near_positions[-1], channel, "near_rolling"):
                return True
            state.flags.append("NEAR_CLEAR_CONTRADICTION")
            return self._fallback_wedge(channel)
        if not state.observations:
            state.flags.append("DETECTED_WITHOUT_DIRECTION")
            return False

        polygon = localization_polygon(self.cfg, state.observations)
        if len(polygon) == 0:
            if "EMPTY_LOCALIZATION_REGION" not in state.flags:
                state.flags.append("EMPTY_LOCALIZATION_REGION")
            return self._fallback_wedge(channel)
        center_array, radius = minimum_enclosing_circle(polygon)
        center = (float(center_array[0]), float(center_array[1]))
        state.localization_center = center
        state.localization_radius_m = float(radius)
        if len(state.observations) >= 2 and radius <= self.cfg.safe_clear_radius_m:
            if self.clear(center, channel, "mec_safe_rolling"):
                return True
            state.flags.append("SAFE_MEC_CLEAR_FAILED")
            return self._fallback_polygon(channel, polygon) or self._fallback_wedge(channel)

        previous_count = len(state.observations)
        if previous_count >= 2 and previous_count - 2 >= self.cfg.max_refinements:
            return self._fallback_polygon(channel, polygon) or self._fallback_wedge(channel)
        response = self.measure(preferred_service_point, channel, role="rolling_localization")
        if response.get("measure_result") == "near":
            return self.clear(preferred_service_point, channel, "near_at_rolling_measurement")
        if response.get("measure_result") == "no_signal":
            state.flags.append(
                "GUARANTEED_SECOND_POINT_RETURNED_NO_SIGNAL"
                if previous_count == 1
                else "REFINEMENT_NO_SIGNAL"
            )
            state.status = "DETECTED"
            return self._fallback_wedge(channel)
        self._refresh_localization(state)
        return True

    def _next_service_options(self, channel: int) -> list[Point]:
        """返回该频道当前服务动作的安全候选位置。"""

        state = self.channels[channel]
        if state.status != "DETECTED":
            return []
        if state.near_positions:
            return [state.near_positions[-1]]
        safe = self._safe_clear_point(state)
        if safe is not None:
            return [safe]
        if len(state.observations) == 1:
            return self._second_measurement_choices(state.observations[0])
        if state.observations:
            polygon = localization_polygon(self.cfg, state.observations)
            if len(polygon) == 0:
                return []
            center_array, _ = minimum_enclosing_circle(polygon)
            center = (float(center_array[0]), float(center_array[1]))
            return self._refinement_choices(state, center)
        return []

    def completion(self) -> tuple[bool, str]:
        if self.counters.N_c >= self.cfg.source_count_max:
            return True, "source_upper_bound_reached"
        if all(state.status in TERMINAL_STATES for state in self.channels.values()):
            if self.counters.N_c < self.cfg.source_count_min:
                return False, "resolved_but_below_source_lower_bound"
            return True, "all_channels_cleared_or_absent"
        return False, "unresolved_channels"

    def start_session(self) -> dict[str, Any]:
        """进入模拟器并初始化本局时钟，供自动与手动入口共同使用。"""

        response = self.simulator.enter()
        self.virtual_start_s = float(response.get("virtual_time_s", 0.0))
        self.virtual_time_s = self.virtual_start_s
        remaining = response.get("remaining_real_duration_s")
        if not isinstance(remaining, (int, float)) or isinstance(remaining, bool):
            raise SimulatorClientError("/enter 响应缺少数值型 remaining_real_duration_s")
        self.real_started = time.monotonic()
        self.real_deadline = self.real_started + float(remaining)
        self._record("enter", remaining_real_duration_s=float(remaining))
        return response

    def end_session(self) -> dict[str, Any]:
        """主动退出模拟器；/exit 不改变本地公式时间。"""

        response = self.simulator.exit()
        self.virtual_time_s = float(response.get("virtual_time_s", self.virtual_time_s))
        self._record("exit", exit_reason=response.get("exit_reason"))
        return response

    def total_time_s(self) -> float:
        c = self.counters
        return (
            c.L_m / self.cfg.speed_mps
            + c.N_s * self.cfg.switch_time_s
            + c.N_m * self.cfg.measure_time_s
            + c.N_c * self.cfg.clear_success_time_s
            + c.N_f * self.cfg.clear_fail_time_s
        )

    def summary(
        self,
        run_status: str,
        strategy: str = "certified_skeleton7_fused_rolling_planner_v4",
    ) -> dict[str, Any]:
        certificate, reason = self.completion()
        total = self.total_time_s()
        cleared = self.counters.N_c
        virtual_elapsed = self.virtual_time_s - self.virtual_start_s
        time_error = total - virtual_elapsed
        resolved = sum(state.status in TERMINAL_STATES for state in self.channels.values())
        return {
            "strategy": strategy,
            "run_status": run_status,
            "completion_certificate": certificate,
            "completion_reason": reason,
            "cleared_count": cleared,
            "inferred_total_sources": cleared if certificate else None,
            "clear_ratio": 1.0 if certificate else None,
            "resolved_channel_count": resolved,
            "absent_count": sum(state.status == "ABSENT" for state in self.channels.values()),
            "total_time_s": total,
            "average_time_per_cleared_s": total / cleared if cleared else None,
            "time_breakdown": {
                "L_m": self.counters.L_m,
                "travel_s": self.counters.L_m / self.cfg.speed_mps,
                "N_s": self.counters.N_s,
                "switch_s": self.counters.N_s * self.cfg.switch_time_s,
                "N_m": self.counters.N_m,
                "measure_s": self.counters.N_m * self.cfg.measure_time_s,
                "N_c": self.counters.N_c,
                "clear_success_s": self.counters.N_c * self.cfg.clear_success_time_s,
                "N_f": self.counters.N_f,
                "clear_fail_s": self.counters.N_f * self.cfg.clear_fail_time_s,
            },
            "simulator_virtual_elapsed_s": virtual_elapsed,
            "time_reconciliation_error_s": time_error,
            "real_elapsed_s": time.monotonic() - self.real_started,
            "diagnostics": {
                "detour_ratio_vs_7200m_skeleton": self.counters.L_m / max(1e-12, skeleton_route_length(self.skeleton)),
                "average_measures_per_cleared": self.counters.N_m / cleared if cleared else None,
                "fallback_channels": [state.channel for state in self.channels.values() if state.fallback_used],
                "no_signal_skeleton_records": sum(len(state.no_signal_skeleton_indices) for state in self.channels.values()),
                "phase_distance_m": dict(self.phase_distance_m),
                "localized_during_scan_count": sum(state.localized_during_scan for state in self.channels.values()),
                **self.strategy_diagnostics,
            },
            "coverage": coverage_report(self.cfg),
            "second_measurement_guarantee": guaranteed_second_measurement(self.cfg),
            "channel_states": [
                {
                    "channel": state.channel,
                    "status": state.status,
                    "no_signal_skeleton_indices": sorted(state.no_signal_skeleton_indices),
                    "observation_count": len(state.observations),
                    "clear_attempts": state.clear_attempts,
                    "fallback_used": state.fallback_used,
                    "cleared_at": state.cleared_at,
                    "localization_center": state.localization_center,
                    "localization_radius_m": state.localization_radius_m,
                    "opportunistic_attempts": state.opportunistic_attempts,
                    "opportunistic_directions": state.opportunistic_directions,
                    "localized_during_scan": state.localized_during_scan,
                    "scheduled_clear_edge": state.scheduled_clear_edge,
                    "flags": state.flags,
                }
                for state in self.channels.values()
            ],
            "trajectory": self.trajectory,
            "flags": self.flags,
            "limitations": [
                "机会补测评分、顺路清除阈值与2-opt路线均为可解释启发式，不宣称连续全局时间最优。",
                "概率脊线思想被改写为方向约束可行区域；程序不使用模拟器未提供的RSS或先验概率。",
                "完成证书使用官方附件给出的每个频道最多对应一个干扰源条件。",
                "未连接模拟器时只能验证几何、状态机和合成闭环，不能给出真实演练统计。",
            ],
        }

    def run(self) -> dict[str, Any]:
        entered = False
        safe_to_send_exit = True
        run_status = "failed_before_enter"
        try:
            self.start_session()
            entered = True
            run_status = "running"
            self.fused_rolling_plan()
            run_status = "completed" if self.completion()[0] else "incomplete"
        except RuntimeBudgetExceeded as exc:
            self.flags.append(str(exc))
            run_status = "real_time_reserve_reached"
        except UncertainActionError as exc:
            self.flags.append(f"UNCERTAIN_ACTION: {exc}")
            safe_to_send_exit = False
            run_status = "uncertain_action_stop"
        except (SimulatorClientError, KeyError, ValueError) as exc:
            self.flags.append(f"{type(exc).__name__}: {exc}")
            run_status = "client_or_protocol_error"
        except KeyboardInterrupt:
            self.flags.append("OPERATOR_INTERRUPTED")
            run_status = "operator_interrupted"
        except Exception as exc:  # 实时运行必须保留报告，但明确标为程序异常。
            self.flags.append(f"UNEXPECTED_{type(exc).__name__}: {exc}")
            run_status = "unexpected_error"
        finally:
            if entered and safe_to_send_exit:
                try:
                    self.end_session()
                except Exception as exc:  # 不掩盖主流程结果，但必须记录。
                    self.flags.append(f"exit_failed: {exc!r}")
            elif entered:
                self.flags.append("EXIT_SKIPPED_BECAUSE_LAST_ACTION_OUTCOME_IS_UNCERTAIN")
        return self.summary(run_status)


def coverage_report(cfg: Config) -> dict[str, Any]:
    worst = exact_skeleton_covering_radius(cfg.target_radius_m, cfg.skeleton_ring_radius_m)
    return {
        "skeleton_points": build_skeleton(cfg.skeleton_ring_radius_m),
        "point_count": 7,
        "route_length_m": skeleton_route_length(build_skeleton(cfg.skeleton_ring_radius_m)),
        "exact_worst_nearest_distance_m": worst,
        "guarantee_receive_m": cfg.guarantee_receive_m,
        "margin_m": cfg.guarantee_receive_m - worst,
        "certified": worst <= cfg.guarantee_receive_m + 1e-12,
        "proof": "six equal angular sectors; maximize min(center distance, nearest ring-point distance) at transition or target boundary",
    }


def write_result(result: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"problem3_report_{stamp}.json"
    csv_path = output_dir / f"problem3_summary_{stamp}.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    breakdown = result["time_breakdown"]
    row = {
        "strategy": result["strategy"],
        "run_status": result["run_status"],
        "cleared_count": result["cleared_count"],
        "clear_ratio": result["clear_ratio"],
        "total_time_s": result["total_time_s"],
        "average_time_per_cleared_s": result["average_time_per_cleared_s"],
        "L_m": breakdown["L_m"],
        "N_s": breakdown["N_s"],
        "N_m": breakdown["N_m"],
        "N_c": breakdown["N_c"],
        "N_f": breakdown["N_f"],
        "certificate": result["completion_certificate"],
        "completion_reason": result["completion_reason"],
        "time_reconciliation_error_s": result["time_reconciliation_error_s"],
    }
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return json_path, csv_path


def load_config(path: Path | None) -> Config:
    if path is None:
        cfg = Config()
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("配置文件顶层必须是 JSON 对象")
        cfg = Config.from_mapping(raw)
    cfg.validate()
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="问题3保证覆盖与定位清除求解器")
    parser.add_argument("--config", type=Path, help="JSON 参数文件；省略时使用内置默认值")
    parser.add_argument("--check-coverage", action="store_true", help="仅输出离线覆盖与第二测点保证检查")
    parser.add_argument("--robot-id", help="显式提供后才连接本机官方模拟器")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--output-dir", type=Path, default=Path("competition_b/results"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    offline = {"config": asdict(cfg), "coverage": coverage_report(cfg), "second_measurement_guarantee": guaranteed_second_measurement(cfg)}
    if args.check_coverage or not args.robot_id:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        check_path = args.output_dir / "problem3_offline_check.json"
        check_path.write_text(json.dumps(offline, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({**offline, "offline_report": str(check_path.resolve())}, ensure_ascii=False, indent=2))
        if not args.robot_id and not args.check_coverage:
            print("未提供 --robot-id：仅执行离线检查，未连接模拟器。", file=sys.stderr)
        return 0 if offline["coverage"]["certified"] and offline["second_measurement_guarantee"]["passed"] else 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = ActionLogger(Path("competition_b/logs") / f"problem3_actions_{stamp}.jsonl")
    client = SimulatorClient(base_url=args.base_url, robot_id=str(args.robot_id), logger=logger)
    result = Problem3Solver(cfg, client, logger).run()
    json_path, csv_path = write_result(result, args.output_dir)
    print(json.dumps({"report": str(json_path.resolve()), "summary": str(csv_path.resolve()), "certificate": result["completion_certificate"]}, ensure_ascii=False))
    return 0 if result["completion_certificate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
