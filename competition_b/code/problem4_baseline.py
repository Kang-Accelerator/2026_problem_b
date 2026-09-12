#!/usr/bin/env python3
"""问题4求解器v1.5：28点连续认证双环与边界认证清除。

本版完整保留v1.4的28点连续证书与默认路线，并增加一项可回退优化：

1. 使用“原点+966米内环9点+1829米外环18点”的28点方向骨架；
2. 原点与内环10点构成快速距离覆盖层，随后只补扫外环18点；
3. `no_signal` 只按“距离超界或背向”处理，不删除 1000 米圆盘；
4. 发现方向后复用问题2/问题3的保守定位与清除流程；
5. 二次测点使用 ``b = ratio * d_est`` 和侧向角；该规则只控制风险，
   不声称单个侧向点对任意未知朝向都保证接收；
6. 覆盖路线按连续内外环顺序承诺，只允许零移动批量补测或低绕行服务插入；
7. 代码保留局部三角、主轴候选和跨频道共享补测用于消融，默认关闭；
8. 常规18米安全清除未触发时，可对保守定位多边形逐顶点验证；仅当其
   MEC圆心到全部顶点不超过19.5米时，才使用20米物理清除半径提前清除；
9. 原点方向驱动的同点集顺/逆时针前缀选择仅作为实验消融保留，因离线
   配对存在退化，默认关闭；
10. 所有频道状态只由真实反馈推进，启发式评分不作为安全证书。

本文件不实现或模拟官方模拟器。没有模拟器时只能执行几何/覆盖自检。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

try:
    from .problem3_solver import (
        BearingObservation,
        Config as Problem3Config,
        Problem3Solver,
        RuntimeBudgetExceeded,
        SimulatorClientError,
        UncertainActionError,
        build_skeleton,
        distance,
        localization_polygon,
        minimum_enclosing_circle,
        plan_open_route,
        project_to_target_disk,
        unit,
    )
    from ..action_logger import ActionLogger
    from ..simulator_client import SimulatorClient
except ImportError:  # 支持直接运行 competition_b/code 下的文件。
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root.parent) not in sys.path:
        sys.path.insert(0, str(project_root.parent))
    from competition_b.code.problem3_solver import (
        BearingObservation,
        Config as Problem3Config,
        Problem3Solver,
        RuntimeBudgetExceeded,
        SimulatorClientError,
        UncertainActionError,
        build_skeleton,
        distance,
        localization_polygon,
        minimum_enclosing_circle,
        plan_open_route,
        project_to_target_disk,
        unit,
    )
    from competition_b.action_logger import ActionLogger
    from competition_b.simulator_client import SimulatorClient


Point = tuple[float, float]


class SimulatorLike(Protocol):
    def enter(self) -> dict[str, Any]: ...
    def measure(self, x: float, y: float, channel: int) -> dict[str, Any]: ...
    def clear(self, x: float, y: float, channel: int) -> dict[str, Any]: ...
    def exit(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Problem4Config:
    """问题4基线参数。"""

    target_radius_m: float = 1800.0
    guarantee_receive_m: float = 1000.0
    max_receive_m: float = 1500.0
    bearing_error_deg: float = 1.0
    near_radius_m: float = 5.0
    clear_radius_m: float = 20.0
    safe_clear_radius_m: float = 18.0
    speed_mps: float = 5.0
    switch_time_s: float = 1.0
    measure_time_s: float = 5.0
    clear_success_time_s: float = 5.0
    clear_fail_time_s: float = 3.0
    channel_count: int = 20
    source_count_min: int = 10
    source_count_max: int = 16

    phase1_ring_radius_m: float = 1200.0
    direction_inner_count: int = 9
    direction_inner_radius_m: float = 966.0
    direction_outer_count: int = 18
    direction_outer_radius_m: float = 1829.0
    route_two_opt_passes: int = 8
    coverage_grid_step_m: float = 20.0
    # 默认环形三角剖分的最大边为975m，保留25m接收余量。
    coverage_required_margin_m: float = 20.0

    second_distance_estimate_m: float = 700.0
    second_baseline_ratio: float = 0.6
    second_offset_deg: float = 35.0
    refinement_baseline_m: float = 60.0
    max_refinements: int = 3
    fallback_cell_m: float = 25.0
    max_fallback_clear_attempts: int = 500
    polygon_edges: int = 256
    reserve_real_s: float = 90.0
    opportunistic_max_per_skeleton: int = 4
    opportunistic_min_baseline_m: float = 300.0
    opportunistic_min_cross_angle_deg: float = 20.0
    opportunistic_max_center_distance_m: float = 1500.0
    future_batch_max_attempts_per_channel: int = 2
    future_batch_min_expected_saved_s: float = 20.0
    adaptive_prefix_enabled: bool = False
    adaptive_prefix_min_origin_directions: int = 4
    adaptive_prefix_min_estimated_saved_s: float = 10.0
    verified_boundary_clear_enabled: bool = True
    verified_boundary_clear_margin_m: float = 0.5
    cautious_local_enabled: bool = False
    local_triangle_min_circumradius_m: float = 120.0
    local_triangle_margin_m: float = 10.0
    local_triangle_max_region_radius_m: float = 320.0
    local_triangle_orientation_trials: int = 6
    cautious_axis_offset_ratio: float = 1.5
    cautious_axis_max_offset_m: float = 400.0
    shared_service_enabled: bool = False
    shared_service_max_per_stop: int = 2
    shared_service_max_attempts_per_channel: int = 2
    shared_service_min_saved_s: float = 60.0
    shared_service_min_cross_angle_deg: float = 30.0
    committed_detour_ratio_max: float = 0.20
    committed_detour_abs_max_m: float = 250.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Problem4Config":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"未知配置字段：{unknown}")
        cfg = cls(**{key: raw[key] for key in raw})
        cfg.validate()
        return cfg

    def validate(self) -> None:
        positive = (
            self.target_radius_m,
            self.guarantee_receive_m,
            self.max_receive_m,
            self.clear_radius_m,
            self.speed_mps,
            self.direction_inner_radius_m,
            self.direction_outer_radius_m,
            self.coverage_grid_step_m,
            self.fallback_cell_m,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("半径、网格和速度必须是正有限数")
        if not 1 <= self.source_count_min <= self.source_count_max <= self.channel_count:
            raise ValueError("源数量上下界必须位于频道范围内")
        if self.guarantee_receive_m > self.max_receive_m:
            raise ValueError("保证接收半径不得大于最大接收半径")
        if self.direction_inner_count < 3:
            raise ValueError("direction_inner_count 至少为3")
        if self.direction_outer_count != 2 * self.direction_inner_count:
            raise ValueError("direction_outer_count 必须等于2*direction_inner_count")
        if self.fallback_cell_m / math.sqrt(2.0) > self.clear_radius_m + 1e-12:
            raise ValueError("光学兜底格子不能保证落在清除半径内")
        if not 0.0 < self.safe_clear_radius_m <= self.clear_radius_m:
            raise ValueError("safe_clear_radius_m 必须位于 (0, clear_radius_m]")
        if not 0.0 < self.second_baseline_ratio <= 1.0:
            raise ValueError("second_baseline_ratio 必须位于 (0, 1]")
        if self.coverage_required_margin_m < 0.0:
            raise ValueError("coverage_required_margin_m 不得为负")
        if self.opportunistic_max_per_skeleton < 0:
            raise ValueError("opportunistic_max_per_skeleton 不得为负")
        if self.opportunistic_min_baseline_m < 0.0:
            raise ValueError("opportunistic_min_baseline_m 不得为负")
        if not 0.0 <= self.opportunistic_min_cross_angle_deg <= 90.0:
            raise ValueError("opportunistic_min_cross_angle_deg 必须位于[0,90]")
        if self.opportunistic_max_center_distance_m <= 0.0:
            raise ValueError("opportunistic_max_center_distance_m 必须为正")
        if self.future_batch_max_attempts_per_channel < 0:
            raise ValueError("future_batch_max_attempts_per_channel 不得为负")
        if self.future_batch_min_expected_saved_s < 0.0:
            raise ValueError("future_batch_min_expected_saved_s 不得为负")
        if self.adaptive_prefix_min_origin_directions < 1:
            raise ValueError("adaptive_prefix_min_origin_directions 至少为1")
        if self.adaptive_prefix_min_estimated_saved_s < 0.0:
            raise ValueError("adaptive_prefix_min_estimated_saved_s 不得为负")
        if not 0.0 < self.verified_boundary_clear_margin_m < self.clear_radius_m:
            raise ValueError("verified_boundary_clear_margin_m 必须位于(0, clear_radius_m)")
        if self.local_triangle_min_circumradius_m <= 0.0:
            raise ValueError("local_triangle_min_circumradius_m 必须为正")
        if self.local_triangle_margin_m < 0.0:
            raise ValueError("local_triangle_margin_m 不得为负")
        if self.local_triangle_max_region_radius_m <= 0.0:
            raise ValueError("local_triangle_max_region_radius_m 必须为正")
        if self.local_triangle_orientation_trials < 1:
            raise ValueError("local_triangle_orientation_trials 至少为1")
        if self.cautious_axis_offset_ratio <= 0.0 or self.cautious_axis_max_offset_m <= 0.0:
            raise ValueError("谨慎主轴偏移参数必须为正")
        if self.shared_service_max_per_stop < 0 or self.shared_service_max_attempts_per_channel < 0:
            raise ValueError("共享停靠次数上限不得为负")
        if self.shared_service_min_saved_s < 0.0:
            raise ValueError("shared_service_min_saved_s 不得为负")
        if not 0.0 <= self.shared_service_min_cross_angle_deg <= 90.0:
            raise ValueError("shared_service_min_cross_angle_deg 必须位于[0,90]")
        if not 0.0 <= self.committed_detour_ratio_max <= 1.0:
            raise ValueError("committed_detour_ratio_max 必须位于[0,1]")
        if self.committed_detour_abs_max_m < 0.0:
            raise ValueError("committed_detour_abs_max_m 不得为负")
        certificate = continuous_ring_certificate(self)
        if not certificate["certified"]:
            raise ValueError(f"环形方向骨架未通过连续证书：{certificate}")

    def to_problem3_config(self) -> Problem3Config:
        """把问题4基线参数映射到可复用的问题3几何与状态机参数。"""

        return Problem3Config(
            target_radius_m=self.target_radius_m,
            guarantee_receive_m=self.guarantee_receive_m,
            max_receive_m=self.max_receive_m,
            bearing_error_deg=self.bearing_error_deg,
            near_radius_m=self.near_radius_m,
            clear_radius_m=self.clear_radius_m,
            safe_clear_radius_m=self.safe_clear_radius_m,
            speed_mps=self.speed_mps,
            switch_time_s=self.switch_time_s,
            measure_time_s=self.measure_time_s,
            clear_success_time_s=self.clear_success_time_s,
            clear_fail_time_s=self.clear_fail_time_s,
            channel_count=self.channel_count,
            source_count_min=self.source_count_min,
            source_count_max=self.source_count_max,
            skeleton_ring_radius_m=self.phase1_ring_radius_m,
            second_baseline_m=self.second_distance_estimate_m * self.second_baseline_ratio,
            second_offset_deg=self.second_offset_deg,
            refinement_baseline_m=self.refinement_baseline_m,
            max_refinements=self.max_refinements,
            fallback_cell_m=self.fallback_cell_m,
            max_fallback_clear_attempts=self.max_fallback_clear_attempts,
            reserve_real_s=self.reserve_real_s,
            polygon_edges=self.polygon_edges,
            opportunistic_max_per_skeleton=self.opportunistic_max_per_skeleton,
            opportunistic_min_baseline_m=self.opportunistic_min_baseline_m,
            opportunistic_min_cross_angle_deg=self.opportunistic_min_cross_angle_deg,
            opportunistic_max_center_distance_m=self.opportunistic_max_center_distance_m,
            inline_clear_detour_ratio_max=self.committed_detour_ratio_max,
            inline_clear_detour_abs_max_m=self.committed_detour_abs_max_m,
            route_two_opt_passes=self.route_two_opt_passes,
        )


def _regular_ring(count: int, radius_m: float) -> list[Point]:
    return [
        (
            float(radius_m * math.cos(2.0 * math.pi * index / count)),
            float(radius_m * math.sin(2.0 * math.pi * index / count)),
        )
        for index in range(count)
    ]


def build_direction_skeleton_ordered(
    cfg: Problem4Config,
    inner_start_index: int = 0,
    inner_step: int = 1,
) -> list[Point]:
    """按旋转起点和方向构造同一双环点集的等长连续路线。"""

    if inner_step not in {-1, 1}:
        raise ValueError("inner_step 必须为-1或1")
    inner = _regular_ring(cfg.direction_inner_count, cfg.direction_inner_radius_m)
    outer = _regular_ring(cfg.direction_outer_count, cfg.direction_outer_radius_m)
    n = cfg.direction_inner_count
    m = cfg.direction_outer_count
    start = inner_start_index % n
    inner_order = [(start + inner_step * offset) % n for offset in range(n)]
    last_inner = inner_order[-1]
    outer_start = (2 * last_inner) % m
    outer_order = [
        (outer_start - inner_step * offset) % m for offset in range(m)
    ]
    return (
        [(0.0, 0.0)]
        + [inner[index] for index in inner_order]
        + [outer[index] for index in outer_order]
    )


def build_direction_skeleton(cfg: Problem4Config | None = None) -> list[Point]:
    """构造默认双环连续巡航顺序。"""

    return build_direction_skeleton_ordered(cfg or Problem4Config())


def build_fast_nested_skeleton(cfg: Problem4Config) -> list[Point]:
    """构造原点加内环的快速距离覆盖子集。"""

    return [(0.0, 0.0)] + _regular_ring(
        cfg.direction_inner_count,
        cfg.direction_inner_radius_m,
    )


def nested_skeleton_certificate(cfg: Problem4Config) -> dict[str, Any]:
    """证明快速子骨架距离覆盖，且其证据可直接累计到完整方向骨架。"""

    fast = build_fast_nested_skeleton(cfg)
    full = build_direction_skeleton(cfg)
    half_angle = math.pi / cfg.direction_inner_count
    radial_crossing = cfg.direction_inner_radius_m / (2.0 * math.cos(half_angle))
    boundary_midpoint = math.sqrt(
        cfg.target_radius_m**2
        + cfg.direction_inner_radius_m**2
        - 2.0
        * cfg.target_radius_m
        * cfg.direction_inner_radius_m
        * math.cos(half_angle)
    )
    worst_bound = max(radial_crossing, boundary_midpoint)
    return {
        "fast_points": len(fast),
        "full_points": len(full),
        "remaining_points": len(set(full) - set(fast)),
        "is_nested_subset": set(fast).issubset(set(full)),
        "continuous_distance_bound_m": worst_bound,
        "distance_margin_m": cfg.guarantee_receive_m - worst_bound,
        "distance_coverage_certified": worst_bound <= cfg.guarantee_receive_m + 1e-12,
        "proof": "choose the origin or nearest inner-ring point; the angular separation is at most pi/n and the radial maximum occurs at the crossing or target boundary",
    }


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def convex_hull(points: Sequence[Point]) -> list[Point]:
    """二维点集凸包，返回逆时针顶点。"""

    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) <= 1:
        return unique
    lower: list[Point] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 1e-12:
            lower.pop()
        lower.append(point)
    upper: list[Point] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 1e-12:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def point_in_convex_hull(point: Point, points: Sequence[Point], tol: float = 1e-7) -> bool:
    """判断点是否位于二维点集的凸包内，允许边界容差。"""

    hull = convex_hull(points)
    if not hull:
        return False
    if len(hull) == 1:
        return distance(point, hull[0]) <= tol
    if len(hull) == 2:
        a, b = hull
        cross = abs(_cross(a, b, point))
        if cross > tol * max(1.0, distance(a, b)):
            return False
        dot = (point[0] - a[0]) * (point[0] - b[0]) + (point[1] - a[1]) * (point[1] - b[1])
        return dot <= tol
    signs: list[float] = []
    for index, left in enumerate(hull):
        right = hull[(index + 1) % len(hull)]
        value = _cross(left, right, point)
        scale = max(1.0, distance(left, right))
        signs.append(value / scale)
    return all(value >= -tol for value in signs) or all(value <= tol for value in signs)


def minimum_certificate_radius(
    point: Point,
    skeleton: Sequence[Point],
    max_radius: float,
) -> float | None:
    """返回使 point 落入邻居凸包所需的最小半径。"""

    if not skeleton:
        return None

    def covered(radius: float) -> bool:
        nearby = [
            candidate
            for candidate in skeleton
            if distance(point, candidate) <= radius + 1e-9
        ]
        return point_in_convex_hull(point, nearby)

    high = max(max(distance(point, candidate) for candidate in skeleton), max_radius)
    if not covered(high):
        return None
    low = 0.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        if covered(middle):
            high = middle
        else:
            low = middle
    return high


def continuous_ring_certificate(cfg: Problem4Config) -> dict[str, Any]:
    """原点—双环三角剖分的连续方向覆盖充分证书。"""

    n = cfg.direction_inner_count
    inner = cfg.direction_inner_radius_m
    outer = cfg.direction_outer_radius_m
    inner_chord = 2.0 * inner * math.sin(math.pi / n)
    outer_chord = 2.0 * outer * math.sin(math.pi / (2 * n))
    radial_edge = outer - inner
    cross_edge = math.sqrt(
        inner**2 + outer**2 - 2.0 * inner * outer * math.cos(math.pi / n)
    )
    worst_bound = max(inner, inner_chord, outer_chord, radial_edge, cross_edge)
    margin = cfg.guarantee_receive_m - worst_bound
    outer_apothem = outer * math.cos(math.pi / cfg.direction_outer_count)
    contains_target = outer_apothem >= cfg.target_radius_m - 1e-12
    certified = contains_target and worst_bound <= cfg.guarantee_receive_m + 1e-12
    return {
        "method": "continuous_nested_ring_triangulation_sufficient_condition",
        "skeleton_points": len(build_direction_skeleton(cfg)),
        "inner_count": cfg.direction_inner_count,
        "inner_radius_m": inner,
        "outer_count": cfg.direction_outer_count,
        "outer_radius_m": outer,
        "outer_apothem_m": outer_apothem,
        "target_containment_margin_m": outer_apothem - cfg.target_radius_m,
        "triangle_edge_bounds_m": {
            "center_to_inner": inner,
            "inner_chord": inner_chord,
            "outer_chord": outer_chord,
            "radial": radial_edge,
            "cross": cross_edge,
        },
        "continuous_worst_distance_bound_m": worst_bound,
        "continuous_margin_m": margin,
        "required_margin_m": cfg.coverage_required_margin_m,
        "meets_required_margin": margin >= cfg.coverage_required_margin_m - 1e-12,
        "contains_target_disk": contains_target,
        "certified": certified and margin >= cfg.coverage_required_margin_m - 1e-12,
        "proof": "the outer regular polygon contains the target disk; each target point lies in a center/annular triangle whose three edges are no longer than the reported bound",
    }


def continuous_square_grid_certificate(cfg: Problem4Config) -> dict[str, Any]:
    """兼容旧调用名；v1.2实际返回环形三角剖分证书。"""

    return continuous_ring_certificate(cfg)


def coverage_report(cfg: Problem4Config) -> dict[str, Any]:
    """连续证书加离散加密诊断；离散结果不冒充连续证明。"""

    cfg.validate()
    skeleton = build_direction_skeleton(cfg)
    step = cfg.coverage_grid_step_m
    tested = 0
    passed = 0
    min_certificate_margin = math.inf
    worst_point: Point | None = None
    for iy in range(-int(cfg.target_radius_m // step), int(cfg.target_radius_m // step) + 1):
        y = iy * step
        for ix in range(-int(cfg.target_radius_m // step), int(cfg.target_radius_m // step) + 1):
            x = ix * step
            point = (float(x), float(y))
            if math.hypot(x, y) > cfg.target_radius_m + 1e-9:
                continue
            tested += 1
            nearby = [
                candidate
                for candidate in skeleton
                if distance(point, candidate) <= cfg.guarantee_receive_m + 1e-9
            ]
            required_radius = minimum_certificate_radius(
                point,
                skeleton,
                cfg.guarantee_receive_m,
            )
            if required_radius is None:
                continue
            margin = cfg.guarantee_receive_m - required_radius
            if margin < min_certificate_margin:
                min_certificate_margin = margin
                worst_point = point
            if point_in_convex_hull(point, nearby):
                passed += 1
    if tested == 0:
        raise ValueError("覆盖验证没有生成任何网格点")
    continuous = continuous_ring_certificate(cfg)
    nested = nested_skeleton_certificate(cfg)
    return {
        "skeleton_points": len(skeleton),
        "topology": (
            f"origin_plus_inner{cfg.direction_inner_count}"
            f"_outer{cfg.direction_outer_count}"
        ),
        "inner_radius_m": cfg.direction_inner_radius_m,
        "outer_radius_m": cfg.direction_outer_radius_m,
        "grid_step_m": step,
        "tested_points": tested,
        "passed_points": passed,
        "pass_rate": passed / tested,
        "sampled_min_certificate_margin_m": min_certificate_margin,
        "required_margin_m": cfg.coverage_required_margin_m,
        "sampled_passed": passed == tested,
        "sampled_worst_point": worst_point,
        "continuous_certificate": continuous,
        "nested_certificate": nested,
        "certified": (
            continuous["certified"]
            and nested["is_nested_subset"]
            and nested["distance_coverage_certified"]
            and passed == tested
        ),
    }


class Problem4BaselineSolver(Problem3Solver):
    """问题4连续认证双环融合求解器。"""

    def __init__(
        self,
        cfg: Problem4Config,
        simulator: SimulatorLike,
        event_logger: ActionLogger | None = None,
    ) -> None:
        cfg.validate()
        super().__init__(cfg.to_problem3_config(), simulator, event_logger)
        self.cfg4 = cfg
        self.full_direction_skeleton = build_direction_skeleton(cfg)
        self.phase1_skeleton = build_fast_nested_skeleton(cfg)
        fast_points = set(self.phase1_skeleton)
        self.phase2_skeleton = [
            point for point in self.full_direction_skeleton if point not in fast_points
        ]
        self.full_skeleton_index = {
            point: index for index, point in enumerate(self.full_direction_skeleton)
        }
        self.nested_coverage_certificate = nested_skeleton_certificate(cfg)
        if not self.nested_coverage_certificate["distance_coverage_certified"]:
            raise ValueError(f"快速子骨架未通过距离覆盖证书：{self.nested_coverage_certificate}")
        self.phase2_no_signal_count = {
            channel: 0 for channel in range(1, cfg.channel_count + 1)
        }
        self.coverage_measured_points: dict[int, set[Point]] = {
            channel: set() for channel in range(1, cfg.channel_count + 1)
        }
        self.coverage_no_signal_points: dict[int, set[Point]] = {
            channel: set() for channel in range(1, cfg.channel_count + 1)
        }
        # 兼容早期报告字段；集合同时包含快速内环层与外环补扫层。
        self.phase2_measured_points = self.coverage_measured_points
        self.phase2_no_signal_points = self.coverage_no_signal_points
        self.directional_no_signal_points: dict[int, list[Point]] = {
            channel: [] for channel in range(1, cfg.channel_count + 1)
        }
        self.directional_service_attempted_points: dict[int, set[Point]] = {
            channel: set() for channel in range(1, cfg.channel_count + 1)
        }
        self.directional_refinement_attempts: dict[int, int] = {
            channel: 0 for channel in range(1, cfg.channel_count + 1)
        }
        self.shared_service_attempts: dict[int, int] = {
            channel: 0 for channel in range(1, cfg.channel_count + 1)
        }
        self._local_triangle_signatures: set[tuple[int, tuple[Point, ...]]] = set()
        self._local_triangle_cache: dict[tuple[int, int], dict[str, Any] | None] = {}
        # A triangle is a joint reception certificate: for every feasible source
        # position and orientation, at least one of its three vertices receives.
        # Therefore a successful first bearing must not silently replace the
        # remaining vertices with a newly generated triangle.
        self._active_local_triangles: dict[int, dict[str, Any]] = {}
        self._coverage_known_channels: set[int] = set()
        self.directional_coverage_certificate = continuous_ring_certificate(cfg)
        if not self.directional_coverage_certificate["certified"]:
            raise ValueError(
                f"方向感知骨架未通过连续覆盖证书：{self.directional_coverage_certificate}"
            )
        # Problem3 的距离分阶段键不认识问题4阶段；显式扩展后才能正确归因路程。
        self.phase_distance_m = {
            "phase1": 0.0,
            "localization_phase1": 0.0,
            "phase2": 0.0,
            "fused_service": 0.0,
            "fallback": 0.0,
            "post_processing": 0.0,
            "other": 0.0,
        }
        self.phase1_hits = 0
        self.phase2_hits = 0
        self.visited_coverage_points: set[Point] = set()
        self.coverage_route = list(self.full_direction_skeleton)
        self.coverage_cursor = 0
        self.committed_edge_start: Point = (0.0, 0.0)
        self.committed_edge_path_m = 0.0
        self.strategy_diagnostics.update(
            {
                "nested_fusion_decisions": 0,
                "nested_fast_visits": 0,
                "nested_remaining_visits": 0,
                "directional_service_actions": 0,
                "directional_fallback_actions": 0,
                "committed_service_insertions": 0,
                "committed_detour_m": 0.0,
                "max_committed_edge_detour_m": 0.0,
                "future_batch_attempts": 0,
                "future_batch_directions": 0,
                "future_batch_no_signal": 0,
                "future_batch_estimated_saved_s": 0.0,
                "local_triangle_groups_built": 0,
                "local_triangle_groups_completed": 0,
                "local_triangle_probe_attempts": 0,
                "local_triangle_probe_directions": 0,
                "local_triangle_probe_no_signal": 0,
                "cautious_axis_refinement_uses": 0,
                "shared_service_attempts": 0,
                "shared_service_directions": 0,
                "shared_service_no_signal": 0,
                "shared_service_estimated_saved_s": 0.0,
                "adaptive_prefix_considered": False,
                "adaptive_prefix_replanned": False,
                "adaptive_prefix_origin_directions": 0,
                "adaptive_prefix_start_index": 0,
                "adaptive_prefix_step": 1,
                "adaptive_prefix_default_objective_s": None,
                "adaptive_prefix_selected_objective_s": None,
                "adaptive_prefix_estimated_saved_s": 0.0,
                "coverage_terminated_at_source_upper_bound": False,
                "coverage_visits_at_upper_bound": None,
                "coverage_prefix_history": [],
                "verified_boundary_clear_attempts": 0,
                "verified_boundary_clear_successes": 0,
                "verified_boundary_clear_failures": 0,
            }
        )

    def measure(
        self,
        point: Point,
        channel: int,
        skeleton_index: int | None = None,
        role: str = "standard",
    ) -> dict[str, Any]:
        """问题4允许在目标圆域外测量，必须执行并记录原始计划坐标。

        问题3把测量点投影进目标圆盘，这会破坏问题4边界朝外源所需的
        方向覆盖。这里保留问题3的协议、状态机和记账，只移除该投影。
        """

        self._guard_time()
        point = (float(point[0]), float(point[1]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("测量坐标必须为有限数")
        response = self.simulator.measure(point[0], point[1], int(channel))
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
        batch_role = role in {"opportunistic", "future_skeleton_batch"}
        if batch_role:
            state.opportunistic_attempts += 1
            self.strategy_diagnostics["opportunistic_attempts"] += 1
            if role == "future_skeleton_batch":
                self.strategy_diagnostics["future_batch_attempts"] += 1
        if result == "no_signal":
            if skeleton_index is not None:
                state.no_signal_skeleton_indices.add(skeleton_index)
            if state.status not in {"DETECTED", "CLEARED"}:
                state.status = "PARTIAL"
            if state.status == "DETECTED" or role.startswith("directional"):
                self.directional_no_signal_points[channel].append(point)
            if batch_role:
                self.strategy_diagnostics["opportunistic_no_signal"] += 1
                if role == "future_skeleton_batch":
                    self.strategy_diagnostics["future_batch_no_signal"] += 1
        elif result == "direction":
            bearing = response.get("svd_deg")
            if not isinstance(bearing, (int, float)) or isinstance(bearing, bool):
                raise SimulatorClientError("direction 响应缺少数值型 svd_deg")
            state.observations.append(BearingObservation(point, float(bearing), skeleton_index))
            state.status = "DETECTED"
            if batch_role:
                state.opportunistic_directions += 1
                self.strategy_diagnostics["opportunistic_directions"] += 1
                if role == "future_skeleton_batch":
                    self.strategy_diagnostics["future_batch_directions"] += 1
        elif result == "near":
            state.near_positions.append(point)
            state.status = "DETECTED"
        else:
            raise SimulatorClientError(f"未知 measure_result：{result!r}")
        self._record("measure", position=point, channel=channel, result=result, status=state.status, role=role)
        return response

    def _second_measurement_choices(self, observation: BearingObservation) -> list[Point]:
        """定向源第二次测点：短基线加左右侧前方；单点不作保证接收声明。"""

        baseline = self.cfg4.second_distance_estimate_m * self.cfg4.second_baseline_ratio
        choices: list[Point] = []
        for sign in (1.0, -1.0):
            direction = unit(observation.bearing_deg + sign * self.cfg4.second_offset_deg)
            choices.append(
                (
                    observation.position[0] + baseline * direction[0],
                    observation.position[1] + baseline * direction[1],
                )
            )
        return choices

    def locate_and_clear(self, channel: int, preferred_service_point: Point | None = None) -> bool:
        """定向源先尝试左右两个二测点；两侧均无信号再进入光学兜底。"""

        state = self.channels[channel]
        if state.status == "CLEARED" or state.near_positions or len(state.observations) != 1:
            return super().locate_and_clear(channel, preferred_service_point)

        choices = self._second_measurement_choices(state.observations[0])
        if preferred_service_point in choices:
            choices.remove(preferred_service_point)
            choices.insert(0, preferred_service_point)
        else:
            choices.sort(key=lambda candidate: (distance(self.position, candidate), candidate))

        for index, point in enumerate(choices, start=1):
            response = self.measure(point, channel, role=f"directional_second_side_{index}")
            result = response.get("measure_result")
            if result == "near":
                return self.clear(point, channel, "near_at_directional_second")
            if result == "direction":
                return super().locate_and_clear(channel)
            state.flags.append(f"DIRECTIONAL_SECOND_SIDE_{index}_NO_SIGNAL")

        state.status = "DETECTED"
        state.flags.append("DIRECTIONAL_SECOND_BOTH_SIDES_NO_SIGNAL")
        return self._fallback_wedge(channel)

    def _plan_route(self, points: Sequence[Point]) -> list[Point]:
        targets = {index: point for index, point in enumerate(points)}
        order = plan_open_route(self.position, targets, self.cfg4.route_two_opt_passes)
        return [points[index] for index in order]

    def _record_coverage_result(self, channel: int, point: Point, result: str) -> None:
        """记录真实执行的嵌套骨架证据；只允许完整认证骨架无信号判空。"""

        self.coverage_measured_points[channel].add(point)
        if result != "no_signal":
            return
        self.coverage_no_signal_points[channel].add(point)
        self.phase2_no_signal_count[channel] = len(self.coverage_no_signal_points[channel])
        expected = set(self.full_direction_skeleton)
        if expected.issubset(self.coverage_no_signal_points[channel]):
            self.channels[channel].status = "ABSENT"









    def _future_batch_score(
        self,
        channel: int,
        point: Point,
        skeleton_index: int,
    ) -> tuple[float, float] | None:
        """评价必经骨架点的零移动补测；只用于效率排序。"""

        state = self.channels[channel]
        if state.status != "DETECTED" or len(state.observations) != 1 or state.near_positions:
            return None
        if state.opportunistic_attempts >= self.cfg4.future_batch_max_attempts_per_channel:
            return None
        if any(
            observation.skeleton_index == skeleton_index
            or distance(observation.position, point) <= 1e-8
            for observation in state.observations
        ):
            return None
        if self._safe_clear_point(state) is not None:
            return None
        polygon = localization_polygon(self.cfg, state.observations)
        if len(polygon) == 0:
            return None
        center_array, _ = minimum_enclosing_circle(polygon)
        center = (float(center_array[0]), float(center_array[1]))
        if distance(point, center) > self.cfg4.opportunistic_max_center_distance_m:
            return None
        baseline = max(distance(point, observation.position) for observation in state.observations)
        if baseline < self.cfg4.opportunistic_min_baseline_m:
            return None
        cross_angle = max(
            self._acute_cross_angle(center, observation.position, point)
            for observation in state.observations
        )
        if cross_angle < self.cfg4.opportunistic_min_cross_angle_deg:
            return None
        service_options = self._directional_service_options(channel)
        measure_points = [
            option["point"] for option in service_options if option["action"] == "measure"
        ]
        if not measure_points:
            return None
        estimated_saved_s = min(distance(point, candidate) for candidate in measure_points) / self.cfg.speed_mps
        if estimated_saved_s < self.cfg4.future_batch_min_expected_saved_s:
            return None
        score = estimated_saved_s + 2.0 * cross_angle / 90.0
        return score, estimated_saved_s

    def _take_future_skeleton_measurements(self, point: Point, skeleton_index: int) -> None:
        """在已经到达的骨架坐标批量补测，不产生任何额外移动。"""

        ranked: list[tuple[float, int, float]] = []
        for channel in self.channels:
            result = self._future_batch_score(channel, point, skeleton_index)
            if result is not None:
                ranked.append((result[0], channel, result[1]))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        for score, channel, estimated_saved_s in ranked[: self.cfg4.opportunistic_max_per_skeleton]:
            before = self.position
            response = self.measure(
                point,
                channel,
                skeleton_index,
                role="future_skeleton_batch",
            )
            if distance(before, self.position) > 1e-9:
                raise AssertionError("未来骨架批量补测产生了非零移动")
            self.strategy_diagnostics["future_batch_estimated_saved_s"] += estimated_saved_s
            self._record(
                "future_skeleton_batch_selection",
                channel=channel,
                skeleton_index=skeleton_index,
                score=score,
                estimated_saved_s=estimated_saved_s,
            )
            if response.get("measure_result") == "near":
                if not self.clear(point, channel, "near_at_future_skeleton_batch"):
                    self.channels[channel].flags.append("FUTURE_BATCH_NEAR_CLEAR_FAILED")
            elif response.get("measure_result") == "direction":
                self._refresh_localization(self.channels[channel])
            if self.counters.N_c >= self.cfg.source_count_max:
                break

    def _visit_coverage_point(self, point: Point, phase_name: str) -> None:
        """在同一坐标批量扫描未决频道，并复用位置做机会补测。"""

        self.phase = phase_name
        skeleton_index = self.full_skeleton_index[point]
        candidates = [
            channel
            for channel, state in self.channels.items()
            if state.status in {"UNKNOWN", "PARTIAL"}
        ]
        for channel in self._channel_order(candidates):
            response = self.measure(point, channel, skeleton_index, role=phase_name)
            result = str(response.get("measure_result"))
            self._record_coverage_result(channel, point, result)
            if result == "near":
                if phase_name == "phase1":
                    self.phase1_hits += 1
                else:
                    self.phase2_hits += 1
                if not self.clear(point, channel, f"near_problem4_{phase_name}"):
                    self.channels[channel].flags.append(f"NEAR_{phase_name.upper()}_CLEAR_FAILED")
            elif result == "direction":
                if phase_name == "phase1":
                    self.phase1_hits += 1
                else:
                    self.phase2_hits += 1
            if self.counters.N_c >= self.cfg.source_count_max or self._known_source_count() >= self.cfg.source_count_max:
                break
        if self.counters.N_c < self.cfg.source_count_max:
            self._take_future_skeleton_measurements(point, skeleton_index)
        self._record(
            "nested_coverage_visit",
            position=point,
            skeleton_index=skeleton_index,
            stage=phase_name,
            measured_channels=self._channel_order(candidates),
        )

    def _phase1_scan(self) -> None:
        self.phase = "phase1"
        for point in self.phase1_skeleton:
            self._visit_coverage_point(point, "phase1")
            if self.counters.N_c >= self.cfg.source_count_max or self._known_source_count() >= self.cfg.source_count_max:
                return

    def _phase2_scan(self) -> None:
        pending = [
            channel
            for channel, state in self.channels.items()
            if state.status in {"UNKNOWN", "PARTIAL"}
        ]
        if not pending:
            return
        self.phase = "phase2"
        for point in self.phase2_skeleton:
            self._visit_coverage_point(point, "phase2")
            if self.counters.N_c >= self.cfg.source_count_max or self._known_source_count() >= self.cfg.source_count_max:
                return

    def _service_detected_channels(self, phase_name: str) -> None:
        self.phase = phase_name
        stalled: set[int] = set()
        while True:
            if self.counters.N_c >= self.cfg.source_count_max:
                return
            candidates: list[tuple[float, int, Point]] = []
            for channel, state in self.channels.items():
                if state.status != "DETECTED" or channel in stalled:
                    continue
                options = self._next_service_options(channel)
                if options:
                    point = min(options, key=lambda item: (distance(self.position, item), item))
                    candidates.append((distance(self.position, point), channel, point))
            if not candidates:
                return
            _, channel, point = min(candidates)
            state = self.channels[channel]
            if not self.locate_and_clear(channel, point):
                state.flags.append("PHASE4_BASELINE_LOCALIZATION_INCOMPLETE")
                stalled.add(channel)

    @staticmethod
    def _polygon_principal_axes(polygon: np.ndarray) -> tuple[Point, Point, float, float]:
        """返回保守多边形的长轴、短轴及对应离散方差；仅用于选点。"""

        if len(polygon) == 0:
            raise ValueError("空多边形没有主轴")
        centered = polygon - np.mean(polygon, axis=0)
        covariance = centered.T @ centered / max(1, len(polygon))
        values, vectors = np.linalg.eigh(covariance)
        major_array = vectors[:, int(np.argmax(values))]
        major = (float(major_array[0]), float(major_array[1]))
        if major[0] < -1e-12 or (abs(major[0]) <= 1e-12 and major[1] < 0.0):
            major = (-major[0], -major[1])
        minor = (-major[1], major[0])
        ordered = sorted((float(values[0]), float(values[1])), reverse=True)
        return major, minor, ordered[0], ordered[1]

    def _certified_local_triangle(self, channel: int) -> dict[str, Any] | None:
        """构造覆盖整个定位区域且在1000米保证距离内的局部等边三角形。"""

        if not self.cfg4.cautious_local_enabled:
            return None
        state = self.channels[channel]
        if len(state.observations) < 2:
            return None
        active = self._active_local_triangles.get(channel)
        if active is not None:
            attempted = self.directional_service_attempted_points[channel]
            if any(point not in attempted for point in active["vertices"]):
                return active
            self.strategy_diagnostics["local_triangle_groups_completed"] += 1
            del self._active_local_triangles[channel]
        cache_key = (channel, len(state.observations))
        if cache_key in self._local_triangle_cache:
            return self._local_triangle_cache[cache_key]
        polygon = localization_polygon(self.cfg, state.observations)
        if len(polygon) == 0:
            self._local_triangle_cache[cache_key] = None
            return None
        center_array, region_radius = minimum_enclosing_circle(polygon)
        center = (float(center_array[0]), float(center_array[1]))
        region_radius = float(region_radius)
        if region_radius > self.cfg4.local_triangle_max_region_radius_m + 1e-9:
            self._local_triangle_cache[cache_key] = None
            return None
        circumradius = max(
            self.cfg4.local_triangle_min_circumradius_m,
            2.0 * region_radius + self.cfg4.local_triangle_margin_m,
        )
        if circumradius + region_radius > self.cfg.guarantee_receive_m + 1e-9:
            self._local_triangle_cache[cache_key] = None
            return None
        major, minor, major_variance, minor_variance = self._polygon_principal_axes(polygon)
        base_angle = math.degrees(math.atan2(minor[1], minor[0]))
        trial_count = self.cfg4.local_triangle_orientation_trials
        offsets = [0.0]
        for index in range(1, trial_count):
            magnitude = 10.0 * ((index + 1) // 2)
            offsets.append(magnitude if index % 2 else -magnitude)
        best: tuple[float, tuple[Point, ...], float] | None = None
        for offset in offsets:
            vertices = tuple(
                (
                    center[0] + circumradius * math.cos(math.radians(base_angle + offset + 120.0 * index)),
                    center[1] + circumradius * math.sin(math.radians(base_angle + offset + 120.0 * index)),
                )
                for index in range(3)
            )
            if not all(
                point_in_convex_hull((float(row[0]), float(row[1])), vertices, tol=1e-6)
                for row in polygon
            ):
                continue
            max_distance = max(
                distance(vertex, row)
                for vertex in vertices
                for row in polygon
            )
            if max_distance > self.cfg.guarantee_receive_m + 1e-7:
                continue
            route = min(
                itertools.permutations(vertices),
                key=lambda order: sum(
                    distance(left, right)
                    for left, right in zip((self.position,) + order[:-1], order)
                ),
            )
            route_length = sum(
                distance(left, right)
                for left, right in zip((self.position,) + route[:-1], route)
            )
            candidate = (route_length, route, max_distance)
            if best is None or candidate[0] < best[0] - 1e-9:
                best = candidate
        if best is None:
            self._local_triangle_cache[cache_key] = None
            return None
        route_length, route, max_distance = best
        signature = (
            channel,
            tuple((round(point[0], 6), round(point[1], 6)) for point in route),
        )
        if signature not in self._local_triangle_signatures:
            self._local_triangle_signatures.add(signature)
            self.strategy_diagnostics["local_triangle_groups_built"] += 1
        result = {
            "vertices": list(route),
            "center": center,
            "region_radius_m": region_radius,
            "circumradius_m": circumradius,
            "max_vertex_to_region_m": max_distance,
            "major_axis": major,
            "minor_axis": minor,
            "major_variance_m2": major_variance,
            "minor_variance_m2": minor_variance,
            "route_length_m": route_length,
            "certificate": "polygon_inside_triangle_and_all_vertex_to_polygon_distances_le_1000",
        }
        self._local_triangle_cache[cache_key] = result
        self._active_local_triangles[channel] = result
        return result

    def _cautious_axis_choices(self, channel: int) -> list[Point]:
        """沿不确定区域长轴的法向给出左右两个闭式候选，并选择可保证距离者。"""

        if not self.cfg4.cautious_local_enabled:
            return []
        state = self.channels[channel]
        polygon = localization_polygon(self.cfg, state.observations)
        if len(polygon) == 0:
            return []
        center_array, region_radius = minimum_enclosing_circle(polygon)
        center = (float(center_array[0]), float(center_array[1]))
        _, minor, major_variance, _ = self._polygon_principal_axes(polygon)
        offset = max(
            self.cfg4.refinement_baseline_m,
            min(
                self.cfg4.cautious_axis_max_offset_m,
                self.cfg4.cautious_axis_offset_ratio * math.sqrt(max(0.0, major_variance)),
            ),
        )
        offset = min(offset, max(0.0, self.cfg.guarantee_receive_m - float(region_radius)))
        if offset <= 1e-9:
            return []
        choices = [
            (center[0] + sign * offset * minor[0], center[1] + sign * offset * minor[1])
            for sign in (1.0, -1.0)
        ]
        return [
            point
            for point in choices
            if max(distance(point, row) for row in polygon)
            <= self.cfg.guarantee_receive_m + 1e-7
        ]

    def _shared_service_score(self, channel: int, point: Point) -> tuple[float, float] | None:
        """评价服务停靠点上的额外频道测量；只允许零移动且要求全区域在1000米内。"""

        if not self.cfg4.shared_service_enabled:
            return None
        state = self.channels[channel]
        if state.status != "DETECTED" or not state.observations or state.near_positions:
            return None
        if self.shared_service_attempts[channel] >= self.cfg4.shared_service_max_attempts_per_channel:
            return None
        if any(distance(observation.position, point) <= 1e-8 for observation in state.observations):
            return None
        if self._safe_clear_point(state) is not None:
            return None
        polygon = localization_polygon(self.cfg, state.observations)
        if len(polygon) == 0 or max(distance(point, row) for row in polygon) > self.cfg.guarantee_receive_m + 1e-7:
            return None
        center_array, _ = minimum_enclosing_circle(polygon)
        center = (float(center_array[0]), float(center_array[1]))
        if max(distance(point, observation.position) for observation in state.observations) < self.cfg4.opportunistic_min_baseline_m:
            return None
        cross_angle = max(
            self._acute_cross_angle(center, observation.position, point)
            for observation in state.observations
        )
        if cross_angle < self.cfg4.shared_service_min_cross_angle_deg:
            return None
        options = self._directional_service_options(channel)
        measure_points = [option["point"] for option in options if option["action"] == "measure"]
        if not measure_points:
            return None
        estimated_saved_s = min(distance(point, candidate) for candidate in measure_points) / self.cfg.speed_mps
        if estimated_saved_s < self.cfg4.shared_service_min_saved_s:
            return None
        batch_cost = self.cfg.measure_time_s + (
            self.cfg.switch_time_s if channel != self.current_measure_channel else 0.0
        )
        return estimated_saved_s - batch_cost, estimated_saved_s

    def _take_shared_service_measurements(self, point: Point, owner_channel: int) -> None:
        """复用已经到达的服务坐标；每个响应均真实进入状态机。"""

        ranked: list[tuple[float, int, float]] = []
        for channel in self.channels:
            if channel == owner_channel:
                continue
            score = self._shared_service_score(channel, point)
            if score is not None and score[0] > 0.0:
                ranked.append((score[0], channel, score[1]))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        for score, channel, estimated_saved_s in ranked[: self.cfg4.shared_service_max_per_stop]:
            before = self.position
            self.shared_service_attempts[channel] += 1
            self.directional_service_attempted_points[channel].add(point)
            self.strategy_diagnostics["shared_service_attempts"] += 1
            response = self.measure(point, channel, role="shared_service_batch")
            if distance(before, self.position) > 1e-9:
                raise AssertionError("共享停靠补测产生了非零移动")
            self.strategy_diagnostics["shared_service_estimated_saved_s"] += estimated_saved_s
            result = response.get("measure_result")
            if result == "direction":
                self.strategy_diagnostics["shared_service_directions"] += 1
                self._refresh_localization(self.channels[channel])
            elif result == "near":
                if not self.clear(point, channel, "near_at_shared_service"):
                    self.channels[channel].flags.append("SHARED_SERVICE_NEAR_CLEAR_FAILED")
            else:
                self.strategy_diagnostics["shared_service_no_signal"] += 1
            self._record(
                "shared_service_selection",
                owner_channel=owner_channel,
                channel=channel,
                point=point,
                score_s=score,
                estimated_saved_s=estimated_saved_s,
                reception_semantics="distance_guaranteed_but_directional_front_halfplane_not_guaranteed",
            )
            if self.counters.N_c >= self.cfg.source_count_max:
                break

    def _verified_boundary_clear_point(self, state: ChannelState) -> Point | None:
        """以物理20米半径逐顶点认证，保留独立数值余量。"""

        if not self.cfg4.verified_boundary_clear_enabled or len(state.observations) < 2:
            return None
        if "PROBLEM4_VERIFIED_BOUNDARY_CLEAR_FAILED" in state.flags:
            # 若官方反馈与有限精度几何证书矛盾，不在承诺边期间重复清除；
            # 交回既有补测/兜底流程，并把矛盾保留在报告中。
            return None
        polygon = localization_polygon(self.cfg, state.observations)
        if len(polygon) == 0:
            return None
        center_array, radius = minimum_enclosing_circle(polygon)
        center = (float(center_array[0]), float(center_array[1]))
        certified_limit = (
            self.cfg.clear_radius_m - self.cfg4.verified_boundary_clear_margin_m
        )
        if float(radius) > certified_limit + 1e-9:
            return None
        if max(distance(center, row) for row in polygon) > certified_limit + 1e-7:
            return None
        return center

    def _directional_service_options(self, channel: int) -> list[dict[str, Any]]:
        """返回频道当前可执行的独立服务动作，不先压缩成最近的一个点。"""

        state = self.channels[channel]
        if state.status != "DETECTED":
            return []
        if state.near_positions:
            return [{"action": "clear", "point": state.near_positions[-1], "role": "near"}]
        safe = self._safe_clear_point(state)
        if safe is not None:
            return [{"action": "clear", "point": safe, "role": "mec_safe"}]
        verified_boundary = self._verified_boundary_clear_point(state)
        if verified_boundary is not None:
            return [
                {
                    "action": "clear",
                    "point": verified_boundary,
                    "role": "verified_boundary_clear",
                    "certified_radius_m": (
                        self.cfg.clear_radius_m
                        - self.cfg4.verified_boundary_clear_margin_m
                    ),
                }
            ]
        attempted = self.directional_service_attempted_points[channel]
        if len(state.observations) == 1:
            return [
                {"action": "measure", "point": point, "role": "directional_second"}
                for point in self._second_measurement_choices(state.observations[0])
                if point not in attempted
            ]
        if state.observations and self.directional_refinement_attempts[channel] < self.cfg.max_refinements:
            polygon = localization_polygon(self.cfg, state.observations)
            if len(polygon) == 0:
                return []
            triangle = self._certified_local_triangle(channel)
            if triangle is not None:
                available = [
                    point for point in triangle["vertices"] if point not in attempted
                ]
                if available:
                    return [
                        {
                            "action": "measure",
                            "point": point,
                            "role": "certified_local_triangle",
                            "triangle_certificate": triangle["certificate"],
                            "triangle_region_radius_m": triangle["region_radius_m"],
                            "triangle_max_distance_m": triangle["max_vertex_to_region_m"],
                        }
                        for point in available
                    ]
                if "CERTIFIED_LOCAL_TRIANGLE_EXHAUSTED" not in state.flags:
                    state.flags.append("CERTIFIED_LOCAL_TRIANGLE_EXHAUSTED")
                return []
            center_array, _ = minimum_enclosing_circle(polygon)
            center = (float(center_array[0]), float(center_array[1]))
            cautious = self._cautious_axis_choices(channel)
            if cautious:
                return [
                    {"action": "measure", "point": point, "role": "cautious_axis_refinement"}
                    for point in cautious
                    if point not in attempted
                ]
            return [
                {"action": "measure", "point": point, "role": "directional_refinement"}
                for point in self._refinement_choices(state, center)
                if point not in attempted
            ]
        return []

    def _prefix_route_objective_s(
        self,
        route: Sequence[Point],
        observations: Sequence[BearingObservation],
    ) -> float:
        """估计原点已发现频道取得第二方向的前缀到达总时间。"""

        phase1 = list(route[: 1 + self.cfg4.direction_inner_count])
        edges = list(zip(phase1, phase1[1:]))
        cumulative_to_start: list[float] = []
        accumulated = 0.0
        for start, end in edges:
            cumulative_to_start.append(accumulated)
            accumulated += distance(start, end)
        objective_s = 0.0
        for observation in observations:
            choices = self._second_measurement_choices(observation)
            feasible: list[float] = []
            for edge_index, (start, end) in enumerate(edges):
                direct = distance(start, end)
                limit = min(
                    self.cfg4.committed_detour_abs_max_m,
                    self.cfg4.committed_detour_ratio_max * direct,
                )
                for point in choices:
                    detour = distance(start, point) + distance(point, end) - direct
                    if detour <= limit + 1e-9:
                        feasible.append(
                            (
                                cumulative_to_start[edge_index]
                                + distance(start, point)
                            )
                            / self.cfg.speed_mps
                            + self.cfg.measure_time_s
                        )
            if feasible:
                objective_s += min(feasible)
                continue
            last = phase1[-1]
            objective_s += (
                accumulated + min(distance(last, point) for point in choices)
            ) / self.cfg.speed_mps + self.cfg.measure_time_s
        return objective_s

    def _adapt_coverage_prefix_after_origin(self) -> None:
        """只旋转/镜像同一认证点集，优化已有原点方向的早期二测机会。"""

        if not self.cfg4.adaptive_prefix_enabled:
            return
        observations = [
            state.observations[0]
            for state in self.channels.values()
            if len(state.observations) == 1
            and distance(state.observations[0].position, (0.0, 0.0)) <= 1e-8
        ]
        self.strategy_diagnostics["adaptive_prefix_considered"] = True
        self.strategy_diagnostics["adaptive_prefix_origin_directions"] = len(observations)
        if len(observations) < self.cfg4.adaptive_prefix_min_origin_directions:
            return
        candidates: list[tuple[float, int, int, list[Point]]] = []
        for start_index in (0,):
            for step in (1, -1):
                route = build_direction_skeleton_ordered(
                    self.cfg4,
                    inner_start_index=start_index,
                    inner_step=step,
                )
                objective_s = self._prefix_route_objective_s(route, observations)
                candidates.append((objective_s, start_index, step, route))
        default = next(
            item for item in candidates if item[1] == 0 and item[2] == 1
        )
        selected = min(
            candidates,
            key=lambda item: (
                item[0],
                0 if (item[1], item[2]) == (0, 1) else 1,
                item[1],
                -item[2],
            ),
        )
        estimated_saved_s = max(0.0, default[0] - selected[0])
        self.strategy_diagnostics["adaptive_prefix_default_objective_s"] = default[0]
        self.strategy_diagnostics["adaptive_prefix_selected_objective_s"] = selected[0]
        self.strategy_diagnostics["adaptive_prefix_estimated_saved_s"] = estimated_saved_s
        if (
            (selected[1], selected[2]) == (0, 1)
            or estimated_saved_s + 1e-9
            < self.cfg4.adaptive_prefix_min_estimated_saved_s
        ):
            return
        route = selected[3]
        split = 1 + self.cfg4.direction_inner_count
        self.full_direction_skeleton = list(route)
        self.phase1_skeleton = list(route[:split])
        self.phase2_skeleton = list(route[split:])
        self.coverage_route = list(route)
        self.full_skeleton_index = {
            point: index for index, point in enumerate(self.full_direction_skeleton)
        }
        self.strategy_diagnostics["adaptive_prefix_replanned"] = True
        self.strategy_diagnostics["adaptive_prefix_start_index"] = selected[1]
        self.strategy_diagnostics["adaptive_prefix_step"] = selected[2]
        self._record(
            "adaptive_prefix_replan",
            origin_direction_channels=[
                state.channel
                for state in self.channels.values()
                if len(state.observations) == 1
                and distance(state.observations[0].position, (0.0, 0.0)) <= 1e-8
            ],
            selected_start_index=selected[1],
            selected_step=selected[2],
            default_objective_s=default[0],
            selected_objective_s=selected[0],
            estimated_saved_s=estimated_saved_s,
            safety_semantics="same_certified_point_set_and_same_complete_route_length",
        )

    def _active_coverage_points(self, visited: set[Point]) -> list[tuple[Point, str]]:
        """只返回承诺路线中的下一个覆盖点，禁止跨骨架跳点。"""

        if self._all_sources_already_known():
            return []
        if not any(state.status in {"UNKNOWN", "PARTIAL"} for state in self.channels.values()):
            return []
        while self.coverage_cursor < len(self.coverage_route):
            point = self.coverage_route[self.coverage_cursor]
            if point not in visited:
                stage = "phase1" if self.coverage_cursor < len(self.phase1_skeleton) else "phase2"
                return [(point, stage)]
            self.coverage_cursor += 1
        return []

    def _all_sources_already_known(self) -> bool:
        return self._known_source_count() >= self.cfg.source_count_max

    def _coverage_commitment_points(self, visited: set[Point]) -> list[Point]:
        """按真实承诺顺序返回尚未访问的覆盖债务。"""

        if self._all_sources_already_known():
            return []
        if not any(state.status in {"UNKNOWN", "PARTIAL"} for state in self.channels.values()):
            return []
        return [point for point in self.coverage_route[self.coverage_cursor :] if point not in visited]

    def _committed_service_detour(self, point: Point, visited: set[Point]) -> tuple[float, float] | None:
        """返回插入当前承诺边后的累计ΔL及双阈值允许上限。"""

        active = self._active_coverage_points(visited)
        if not active:
            return None
        target = active[0][0]
        direct = distance(self.committed_edge_start, target)
        projected_path = (
            self.committed_edge_path_m
            + distance(self.position, point)
            + distance(point, target)
        )
        delta_l = max(0.0, projected_path - direct)
        limit = min(
            self.cfg4.committed_detour_abs_max_m,
            self.cfg4.committed_detour_ratio_max * direct,
        )
        return delta_l, limit

    def _remaining_route_points(
        self,
        visited: set[Point],
        skip_channel: int | None = None,
        stalled: set[int] | None = None,
        reference_position: Point | None = None,
    ) -> list[Point]:
        stalled = stalled or set()
        reference_position = reference_position or self.position
        points = self._coverage_commitment_points(visited)
        for channel, state in self.channels.items():
            if channel == skip_channel or channel in stalled or state.status != "DETECTED":
                continue
            options = self._directional_service_options(channel)
            if options:
                points.append(
                    min(
                        (option["point"] for option in options),
                        key=lambda point: (distance(reference_position, point), point),
                    )
                )
        return points

    def _choose_nested_fused_action(
        self,
        visited: set[Point],
        stalled: set[int],
    ) -> dict[str, Any] | None:
        """沿承诺边推进；只插入满足第三问式ΔL双阈值的服务动作。"""

        active = self._active_coverage_points(visited)
        if active:
            point, stage = active[0]
            service_actions: list[dict[str, Any]] = []
            for channel, state in self.channels.items():
                if state.status != "DETECTED" or channel in stalled:
                    continue
                for option in self._directional_service_options(channel):
                    detour = self._committed_service_detour(option["point"], visited)
                    if detour is None:
                        continue
                    delta_l, limit = detour
                    if delta_l > limit + 1e-9:
                        continue
                    action_time = (
                        self.cfg.clear_success_time_s
                        if option["action"] == "clear"
                        else self.cfg.measure_time_s
                        + (self.cfg.switch_time_s if channel != self.current_measure_channel else 0.0)
                    )
                    service_actions.append(
                        {
                            "kind": option["action"],
                            "channel": channel,
                            "point": option["point"],
                            "stage": "service",
                            "role": option["role"],
                            "objective_s": delta_l / self.cfg.speed_mps + action_time,
                            "delta_l_m": delta_l,
                            "detour_limit_m": limit,
                            "committed_skeleton_index": self.coverage_cursor,
                            "service_metadata": {
                                key: value
                                for key, value in option.items()
                                if key not in {"action", "point", "role"}
                            },
                        }
                    )
            if service_actions:
                return min(
                    service_actions,
                    key=lambda action: (
                        0 if action["kind"] == "clear" else 1,
                        action["objective_s"],
                        action["channel"],
                    ),
                )

            pending = [
                channel
                for channel, state in self.channels.items()
                if state.status in {"UNKNOWN", "PARTIAL"}
            ]
            ordered_pending = self._channel_order(pending) if pending else []
            switches = sum(
                channel != previous
                for previous, channel in zip(
                    [self.current_measure_channel] + ordered_pending,
                    ordered_pending,
                )
            )
            immediate = distance(self.position, point) / self.cfg.speed_mps
            immediate += len(ordered_pending) * self.cfg.measure_time_s
            immediate += switches * self.cfg.switch_time_s
            return {
                "kind": "coverage",
                "channel": None,
                "point": point,
                "stage": stage,
                "role": stage,
                "objective_s": immediate,
                "delta_l_m": 0.0,
                "detour_limit_m": 0.0,
                "committed_skeleton_index": self.coverage_cursor,
            }

        actions: list[dict[str, Any]] = []
        for channel, state in self.channels.items():
            if state.status != "DETECTED" or channel in stalled:
                continue
            for option in self._directional_service_options(channel):
                point = option["point"]
                immediate = distance(self.position, point) / self.cfg.speed_mps
                if option["action"] == "measure":
                    immediate += self.cfg.measure_time_s
                    if channel != self.current_measure_channel:
                        immediate += self.cfg.switch_time_s
                else:
                    immediate += self.cfg.clear_success_time_s
                actions.append(
                    {
                        "kind": option["action"],
                        "channel": channel,
                        "point": point,
                        "stage": "service",
                        "role": option["role"],
                        "objective_s": immediate,
                        "delta_l_m": None,
                        "detour_limit_m": None,
                        "committed_skeleton_index": None,
                        "service_metadata": {
                            key: value
                            for key, value in option.items()
                            if key not in {"action", "point", "role"}
                        },
                    }
                )
        if not actions:
            return None
        return min(
            actions,
            key=lambda action: (
                action["objective_s"],
                0 if action["kind"] == "clear" else 1,
                -1 if action["channel"] is None else action["channel"],
                action["point"],
            ),
        )

    def _execute_directional_service(self, action: Mapping[str, Any]) -> bool:
        channel = int(action["channel"])
        point = (float(action["point"][0]), float(action["point"][1]))
        role = str(action["role"])
        state = self.channels[channel]
        position_before = self.position
        commitment_active = action.get("committed_skeleton_index") is not None
        self.phase = "fused_service"
        self.strategy_diagnostics["directional_service_actions"] += 1
        if action["kind"] == "clear":
            if role == "verified_boundary_clear":
                self.strategy_diagnostics["verified_boundary_clear_attempts"] += 1
            success = self.clear(point, channel, f"problem4_{role}")
            if role == "verified_boundary_clear":
                key = (
                    "verified_boundary_clear_successes"
                    if success
                    else "verified_boundary_clear_failures"
                )
                self.strategy_diagnostics[key] += 1
            if commitment_active:
                self.committed_edge_path_m += distance(position_before, point)
                self.strategy_diagnostics["committed_service_insertions"] += 1
            if success:
                self._take_shared_service_measurements(point, channel)
            else:
                failure_flag = (
                    "PROBLEM4_VERIFIED_BOUNDARY_CLEAR_FAILED"
                    if role == "verified_boundary_clear"
                    else f"PROBLEM4_{role.upper()}_CLEAR_FAILED"
                )
                state.flags.append(failure_flag)
                if commitment_active:
                    state.status = "DETECTED"
                    state.flags.append("FALLBACK_DEFERRED_UNTIL_COVERAGE_COMMITMENT_ENDS")
                    return True
                polygon = localization_polygon(self.cfg, state.observations) if state.observations else np.empty((0, 2))
                previous_phase = self.phase
                self.phase = "fallback"
                self.strategy_diagnostics["directional_fallback_actions"] += 1
                success = (
                    self._fallback_polygon(channel, polygon)
                    if len(polygon) > 0
                    else False
                ) or self._fallback_wedge(channel)
                self.phase = previous_phase
            return success

        self.directional_service_attempted_points[channel].add(point)
        if role in {
            "directional_refinement",
            "cautious_axis_refinement",
            "certified_local_triangle",
        }:
            self.directional_refinement_attempts[channel] += 1
        if role == "certified_local_triangle":
            self.strategy_diagnostics["local_triangle_probe_attempts"] += 1
        elif role == "cautious_axis_refinement":
            self.strategy_diagnostics["cautious_axis_refinement_uses"] += 1
        response = self.measure(point, channel, role=role)
        if commitment_active:
            self.committed_edge_path_m += distance(position_before, point)
            self.strategy_diagnostics["committed_service_insertions"] += 1
        result = response.get("measure_result")
        if result == "near":
            success = self.clear(point, channel, f"near_at_{role}")
            if success:
                self._take_shared_service_measurements(point, channel)
            return success
        if result == "direction":
            if role == "certified_local_triangle":
                self.strategy_diagnostics["local_triangle_probe_directions"] += 1
            self._refresh_localization(state)
            self._take_shared_service_measurements(point, channel)
            return True
        if role == "certified_local_triangle":
            self.strategy_diagnostics["local_triangle_probe_no_signal"] += 1
        self._take_shared_service_measurements(point, channel)
        state.flags.append(f"{role.upper()}_NO_SIGNAL")
        state.status = "DETECTED"
        return True

    def _fallback_one_detected_channel(self, stalled: set[int]) -> bool:
        candidates = [
            channel
            for channel, state in self.channels.items()
            if state.status == "DETECTED" and channel not in stalled
        ]
        if not candidates:
            return False
        channel = min(
            candidates,
            key=lambda item: (
                distance(
                    self.position,
                    self.channels[item].localization_center
                    or self.channels[item].observations[0].position,
                ),
                item,
            ),
        )
        state = self.channels[channel]
        polygon = localization_polygon(self.cfg, state.observations) if state.observations else np.empty((0, 2))
        previous_phase = self.phase
        self.phase = "fallback"
        self.strategy_diagnostics["directional_fallback_actions"] += 1
        success = (
            self._fallback_polygon(channel, polygon)
            if len(polygon) > 0
            else False
        ) or self._fallback_wedge(channel)
        self.phase = previous_phase
        if not success:
            state.flags.append("PROBLEM4_DIRECTIONAL_FALLBACK_INCOMPLETE")
            stalled.add(channel)
        return True

    def fused_nested_plan(self) -> None:
        """嵌套骨架和频道服务的单步融合滚动规划。"""

        visited: set[Point] = set()
        stalled: set[int] = set()
        while self.counters.N_c < self.cfg.source_count_max:
            if self.completion()[0]:
                break
            action = self._choose_nested_fused_action(visited, stalled)
            if action is None:
                if self._fallback_one_detected_channel(stalled):
                    continue
                break
            self.strategy_diagnostics["nested_fusion_decisions"] += 1
            self._record(
                "nested_fusion_decision",
                action_kind=action["kind"],
                channel=action["channel"],
                point=action["point"],
                stage=action["stage"],
                role=action["role"],
                objective_s=action["objective_s"],
                delta_l_m=action.get("delta_l_m"),
                detour_limit_m=action.get("detour_limit_m"),
                committed_skeleton_index=action.get("committed_skeleton_index"),
                service_metadata=action.get("service_metadata", {}),
            )
            if action["kind"] == "coverage":
                point = action["point"]
                stage = str(action["stage"])
                final_leg = distance(self.position, point)
                direct = distance(self.committed_edge_start, point)
                actual_detour = max(
                    0.0,
                    self.committed_edge_path_m + final_leg - direct,
                )
                self._visit_coverage_point(point, stage)
                visited.add(point)
                self.visited_coverage_points.add(point)
                known_channels = {
                    channel
                    for channel, state in self.channels.items()
                    if state.status in {"DETECTED", "CLEARED"}
                }
                newly_known = sorted(
                    known_channels - self._coverage_known_channels
                )
                self._coverage_known_channels.update(known_channels)
                self.strategy_diagnostics["coverage_prefix_history"].append(
                    {
                        "visit": len(visited),
                        "stage": stage,
                        "point": point,
                        "newly_known_channels": newly_known,
                        "known_source_count": len(known_channels),
                        "cleared_count": self.counters.N_c,
                        "remaining_to_source_upper_bound": max(
                            0,
                            self.cfg.source_count_max - len(known_channels),
                        ),
                    }
                )
                self.strategy_diagnostics["committed_detour_m"] += actual_detour
                self.strategy_diagnostics["max_committed_edge_detour_m"] = max(
                    self.strategy_diagnostics["max_committed_edge_detour_m"],
                    actual_detour,
                )
                self._record(
                    "coverage_commitment_completed",
                    skeleton_index=self.coverage_cursor,
                    point=point,
                    direct_edge_m=direct,
                    actual_path_m=self.committed_edge_path_m + final_leg,
                    delta_l_m=actual_detour,
                    ratio_limit_m=self.cfg4.committed_detour_ratio_max * direct,
                    absolute_limit_m=self.cfg4.committed_detour_abs_max_m,
                )
                self.coverage_cursor += 1
                if (
                    self.coverage_cursor == 1
                    and distance(point, (0.0, 0.0)) <= 1e-8
                ):
                    self._adapt_coverage_prefix_after_origin()
                if (
                    self._all_sources_already_known()
                    and len(visited) < len(self.coverage_route)
                ):
                    self.strategy_diagnostics[
                        "coverage_terminated_at_source_upper_bound"
                    ] = True
                    self.strategy_diagnostics[
                        "coverage_visits_at_upper_bound"
                    ] = len(visited)
                self.committed_edge_start = point
                self.committed_edge_path_m = 0.0
                key = "nested_fast_visits" if stage == "phase1" else "nested_remaining_visits"
                self.strategy_diagnostics[key] += 1
                continue
            if not self._execute_directional_service(action):
                stalled.add(int(action["channel"]))
        self.phase = "post_processing"

    def baseline_plan(self) -> None:
        """兼容旧入口；当前实际执行嵌套骨架融合滚动规划。"""

        self.fused_nested_plan()

    def summary4(self, run_status: str) -> dict[str, Any]:
        result = self.summary(
            run_status,
            strategy="problem4_ring28_verified_clear_v1.5",
        )
        result["legacy_problem3_reference_coverage"] = result.pop("coverage")
        distance_only = result.pop("second_measurement_guarantee")
        result["nested_coverage"] = self.nested_coverage_certificate
        result["directional_coverage"] = self.directional_coverage_certificate
        result["second_measurement_guarantee"] = {
            "distance_only": distance_only,
            "directional_reception_guaranteed_for_one_side": False,
            "policy": "treat both lateral sides as independent rolling actions; reuse future skeleton positions opportunistically; then use orientation-independent fallback",
            "limitation": "unknown source orientation prevents a non-collinear single second point from being universally reception-guaranteed",
        }
        result["limitations"] = [
            "28点双环骨架的连续证书来自三角剖分最长边充分条件；20米步长结果只作加密诊断。",
            "方向源单个非共线第二测点无法对所有未知朝向保证接收；程序尝试左右两侧，失败后使用与朝向无关的清除兜底。",
            "局部三角组对当前保守位置多边形提供距离与方向半平面联合充分条件；其访问顺序仍是路线启发式。",
            "谨慎主轴点与共享停靠评分只用于效率优化，不构成接收或连续全局最优证明；no_signal不会虚构方向。",
            "覆盖路线采用原点—内环—外环连续承诺；低绕行服务使用ΔL双阈值，覆盖结束后的服务顺序仍是启发式，不宣称全局最优。",
            "自适应前缀只在同一认证点集的等长旋转/镜像路线中选择；评分基于原点已观测频道，不预测尚未发现源的位置。",
            "边界认证清除逐顶点验证保守定位多边形，并在20米物理半径内保留独立数值余量；验证失败时不改变v1.4流程。",
            "正式运行依赖官方模拟器允许在目标圆域外测量；问题4方向覆盖不能把域外点投影回目标圆。",
        ]
        result["phase4"] = {
            "phase1_skeleton_points": len(self.phase1_skeleton),
            "phase2_remaining_skeleton_points": len(self.phase2_skeleton),
            "full_direction_skeleton_points": len(self.full_direction_skeleton),
            "visited_coverage_points": len(self.visited_coverage_points),
            "phase1_hits": self.phase1_hits,
            "phase2_hits": self.phase2_hits,
            "coverage_no_signal_unique_counts": dict(self.phase2_no_signal_count),
            "coverage_measured_unique_counts": {
                channel: len(points) for channel, points in self.coverage_measured_points.items()
            },
            "phase2_pending_channels": [
                channel
                for channel, state in self.channels.items()
                if state.status in {"UNKNOWN", "PARTIAL"}
            ],
            "coverage_route_planned_length_m": sum(
                distance(left, right)
                for left, right in zip(self.coverage_route, self.coverage_route[1:])
            ),
            "detour_rule": {
                "formula": "delta_L=d(A,q)+d(q,B)-d(A,B), accumulated on the committed edge",
                "ratio_max": self.cfg4.committed_detour_ratio_max,
                "absolute_max_m": self.cfg4.committed_detour_abs_max_m,
            },
            "adaptive_prefix_rule": {
                "enabled": self.cfg4.adaptive_prefix_enabled,
                "minimum_origin_directions": self.cfg4.adaptive_prefix_min_origin_directions,
                "minimum_estimated_saved_s": self.cfg4.adaptive_prefix_min_estimated_saved_s,
                "candidate_count": 2,
                "invariant": "same 28 certified points and equal complete route length",
            },
            "verified_boundary_clear_rule": {
                "enabled": self.cfg4.verified_boundary_clear_enabled,
                "physical_clear_radius_m": self.cfg.clear_radius_m,
                "numerical_margin_m": self.cfg4.verified_boundary_clear_margin_m,
                "certified_limit_m": (
                    self.cfg.clear_radius_m
                    - self.cfg4.verified_boundary_clear_margin_m
                ),
                "sufficient_condition": "every conservative polygon vertex lies within the certified limit of the MEC center",
            },
            "local_triangle_rule": {
                "enabled": self.cfg4.cautious_local_enabled,
                "sufficient_condition": "U inside conv(q1,q2,q3) and max distance(q_i,g)<=1000 for every polygon vertex g",
                "max_region_radius_m": self.cfg4.local_triangle_max_region_radius_m,
                "minimum_circumradius_m": self.cfg4.local_triangle_min_circumradius_m,
                "measurement_limit_per_channel": self.cfg.max_refinements,
            },
            "shared_service_rule": {
                "enabled": self.cfg4.shared_service_enabled,
                "max_per_stop": self.cfg4.shared_service_max_per_stop,
                "max_attempts_per_channel": self.cfg4.shared_service_max_attempts_per_channel,
                "minimum_estimated_saved_s": self.cfg4.shared_service_min_saved_s,
                "safety_semantics": "zero-movement efficiency action; distance guaranteed, directional front halfplane not guaranteed",
            },
        }
        return result

    def run(self) -> dict[str, Any]:
        entered = False
        safe_to_send_exit = True
        run_status = "failed_before_enter"
        try:
            self.start_session()
            entered = True
            run_status = "running"
            self.baseline_plan()
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
        except Exception as exc:
            self.flags.append(f"UNEXPECTED_{type(exc).__name__}: {exc}")
            run_status = "unexpected_error"
        finally:
            if entered and safe_to_send_exit:
                try:
                    self.end_session()
                except Exception as exc:
                    self.flags.append(f"exit_failed: {exc!r}")
            elif entered:
                self.flags.append("EXIT_SKIPPED_BECAUSE_LAST_ACTION_OUTCOME_IS_UNCERTAIN")
        return self.summary4(run_status)


def load_config(path: str | Path) -> Problem4Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("配置文件必须是 JSON 对象")
    return Problem4Config.from_mapping(raw)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="问题4基线求解器")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("competition_b/data/problem4_config.json"),
    )
    parser.add_argument("--robot-id", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--output-dir", type=Path, default=Path("competition_b/results"))
    parser.add_argument("--check-coverage", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.check_coverage:
        report = coverage_report(cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["certified"] else 1
    if not args.robot_id:
        parser.error("真实运行必须提供 --robot-id")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = ActionLogger(Path("competition_b/logs") / f"problem4_baseline_{stamp}.jsonl")
    simulator = SimulatorClient(
        base_url=args.base_url,
        robot_id=args.robot_id,
        logger=logger,
    )
    result = Problem4BaselineSolver(cfg, simulator, logger).run()
    output = args.output_dir / f"problem4_baseline_{stamp}.json"
    write_json(output, result)
    print(f"结果已写入：{output}")
    return 0 if result.get("completion_certificate") else 1


if __name__ == "__main__":
    raise SystemExit(main())
