"""问题 2 的可行源集合、保守外包与场景采样。

本模块只负责几何状态，不负责选择第二个检测点。问题1的多边形裁剪和
最小包围圆实现通过适配层复用，不复制问题1算法。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .problem1_geometry import (
        DEFAULT_TOL,
        GeometryTolerance,
        bearing_halfplane_functions,
        build_regular_ngon,
        clean_polygon,
        clip_halfplane,
        minimum_enclosing_circle,
    )
except ImportError:  # 支持直接执行 competition_b/code 下的文件。
    from problem1_geometry import (  # type: ignore
        DEFAULT_TOL,
        GeometryTolerance,
        bearing_halfplane_functions,
        build_regular_ngon,
        clean_polygon,
        clip_halfplane,
        minimum_enclosing_circle,
    )


EPS = 1e-9


@dataclass(frozen=True)
class TargetCircle:
    """目标圆域。"""

    cx: float
    cy: float
    radius: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=float)


@dataclass(frozen=True)
class Problem2Config:
    """从案例 JSON 解析出的稳定参数。"""

    case_id: str
    s1: np.ndarray
    theta1_deg: float
    delta_deg: float
    target: TargetCircle
    rho_min: float
    rho_max: float
    near_threshold_m: float
    budget_b_m: float
    source_n_angle: int
    source_n_distance: int
    candidate_step_m: float
    candidate_b_substeps: int
    region_base_cell_m: float
    region_max_depth: int
    region_max_base_cells: int
    region_source_limit: int
    error_grid_deg: tuple[float, ...]
    ngon_edges: int
    subdivision_budget: int
    lambda_weight: float
    r_ref_m: float
    b_ref_m: float
    epsilon_j: float
    focus_epsilon_j: float
    random_seed: int
    max_candidates: int
    search_source_limit: int
    validation_source_count: int

    @property
    def region_min_cell_m(self) -> float:
        return self.region_base_cell_m / (2 ** self.region_max_depth)


def wrap_angle_deg(angle_deg: float) -> float:
    """把角度归一到 [0, 360)。"""

    return float(angle_deg % 360.0)


def angular_difference_deg(a_deg: float, b_deg: float) -> float:
    """返回两个角度的最小有符号差的绝对值。"""

    return abs((float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0)


def observation_axis_basis(config: Problem2Config) -> tuple[np.ndarray, np.ndarray]:
    """返回首次示向中轴线的单位切向量和法向量。"""

    theta = math.radians(config.theta1_deg)
    tangent = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    normal = np.array([-math.sin(theta), math.cos(theta)], dtype=float)
    return tangent, normal


def reflect_about_observation_axis(config: Problem2Config, point: Sequence[float]) -> np.ndarray:
    """把一个世界坐标点关于经过 S1 的首次示向中轴线镜像。"""

    tangent, normal = observation_axis_basis(config)
    offset = np.asarray(point, dtype=float) - config.s1
    return config.s1 + float(np.dot(offset, tangent)) * tangent - float(np.dot(offset, normal)) * normal


def has_observation_axis_symmetry(config: Problem2Config, tol: float = 1e-7) -> bool:
    """判断输入几何及误差场景是否关于首次示向中轴线严格对称。"""

    _, normal = observation_axis_basis(config)
    center_offset = config.target.center - config.s1
    geometry_symmetric = abs(float(np.dot(center_offset, normal))) <= tol
    errors = sorted(float(value) for value in config.error_grid_deg)
    error_symmetric = len(errors) == len(config.error_grid_deg) and all(
        abs(left + right) <= tol for left, right in zip(errors, reversed(errors))
    )
    return bool(geometry_symmetric and error_symmetric)


def _evenly_spaced_positions(count: int, selected_count: int) -> list[int]:
    if selected_count <= 0 or count <= 0:
        return []
    if selected_count >= count:
        return list(range(count))
    return sorted(set(int(index) for index in np.linspace(0, count - 1, selected_count, dtype=int)))


def symmetry_preserving_subsample(config: Problem2Config, points: np.ndarray, limit: int) -> np.ndarray:
    """压缩源样本；对严格对称案例只保留完整镜像轨道，并实际返回两侧点。"""

    points = np.asarray(points, dtype=float)
    limit = max(1, int(limit))
    if len(points) <= limit:
        return points
    if not has_observation_axis_symmetry(config):
        indices = np.linspace(0, len(points) - 1, limit, dtype=int)
        return points[np.unique(indices)]

    tangent, normal = observation_axis_basis(config)
    offsets = points - config.s1[None, :]
    along = offsets @ tangent
    across = offsets @ normal
    groups_by_key: dict[tuple[float, float], list[int]] = {}
    for index, (u_value, v_value) in enumerate(zip(along, across)):
        key = (round(float(u_value), 7), round(abs(float(v_value)), 7))
        groups_by_key.setdefault(key, []).append(index)
    groups = list(groups_by_key.values())
    singles = [group for group in groups if len(group) == 1 and abs(float(across[group[0]])) <= 1e-6]
    pairs = [group for group in groups if len(group) == 2 and float(across[group[0]]) * float(across[group[1]]) < 0.0]
    if len(singles) + len(pairs) != len(groups):
        indices = np.linspace(0, len(points) - 1, limit, dtype=int)
        return points[np.unique(indices)]

    singles.sort(key=lambda group: float(along[group[0]]))
    pairs.sort(key=lambda group: (math.atan2(abs(float(across[group[0]])), float(along[group[0]])), float(np.linalg.norm(offsets[group[0]]))))
    feasible: list[tuple[float, int, int]] = []
    for single_count in range(min(len(singles), limit) + 1):
        remaining = limit - single_count
        if remaining % 2:
            continue
        pair_count = remaining // 2
        if pair_count <= len(pairs):
            expected_singles = limit * len(singles) / max(1, len(points))
            feasible.append((abs(single_count - expected_singles), -single_count, pair_count))
    if not feasible:
        indices = np.linspace(0, len(points) - 1, limit, dtype=int)
        return points[np.unique(indices)]
    _, negative_single_count, pair_count = min(feasible)
    single_count = -negative_single_count
    chosen_groups = [singles[index] for index in _evenly_spaced_positions(len(singles), single_count)]
    chosen_groups.extend(pairs[index] for index in _evenly_spaced_positions(len(pairs), pair_count))
    chosen_indices = sorted(index for group in chosen_groups for index in group)
    return points[chosen_indices]


def parse_config(case: Mapping[str, Any]) -> Problem2Config:
    """校验并解析一个问题2案例。"""

    s1_raw = case.get("S1")
    target_raw = case.get("target_region")
    if not isinstance(s1_raw, Mapping) or not isinstance(target_raw, Mapping):
        raise ValueError("案例必须包含 S1 和 target_region 对象")
    s1 = np.array([float(s1_raw["x"]), float(s1_raw["y"])], dtype=float)
    target = TargetCircle(
        float(target_raw.get("cx", 0.0)),
        float(target_raw.get("cy", 0.0)),
        float(target_raw["R"]),
    )
    if not np.isfinite(s1).all() or not all(math.isfinite(v) for v in (target.cx, target.cy, target.radius)):
        raise ValueError("S1 和目标圆参数必须有限")
    if target.radius <= 0:
        raise ValueError("目标圆半径必须为正")
    receive = case.get("receive_range", [1000.0, 1500.0])
    if not isinstance(receive, Sequence) or len(receive) != 2:
        raise ValueError("receive_range 必须是 [rho_min, rho_max]")
    rho_min, rho_max = map(float, receive)
    if not (0 < rho_min <= rho_max):
        raise ValueError("receive_range 必须满足 0 < rho_min <= rho_max")
    source_grid = case.get("source_grid", {})
    candidate_grid = case.get("candidate_grid", {})
    certification = case.get("certification", {})
    objective = case.get("objective", {})
    errors = tuple(float(v) for v in case.get("error_grid_deg", [-1.0, 0.0, 1.0]))
    if not errors or min(errors) < -float(case.get("delta_deg", 1.0)) - EPS or max(errors) > float(case.get("delta_deg", 1.0)) + EPS:
        raise ValueError("error_grid_deg 必须位于第二次示向度误差范围内")
    epsilon_j = max(0.0, float(objective.get("epsilon_J", 0.02)))
    focus_epsilon_j = max(0.0, float(objective.get("focus_epsilon_J", min(epsilon_j, 0.005))))
    candidate_step_m = max(1e-6, float(candidate_grid.get("step_m", 25.0)))
    region_base_cell_m = max(1e-6, float(candidate_grid.get("region_base_cell_m", candidate_step_m / 2.0)))
    search_source_limit = max(8, int(case.get("search_source_limit", 96)))
    return Problem2Config(
        case_id=str(case.get("case_id", "problem2_unnamed")),
        s1=s1,
        theta1_deg=float(case["theta1_deg"]),
        delta_deg=float(case.get("delta_deg", 1.0)),
        target=target,
        rho_min=rho_min,
        rho_max=rho_max,
        near_threshold_m=float(case.get("near_threshold_m", 5.0)),
        budget_b_m=float(case.get("budget_B_m", 3000.0)),
        source_n_angle=max(3, int(source_grid.get("n_angle", 33))),
        source_n_distance=max(2, int(source_grid.get("n_distance", 33))),
        candidate_step_m=candidate_step_m,
        candidate_b_substeps=max(1, int(candidate_grid.get("b_substeps", 8))),
        region_base_cell_m=region_base_cell_m,
        region_max_depth=max(0, int(candidate_grid.get("region_max_depth", 1))),
        region_max_base_cells=max(16, int(candidate_grid.get("region_max_base_cells", 256))),
        region_source_limit=max(8, int(candidate_grid.get("region_source_limit", min(search_source_limit, 24)))),
        error_grid_deg=errors,
        ngon_edges=max(32, int(case.get("ngon_edges", 3600))),
        subdivision_budget=max(1, int(certification.get("subdivision_budget", 200000))),
        lambda_weight=float(objective.get("lambda", 0.0)),
        r_ref_m=max(EPS, float(objective.get("R_ref_m", 1500.0))),
        b_ref_m=max(EPS, float(objective.get("B_ref_m", 1500.0))),
        epsilon_j=epsilon_j,
        focus_epsilon_j=focus_epsilon_j,
        random_seed=int(case.get("random_seed", 20260911)),
        max_candidates=max(25, int(candidate_grid.get("max_candidates", 900))),
        search_source_limit=search_source_limit,
        validation_source_count=max(8, int(case.get("validation_source_count", 96))),
    )


def ray_circle_interval(
    s1: Sequence[float], angle_deg: float, target: TargetCircle, d_upper: float = 1500.0
) -> tuple[float, float] | None:
    """求从 S1 沿给定角度射线进入目标圆的距离区间。"""

    s = np.asarray(s1, dtype=float)
    u = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))], dtype=float)
    v = s - target.center
    vu = float(np.dot(v, u))
    discriminant = vu * vu - (float(np.dot(v, v)) - target.radius * target.radius)
    if discriminant < -1e-8:
        return None
    root = math.sqrt(max(0.0, discriminant))
    lo = -vu - root
    hi = -vu + root
    lo = max(0.0, lo)
    hi = min(float(d_upper), hi)
    if hi <= lo + EPS:
        return None
    return lo, hi


def per_angle_distance_intervals(config: Problem2Config) -> list[dict[str, Any]]:
    """逐角度计算目标圆和 1500 米距离上界的交集。"""

    values = np.linspace(
        config.theta1_deg - config.delta_deg,
        config.theta1_deg + config.delta_deg,
        config.source_n_angle,
    )
    intervals: list[dict[str, Any]] = []
    for angle in values:
        interval = ray_circle_interval(config.s1, float(angle), config.target, config.rho_max)
        intervals.append(
            {
                "angle_deg": wrap_angle_deg(float(angle)),
                "angle_unwrapped_deg": float(angle),
                "d_min": None if interval is None else interval[0],
                "d_max": None if interval is None else interval[1],
                "has_interval": interval is not None,
            }
        )
    return intervals


def _valid_source(point: np.ndarray, config: Problem2Config) -> bool:
    d1 = float(np.linalg.norm(point - config.s1))
    if not (d1 > config.near_threshold_m + 1e-8 and d1 <= config.rho_max + 1e-7):
        return False
    if float(np.linalg.norm(point - config.target.center)) > config.target.radius + 1e-7:
        return False
    bearing = math.degrees(math.atan2(point[1] - config.s1[1], point[0] - config.s1[0]))
    return angular_difference_deg(bearing, config.theta1_deg) <= config.delta_deg + 1e-7


def sample_exact_source_points(config: Problem2Config, phase: float = 0.0) -> np.ndarray:
    """按角度和距离网格采样精确可行集 U，严格排除 near 圆盘。"""

    values = np.linspace(
        config.theta1_deg - config.delta_deg,
        config.theta1_deg + config.delta_deg,
        config.source_n_angle,
    )
    points: list[np.ndarray] = []
    for angle in values:
        interval = ray_circle_interval(config.s1, float(angle), config.target, config.rho_max)
        if interval is None:
            continue
        lo = max(interval[0], config.near_threshold_m + 1e-6)
        hi = interval[1]
        if hi <= lo:
            continue
        if config.source_n_distance == 1:
            distances = np.array([(lo + hi) / 2.0])
        else:
            distances = np.linspace(lo, hi, config.source_n_distance)
        for distance in distances:
            angle_rad = math.radians(float(angle) + phase)
            point = config.s1 + float(distance) * np.array([math.cos(angle_rad), math.sin(angle_rad)])
            if _valid_source(point, config):
                points.append(point)
    if not points:
        return np.empty((0, 2), dtype=float)
    rounded: dict[tuple[float, float], np.ndarray] = {}
    for point in points:
        rounded[(round(float(point[0]), 8), round(float(point[1]), 8))] = point
    return np.asarray(list(rounded.values()), dtype=float)


def sample_validation_source_points(config: Problem2Config, count: int | None = None) -> np.ndarray:
    """使用固定种子的独立随机采样构造验证源集。"""

    count = count or config.validation_source_count
    rng = np.random.default_rng(config.random_seed + 7919)
    points: list[np.ndarray] = []
    max_attempts = max(100, count * 80)
    for _ in range(max_attempts):
        angle = rng.uniform(config.theta1_deg - config.delta_deg, config.theta1_deg + config.delta_deg)
        interval = ray_circle_interval(config.s1, float(angle), config.target, config.rho_max)
        if interval is None:
            continue
        lo = max(interval[0], config.near_threshold_m + 1e-6)
        hi = interval[1]
        if hi <= lo:
            continue
        distance = rng.uniform(lo, hi)
        point = config.s1 + distance * np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        if _valid_source(point, config):
            points.append(point)
        if len(points) >= count:
            break
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def _intersect_convex_polygon(polygon: np.ndarray, clip_polygon: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """用问题1的半平面裁剪把两个逆时针凸多边形相交。"""

    result = np.asarray(polygon, dtype=float).reshape((-1, 2))
    clip = np.asarray(clip_polygon, dtype=float).reshape((-1, 2))
    if len(result) == 0 or len(clip) < 3:
        return np.empty((0, 2), dtype=float)
    for i, start in enumerate(clip):
        end = clip[(i + 1) % len(clip)]
        edge = end - start
        result = clip_halfplane(result, lambda p, start=start, edge=edge: float(edge[0] * (p[1] - start[1]) - edge[1] * (p[0] - start[0])), eps)
        result = clean_polygon(result, eps_geo=eps, eps_col=eps)
        if len(result) == 0:
            break
    return result


def build_conservative_outer_polygon(config: Problem2Config, radius: float | None = None) -> np.ndarray:
    """构造 U_bar = T ∩ W(S1) ∩ Disk(S1,1500) 的保守凸外包。"""

    radius = config.rho_max if radius is None else float(radius)
    disk_phase_deg = config.theta1_deg if has_observation_axis_symmetry(config) else 0.0
    outer = disk_polygon(config.target.center, config.target.radius, config.ngon_edges, disk_phase_deg)
    f_minus, f_plus = bearing_halfplane_functions(config.s1, config.theta1_deg, config.delta_deg)
    eps = DEFAULT_TOL.length(max(config.target.radius, radius, float(np.linalg.norm(config.s1))))
    outer = clip_halfplane(outer, f_minus, eps)
    outer = clean_polygon(outer, eps_geo=eps, eps_col=eps)
    outer = clip_halfplane(outer, f_plus, eps)
    outer = clean_polygon(outer, eps_geo=eps, eps_col=eps)
    if len(outer) == 0:
        return outer
    distance_disk = disk_polygon(config.s1, radius, config.ngon_edges, disk_phase_deg)
    outer = _intersect_convex_polygon(outer, distance_disk, eps)
    return clean_polygon(outer, eps_geo=eps, eps_col=eps)


def disk_polygon(center: Sequence[float], radius: float, n: int, phase_deg: float = 0.0) -> np.ndarray:
    """返回平移后的外接正多边形圆盘近似。"""

    polygon = build_regular_ngon(float(radius), int(n))
    if abs(float(phase_deg)) > EPS:
        angle = math.radians(float(phase_deg))
        rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        polygon = polygon @ rotation.T
    return polygon + np.asarray(center, dtype=float)


def posterior_outer_polygon(
    outer_polygon: np.ndarray,
    sensor: Sequence[float],
    bearing_deg: float,
    delta_deg: float,
    distance_radius: float,
    ngon_edges: int,
    tol: GeometryTolerance = DEFAULT_TOL,
    disk_phase_deg: float = 0.0,
) -> np.ndarray:
    """在已有外包上加入第二次扇形和距离上界约束。"""

    sensor = np.asarray(sensor, dtype=float)
    scale = max(1.0, float(np.max(np.linalg.norm(outer_polygon, axis=1))) if len(outer_polygon) else distance_radius)
    eps = tol.length(scale)
    f_minus, f_plus = bearing_halfplane_functions(sensor, bearing_deg, delta_deg)
    result = clip_halfplane(np.asarray(outer_polygon, dtype=float), f_minus, eps)
    result = clean_polygon(result, eps_geo=eps, eps_col=eps)
    if len(result) == 0:
        return result
    result = clip_halfplane(result, f_plus, eps)
    result = clean_polygon(result, eps_geo=eps, eps_col=eps)
    if len(result) == 0:
        return result
    return _intersect_convex_polygon(
        result,
        disk_polygon(sensor, distance_radius, ngon_edges, disk_phase_deg),
        eps,
    )


def source_distance(source: Sequence[float], sensor: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(source, dtype=float) - np.asarray(sensor, dtype=float)))


def minimum_outer_radius(polygon: np.ndarray) -> float | None:
    """复用问题1的 MEC，并只返回半径。"""

    if len(polygon) == 0:
        return None
    _, radius = minimum_enclosing_circle(polygon)
    return float(radius)


def source_set_summary(config: Problem2Config, outer_polygon: np.ndarray, source_points: np.ndarray) -> dict[str, Any]:
    """形成报告中的 possible_source_set。"""

    intervals = per_angle_distance_intervals(config)
    return {
        "exact_definition": "{G in T ∩ W(S1,theta1,delta) : near_threshold < d1 <= rho_max}",
        "sample_generation": "angle_distance_tensor_grid_formula",
        "sample_point_policy": "every listed point is generated by the stated angle-distance grid and exact-U membership test; no placeholder or interpolated points",
        "near_exclusion_open_m": config.near_threshold_m,
        "receive_upper_bound_m": config.rho_max,
        "conservative_outer": True,
        "outer_label": "conservative_outer",
        "convex_outer_vertices": outer_polygon.tolist(),
        "outer_mec_radius_m": minimum_outer_radius(outer_polygon),
        "per_angle_intervals": intervals,
        "samples_inside_exact_U": int(len(source_points)),
        "sample_points": source_points.tolist(),
        "is_empty": len(source_points) == 0 or len(outer_polygon) == 0,
    }


def thales_reference_point(config: Problem2Config, source_points: np.ndarray) -> dict[str, Any]:
    """按代表源和代表基线给出泰勒斯启发式参考点。"""

    if len(source_points) == 0:
        return {
            "note": "启发式参考，不是解析最优；精确源集合为空。",
            "thales_s2": None,
            "thales_s2_pair": [],
            "reference_source": None,
        }
    representative = np.mean(source_points, axis=0)
    d = source_distance(representative, config.s1)
    b = min(config.budget_b_m * 0.5, max(1.0, 0.6 * d))
    if d <= b + EPS:
        b = max(1e-3, 0.5 * d)
    psi = math.degrees(math.acos(min(1.0, max(-1.0, b / max(d, EPS)))))
    # 目标圆会截断首次示向带，代表源的真实方位未必等于示向中轴。
    # 泰勒斯直角关系必须以当前实际选出的代表源为准。
    if has_observation_axis_symmetry(config):
        theta = math.radians(config.theta1_deg)
    else:
        theta = math.atan2(float(representative[1] - config.s1[1]), float(representative[0] - config.s1[0]))
    points = [
        config.s1 + b * np.array([math.cos(theta + sign * math.radians(psi)), math.sin(theta + sign * math.radians(psi))])
        for sign in (1.0, -1.0)
    ]
    return {
        "note": "启发式参考，不是解析最优；泰勒斯圆不限制搜索域。",
        "thales_s2": {"x": float(points[0][0]), "y": float(points[0][1])},
        "thales_s2_pair": [{"x": float(point[0]), "y": float(point[1])} for point in points],
        "reference_source": {"x": float(representative[0]), "y": float(representative[1])},
        "reference_baseline_m": float(b),
        "reference_psi_deg": float(psi),
    }


def parse_case_and_geometry(case: Mapping[str, Any]) -> tuple[Problem2Config, np.ndarray, np.ndarray, dict[str, Any]]:
    """统一构造问题2求解需要的几何输入。"""

    config = parse_config(case)
    source_points = sample_exact_source_points(config)
    outer_polygon = build_conservative_outer_polygon(config)
    return config, source_points, outer_polygon, source_set_summary(config, outer_polygon, source_points)


def exact_source_contains(config: Problem2Config, point: Sequence[float]) -> bool:
    """公开的精确源点判定，测试用于验证 near 排除与旋转平移。"""

    return _valid_source(np.asarray(point, dtype=float), config)
