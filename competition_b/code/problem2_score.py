"""问题 2 的观测分支、后验外包、可靠性认证与样本评分。"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

try:
    from .problem1_geometry import point_in_convex_polygon
    from .problem2_model import (
        Problem2Config,
        _intersect_convex_polygon,
        disk_polygon,
        has_observation_axis_symmetry,
        minimum_outer_radius,
        posterior_outer_polygon,
        source_distance,
        symmetry_preserving_subsample,
        wrap_angle_deg,
    )
except ImportError:  # 支持直接执行 competition_b/code 下的文件。
    from problem1_geometry import point_in_convex_polygon  # type: ignore
    from problem2_model import (  # type: ignore
        Problem2Config,
        _intersect_convex_polygon,
        disk_polygon,
        has_observation_axis_symmetry,
        minimum_outer_radius,
        posterior_outer_polygon,
        source_distance,
        symmetry_preserving_subsample,
        wrap_angle_deg,
    )


EPS = 1e-8


def classify_observation(
    source: Sequence[float], candidate: Sequence[float], rho: float, near_threshold_m: float = 5.0
) -> str:
    """按题目边界含义返回 near、direction 或 no_signal。"""

    distance = source_distance(source, candidate)
    if distance <= near_threshold_m + EPS:
        return "near"
    if distance <= float(rho) + EPS:
        return "direction"
    return "no_signal"


def second_bearing_deg(source: Sequence[float], candidate: Sequence[float]) -> float:
    """计算 direction 分支的无误差真实示向度。"""

    source = np.asarray(source, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    return wrap_angle_deg(math.degrees(math.atan2(source[1] - candidate[1], source[0] - candidate[0])))


def _intersect_polygon_with_disk(
    outer_polygon: np.ndarray, center: Sequence[float], radius: float, ngon_edges: int, phase_deg: float = 0.0
) -> np.ndarray:
    """将外包与平移外接圆盘相交，统一复用问题2模型适配层。"""

    return _intersect_convex_polygon(
        np.asarray(outer_polygon, dtype=float),
        disk_polygon(center, radius, ngon_edges, phase_deg),
        EPS,
    )


def posterior_for_scenario(
    config: Problem2Config,
    outer_polygon: np.ndarray,
    source: Sequence[float],
    candidate: Sequence[float],
    error_deg: float,
    rho_scenario: float | None = None,
) -> dict[str, Any]:
    """先判观测类型，再构造对应后验；no_signal 不生成示向度。

    主模型没有 rho 先验。评分时显式使用 ``rho_scenario=max(1000,d1)``
    作为保守下界观测模型；单独传入 1200/1500 等值时则表示一个明确场景。
    """

    d1 = source_distance(source, config.s1)
    rho_lower = max(config.rho_min, d1)
    r2 = source_distance(source, candidate)
    if rho_scenario is None:
        # 未知 rho 时，只有安全候选可以确定为 direction；上下界之间的
        # 情况不能擅自当作 no_signal 或 direction。
        if r2 <= config.near_threshold_m + EPS:
            branch = "near"
        elif r2 <= rho_lower + EPS:
            branch = "direction"
        elif r2 > config.rho_max + EPS:
            branch = "no_signal"
        else:
            branch = "uncertain"
        rho_used = None
        rho_mode = "unknown_interval"
    else:
        rho_used = float(rho_scenario)
        if not (config.rho_min - EPS <= rho_used <= config.rho_max + EPS):
            raise ValueError("rho_scenario 必须位于 receive_range 内")
        rho_mode = "explicit_scenario"
        branch = classify_observation(source, candidate, rho_used, config.near_threshold_m)
    if branch == "near":
        phase_deg = config.theta1_deg if has_observation_axis_symmetry(config) else 0.0
        posterior = _intersect_polygon_with_disk(
            outer_polygon, candidate, config.near_threshold_m, config.ngon_edges, phase_deg
        )
        return {
            "branch": branch,
            "bearing_deg": None,
            "posterior_vertices": posterior,
            "posterior_rmin_m": minimum_outer_radius(posterior),
            "source_distance_m": r2,
            "rho_lower_m": rho_lower,
            "rho_scenario_m": rho_used,
            "rho_mode": rho_mode,
            "used_no_signal_information": False,
        }
    if branch == "no_signal":
        return {
            "branch": branch,
            "bearing_deg": None,
            "posterior_vertices": np.empty((0, 2), dtype=float),
            "posterior_rmin_m": None,
            "source_distance_m": r2,
            "rho_lower_m": rho_lower,
            "rho_scenario_m": rho_used,
            "rho_mode": rho_mode,
            "used_no_signal_information": False,
            "diagnostic": "no_signal 不生成虚构示向度；如需风险更新应使用非凸排除集。",
        }
    if branch == "uncertain":
        return {
            "branch": branch,
            "bearing_deg": None,
            "posterior_vertices": np.empty((0, 2), dtype=float),
            "posterior_rmin_m": None,
            "source_distance_m": r2,
            "rho_lower_m": rho_lower,
            "rho_scenario_m": None,
            "rho_mode": rho_mode,
            "used_no_signal_information": False,
            "diagnostic": "rho 未知且当前距离位于可接收半径上下界之间，不能确定观测分支。",
        }
    true_bearing = second_bearing_deg(source, candidate)
    measured = wrap_angle_deg(true_bearing + float(error_deg))
    posterior = posterior_outer_polygon(
        outer_polygon,
        candidate,
        measured,
        config.delta_deg,
        config.rho_max,
        config.ngon_edges,
        disk_phase_deg=config.theta1_deg if has_observation_axis_symmetry(config) else 0.0,
    )
    return {
        "branch": branch,
        "bearing_deg": measured,
        "posterior_vertices": posterior,
        "posterior_rmin_m": minimum_outer_radius(posterior),
        "source_distance_m": r2,
        "rho_lower_m": rho_lower,
        "rho_scenario_m": rho_used,
        "rho_mode": rho_mode,
        "used_no_signal_information": False,
    }


def receive_margin(config: Problem2Config, candidate: Sequence[float], source_points: np.ndarray) -> float:
    """返回 sup_G[r2-max(1000,d1)]，非正表示样本层面保证接收。"""

    if len(source_points) == 0:
        return float("nan")
    candidate = np.asarray(candidate, dtype=float)
    d1 = np.linalg.norm(source_points - config.s1[None, :], axis=1)
    r2 = np.linalg.norm(source_points - candidate[None, :], axis=1)
    return float(np.max(r2 - np.maximum(config.rho_min, d1)))


def score_candidate(
    config: Problem2Config,
    outer_polygon: np.ndarray,
    candidate: Sequence[float],
    source_points: np.ndarray,
    error_grid_deg: Sequence[float] | None = None,
    source_limit: int | None = None,
) -> dict[str, Any]:
    """对一个候选点计算分支统计和样本最坏后验半径。"""

    candidate = np.asarray(candidate, dtype=float)
    sources = symmetry_preserving_subsample(
        config,
        np.asarray(source_points, dtype=float),
        source_limit or config.search_source_limit,
    )
    errors = tuple(error_grid_deg or config.error_grid_deg)
    branch_counts = {"near": 0, "direction": 0, "no_signal": 0, "uncertain": 0}
    radii: list[float] = []
    failed: list[dict[str, Any]] = []
    geometry_intersections = 0
    for source_index, source in enumerate(sources):
        for error_deg in errors:
            scenario = posterior_for_scenario(config, outer_polygon, source, candidate, float(error_deg), rho_scenario=None)
            branch = scenario["branch"]
            branch_counts[branch] += 1
            if branch not in {"no_signal", "uncertain"}:
                geometry_intersections += 1
            radius = scenario["posterior_rmin_m"]
            if branch not in {"no_signal", "uncertain"} and radius is None:
                failed.append({"source_index": source_index, "error_deg": float(error_deg), "branch": branch, "reason": "empty_posterior"})
            elif branch == "uncertain":
                failed.append({"source_index": source_index, "error_deg": float(error_deg), "branch": branch, "reason": "rho_interval_does_not_identify_observation"})
            elif radius is not None:
                radii.append(float(radius))
    margin = receive_margin(config, candidate, sources)
    if failed:
        sample_worst = None
    else:
        sample_worst = max(radii) if radii else None
    baseline = float(np.linalg.norm(candidate - config.s1))
    if sample_worst is None:
        j_lambda = float("inf")
    else:
        j_lambda = sample_worst / config.r_ref_m + config.lambda_weight * baseline / config.b_ref_m
    return {
        "candidate": {"x": float(candidate[0]), "y": float(candidate[1])},
        "travel_distance_m": baseline,
        "margin": margin,
        "sample_worst_rmin_m": sample_worst,
        "n_near": branch_counts["near"],
        "n_direction": branch_counts["direction"],
        "n_no_signal": branch_counts["no_signal"],
        "n_uncertain": branch_counts["uncertain"],
        "scenario_count": int(len(sources) * len(errors)),
        "source_count": int(len(sources)),
        "error_count": int(len(errors)),
        "failed_scenarios": failed,
        "mean_miss": None,
        "j_lambda": float(j_lambda),
        "geometry_intersections": geometry_intersections,
    }


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    denom = float(np.dot(direction, direction))
    if denom <= EPS:
        return float(np.linalg.norm(point - start))
    t = min(1.0, max(0.0, float(np.dot(point - start, direction)) / denom))
    return float(np.linalg.norm(point - (start + t * direction)))


def point_to_triangle_distance(point: Sequence[float], triangle: np.ndarray) -> float:
    """点到三角形的距离，供连续可靠性充分条件使用。"""

    p = np.asarray(point, dtype=float)
    tri = np.asarray(triangle, dtype=float).reshape((3, 2))
    a, b, c = tri
    v0, v1, v2 = b - a, c - a, p - a
    den = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(den) > EPS:
        u = float((v2[0] * v1[1] - v1[0] * v2[1]) / den)
        v = float((v0[0] * v2[1] - v2[0] * v0[1]) / den)
        if u >= -EPS and v >= -EPS and u + v <= 1.0 + EPS:
            return 0.0
    return min(_segment_distance(p, a, b), _segment_distance(p, b, c), _segment_distance(p, c, a))


def _split_triangle(triangle: np.ndarray) -> list[np.ndarray]:
    a, b, c = np.asarray(triangle, dtype=float)
    ab, bc, ca = (a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0
    return [
        np.asarray([a, ab, ca]),
        np.asarray([ab, b, bc]),
        np.asarray([ca, bc, c]),
        np.asarray([ab, bc, ca]),
    ]


def _fan_triangles(polygon: np.ndarray) -> list[np.ndarray]:
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) < 3:
        return []
    center = np.mean(polygon, axis=0)
    return [np.asarray([center, polygon[i], polygon[(i + 1) % len(polygon)]]) for i in range(len(polygon))]


def certify_receive(
    config: Problem2Config,
    outer_polygon: np.ndarray,
    candidate: Sequence[float],
    source_points: np.ndarray,
    subdivision_budget: int | None = None,
) -> dict[str, Any]:
    """用保守外包三角剖分给出 passed/violated/unknown。"""

    budget = subdivision_budget or config.subdivision_budget
    candidate = np.asarray(candidate, dtype=float)
    sample_margin = receive_margin(config, candidate, source_points)
    if not np.isfinite(sample_margin):
        return {"status": "unknown", "continuous_worst_case_certified": False, "cert_margin_m": None, "budget_used": 0, "evidence": "精确源采样为空。"}
    if sample_margin > 1e-6:
        return {"status": "violated", "continuous_worst_case_certified": False, "cert_margin_m": -sample_margin, "budget_used": 0, "evidence": "精确源采样中存在保证接收违反点。"}
    triangles = _fan_triangles(outer_polygon)
    if not triangles:
        return {"status": "unknown", "continuous_worst_case_certified": False, "cert_margin_m": None, "budget_used": 0, "evidence": "保守外包退化或为空，无法三角剖分认证。"}
    stack: list[tuple[np.ndarray, int]] = [(triangle, 0) for triangle in triangles]
    min_slack = float("inf")
    used = 0
    cert_eps = 1e-7
    while stack:
        triangle, depth = stack.pop()
        r2_upper = max(float(np.linalg.norm(candidate - vertex)) for vertex in triangle)
        d1_lower = point_to_triangle_distance(config.s1, triangle)
        slack = max(config.rho_min, d1_lower) - r2_upper
        min_slack = min(min_slack, slack)
        if slack >= -cert_eps:
            continue
        if used + 1 >= budget or depth >= 14:
            return {
                "status": "unknown",
                "continuous_worst_case_certified": False,
                "cert_margin_m": float(min_slack),
                "budget_used": used,
                "evidence": "达到连续域细分预算或深度上限，未完成充分条件认证。",
            }
        stack.extend((child, depth + 1) for child in _split_triangle(triangle))
        used += 1
    return {
        "status": "passed",
        "continuous_worst_case_certified": True,
        "cert_margin_m": float(min_slack),
        "budget_used": used,
        "evidence": "保守外包三角剖分上的充分条件全部通过。",
    }


def certify_receive_cell(
    config: Problem2Config,
    outer_polygon: np.ndarray,
    bounds: Sequence[float],
    source_points: np.ndarray,
    subdivision_budget: int | None = None,
    corners: np.ndarray | None = None,
) -> dict[str, Any]:
    """认证矩形或旋转矩形内任意检测点均满足保守接收条件。

    对检测点矩形，距离关于检测点是凸函数，最大值取在矩形角点；对源外包
    三角形同理取三角形顶点。由此得到覆盖整块检测区域的充分条件。
    """

    if corners is None:
        x0, x1, y0, y1 = map(float, bounds)
        detector_corners = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    else:
        detector_corners = np.asarray(corners, dtype=float)
        if detector_corners.shape != (4, 2):
            raise ValueError("corners 必须是 4×2 的旋转矩形顶点")
    if len(source_points) == 0:
        return {
            "status": "unknown",
            "continuous_detector_cell_certified": False,
            "cert_margin_m": None,
            "budget_used": 0,
            "evidence": "精确源采样为空。",
        }
    sources = np.asarray(source_points, dtype=float)
    d1 = np.linalg.norm(sources - config.s1[None, :], axis=1)
    r2_upper = np.max(np.linalg.norm(sources[:, None, :] - detector_corners[None, :, :], axis=2), axis=1)
    sample_slack = np.maximum(config.rho_min, d1) - r2_upper
    if float(np.min(sample_slack)) < -1e-6:
        return {
            "status": "violated",
            "continuous_detector_cell_certified": False,
            "cert_margin_m": float(np.min(sample_slack)),
            "budget_used": 0,
            "evidence": "源样本与矩形角点给出整单元接收违反反例。",
        }
    triangles = _fan_triangles(outer_polygon)
    if not triangles:
        return {
            "status": "unknown",
            "continuous_detector_cell_certified": False,
            "cert_margin_m": None,
            "budget_used": 0,
            "evidence": "保守外包退化或为空，无法认证整块检测区域。",
        }
    budget = subdivision_budget or config.subdivision_budget
    stack: list[tuple[np.ndarray, int]] = [(triangle, 0) for triangle in triangles]
    min_slack = float("inf")
    used = 0
    cert_eps = 1e-7
    while stack:
        triangle, depth = stack.pop()
        upper = max(float(np.linalg.norm(sensor - source)) for sensor in detector_corners for source in triangle)
        d1_lower = point_to_triangle_distance(config.s1, triangle)
        slack = max(config.rho_min, d1_lower) - upper
        min_slack = min(min_slack, slack)
        if slack >= -cert_eps:
            continue
        if used + 1 >= budget or depth >= 14:
            return {
                "status": "unknown",
                "continuous_detector_cell_certified": False,
                "cert_margin_m": float(min_slack),
                "budget_used": used,
                "evidence": "达到整单元接收认证的细分预算或深度上限。",
            }
        stack.extend((child, depth + 1) for child in _split_triangle(triangle))
        used += 1
    return {
        "status": "passed",
        "continuous_detector_cell_certified": True,
        "cert_margin_m": float(min_slack),
        "budget_used": used,
        "evidence": "矩形内任意检测点对保守源外包均满足接收充分条件。",
    }
