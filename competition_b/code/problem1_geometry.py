"""问题 1 的数值稳健计算几何。

主可行区域是外接正 N 边形与每个示向度观测对应的两个半平面的交集。
采用外接多边形是有意的：它是半径 R 圆盘的保守外逼近。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

try:
    from competition_b.config import CLEAR_RADIUS_M, SAFE_CLEAR_RADIUS_M
except ImportError:  # 支持从 competition_b/code 直接执行。
    CLEAR_RADIUS_M = 20.0
    SAFE_CLEAR_RADIUS_M = 18.0

ArrayLike = Sequence[Sequence[float]] | np.ndarray


@dataclass(frozen=True)
class GeometryTolerance:
    """以相关单位表示的绝对容差加相对容差。"""

    abs_m: float = 1e-9
    rel: float = 1e-12
    abs_m2: float = 1e-9

    def length(self, scale_m: float) -> float:
        return self.abs_m + self.rel * max(1.0, abs(float(scale_m)))

    def area(self, scale_m: float) -> float:
        return self.abs_m2 + self.rel * max(1.0, abs(float(scale_m)) ** 2)


DEFAULT_TOL = GeometryTolerance()


def as_points(points: ArrayLike | Iterable[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(list(points) if not isinstance(points, np.ndarray) else points, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        raise ValueError("points 必须是形状为 (n, 2) 的有限数组")
    return arr.copy()


def cross(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def build_regular_ngon(radius: float = 1800.0, n: int = 3600) -> np.ndarray:
    """返回围绕半径 R 圆盘的逆时针外接正 n 边形。"""
    radius = float(radius)
    n = int(n)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius 必须为正的有限数")
    if n < 3:
        raise ValueError("n 至少必须为 3")
    vertex_radius = radius / math.cos(math.pi / n)
    angles = 2.0 * math.pi * np.arange(n, dtype=float) / n
    return vertex_radius * np.column_stack((np.cos(angles), np.sin(angles)))


def bearing_vectors(theta_deg: float, delta_deg: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """返回 theta-delta 和 theta+delta 度方向的单位向量。"""
    theta = float(theta_deg)
    delta = float(delta_deg)
    if not math.isfinite(theta) or not math.isfinite(delta) or delta < 0 or delta >= 90:
        raise ValueError("theta 必须为有限数，且 0 <= delta_deg < 90")
    angles = np.deg2rad([theta - delta, theta + delta])
    return np.array([math.cos(angles[0]), math.sin(angles[0])]), np.array(
        [math.cos(angles[1]), math.sin(angles[1])]
    )


def bearing_halfplane_functions(
    sensor: Sequence[float], theta_deg: float, delta_deg: float = 1.0
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], float]]:
    """返回一个示向扇形对应的 f_minus >= 0 和 f_plus >= 0。"""
    s = np.asarray(sensor, dtype=float)
    if s.shape != (2,) or not np.isfinite(s).all():
        raise ValueError("sensor 必须是有限的二维向量")
    u_minus, u_plus = bearing_vectors(theta_deg, delta_deg)

    def f_minus(point: np.ndarray) -> float:
        return cross(u_minus, np.asarray(point) - s)

    def f_plus(point: np.ndarray) -> float:
        return -cross(u_plus, np.asarray(point) - s)

    return f_minus, f_plus


def clip_halfplane(
    polygon: ArrayLike,
    signed_value: Callable[[np.ndarray], float],
    eps: float = 1e-9,
) -> np.ndarray:
    """使用 Sutherland-Hodgman 算法，将多边形裁剪到 signed_value >= 0 一侧。"""
    poly = as_points(polygon)
    if len(poly) == 0:
        return poly
    out: list[np.ndarray] = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        fp = float(signed_value(p))
        fq = float(signed_value(q))
        p_in = fp >= -eps
        q_in = fq >= -eps
        if p_in:
            out.append(p.copy())
        if p_in != q_in:
            denom = fp - fq
            if abs(denom) > max(np.finfo(float).eps, eps * 1e-6):
                t = fp / denom
                t = min(1.0, max(0.0, t))
                out.append(p + t * (q - p))
    return np.asarray(out, dtype=float).reshape((-1, 2)) if out else np.empty((0, 2))


def clean_polygon(
    polygon: ArrayLike,
    eps_geo: float = 1e-8,
    eps_col: float = 1e-8,
) -> np.ndarray:
    """删除相邻重复点和近共线的中间顶点。"""
    poly = as_points(polygon)
    if len(poly) <= 1:
        return poly

    unique: list[np.ndarray] = []
    for point in poly:
        if not unique or np.linalg.norm(point - unique[-1]) > eps_geo:
            unique.append(point.copy())
    if len(unique) > 1 and np.linalg.norm(unique[0] - unique[-1]) <= eps_geo:
        unique.pop()
    poly = np.asarray(unique, dtype=float).reshape((-1, 2)) if unique else np.empty((0, 2))
    if len(poly) <= 2:
        return poly

    changed = True
    while changed and len(poly) >= 3:
        changed = False
        keep: list[np.ndarray] = []
        m = len(poly)
        for i in range(m):
            prev = poly[(i - 1) % m]
            cur = poly[i]
            nxt = poly[(i + 1) % m]
            v1 = cur - prev
            v2 = nxt - cur
            scale = max(1.0, float(np.linalg.norm(v1) * np.linalg.norm(v2)))
            collinear = abs(cross(v1, v2)) <= eps_col * scale
            # 对完全退化成直线的情形保留两个端点。只有位于相邻点之间的
            # 中间点才可删除；否则它可能是循环表示中的退化凸包端点。
            between = float(np.dot(prev - cur, nxt - cur)) <= eps_geo * eps_geo
            if np.linalg.norm(v1) <= eps_geo or np.linalg.norm(v2) <= eps_geo or (collinear and between):
                changed = True
            else:
                keep.append(cur)
        if not keep:
            return np.empty((0, 2))
        poly = np.asarray(keep, dtype=float)
    if len(poly) >= 3 and signed_area(poly) < 0:
        poly = poly[::-1].copy()
    return poly


def signed_area(polygon: ArrayLike) -> float:
    poly = as_points(polygon)
    if len(poly) < 3:
        return 0.0
    return 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))


def polygon_area(polygon: ArrayLike) -> float:
    return abs(signed_area(polygon))


def polygon_perimeter(polygon: ArrayLike) -> float:
    poly = as_points(polygon)
    if len(poly) < 2:
        return 0.0
    if len(poly) == 2:
        return 2.0 * float(np.linalg.norm(poly[1] - poly[0]))
    return float(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1).sum())


def polygon_diameter(polygon: ArrayLike) -> tuple[float, tuple[np.ndarray, np.ndarray] | None]:
    """用逆时针凸多边形上的旋转卡壳算法返回直径。"""
    poly = clean_polygon(polygon)
    n = len(poly)
    if n == 0:
        return 0.0, None
    if n == 1:
        return 0.0, (poly[0].copy(), poly[0].copy())
    if n == 2:
        return float(np.linalg.norm(poly[1] - poly[0])), (poly[0].copy(), poly[1].copy())

    best_d2 = -1.0
    best = (poly[0], poly[1])

    def update(i: int, j: int) -> None:
        nonlocal best_d2, best
        d = poly[i] - poly[j]
        d2 = float(np.dot(d, d))
        if d2 > best_d2:
            best_d2 = d2
            best = (poly[i].copy(), poly[j].copy())

    j = 1
    for i in range(n):
        ni = (i + 1) % n
        edge = poly[ni] - poly[i]
        guard = 0
        while guard < n:
            nj = (j + 1) % n
            cur = abs(cross(edge, poly[j] - poly[i]))
            nxt = abs(cross(edge, poly[nj] - poly[i]))
            if nxt > cur + 1e-12 * max(1.0, float(np.linalg.norm(edge)) * max(1.0, float(np.linalg.norm(poly[nj] - poly[i])))):
                j = nj
                guard += 1
            else:
                break
        update(i, j)
        update(ni, j)
    return math.sqrt(max(0.0, best_d2)), best


def _circle_from_boundary(boundary: list[np.ndarray], tol: GeometryTolerance) -> tuple[np.ndarray, float]:
    if not boundary:
        return np.zeros(2, dtype=float), 0.0
    if len(boundary) == 1:
        return boundary[0].copy(), 0.0
    if len(boundary) == 2:
        center = (boundary[0] + boundary[1]) / 2.0
        return center, float(np.linalg.norm(boundary[0] - center))

    a, b, c = boundary[:3]
    scale = max(1.0, float(np.linalg.norm(a)), float(np.linalg.norm(b)), float(np.linalg.norm(c)))
    pair_candidates: list[tuple[float, np.ndarray, float]] = []
    for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        center = (boundary[i] + boundary[j]) / 2.0
        radius = float(np.linalg.norm(boundary[i] - center))
        if np.linalg.norm(boundary[k] - center) <= radius + tol.length(scale):
            pair_candidates.append((radius, center, radius))
    if pair_candidates:
        _, center, radius = min(pair_candidates, key=lambda item: item[0])
        return center, radius

    ab = b - a
    ac = c - a
    determinant = 2.0 * cross(ab, ac)
    if abs(determinant) <= tol.area(scale):
        farthest = max(((a, b), (a, c), (b, c)), key=lambda pair: float(np.dot(pair[0] - pair[1], pair[0] - pair[1])))
        return _circle_from_boundary([farthest[0], farthest[1]], tol)
    ab2 = float(np.dot(ab, ab))
    ac2 = float(np.dot(ac, ac))
    center = a + np.array(
        [(ac[1] * ab2 - ab[1] * ac2) / determinant, (ab[0] * ac2 - ac[0] * ab2) / determinant]
    )
    return center, float(np.linalg.norm(center - a))


def minimum_enclosing_circle(points: ArrayLike, tol: GeometryTolerance = DEFAULT_TOL) -> tuple[np.ndarray, float]:
    """使用确定性的 Welzl 算法计算最小包围圆，并处理共线退化。"""
    pts = as_points(points)
    if len(pts) == 0:
        return np.zeros(2, dtype=float), 0.0
    rng = np.random.default_rng(0)
    shuffled = [p.copy() for p in pts[rng.permutation(len(pts))]]
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(shuffled) + 100))

    def contains(circle: tuple[np.ndarray, float], point: np.ndarray) -> bool:
        center, radius = circle
        scale = max(1.0, radius, float(np.linalg.norm(center)), float(np.linalg.norm(point)))
        return float(np.linalg.norm(point - center)) <= radius + tol.length(scale)

    def welzl(index: int, boundary: list[np.ndarray]) -> tuple[np.ndarray, float]:
        if index == 0 or len(boundary) == 3:
            return _circle_from_boundary(boundary, tol)
        point = shuffled[index - 1]
        circle = welzl(index - 1, boundary)
        if contains(circle, point):
            return circle
        return welzl(index - 1, boundary + [point])

    center, radius = welzl(len(shuffled), [])
    # 最后扩大到覆盖所有输入点，避免递归计算后出现一个机器精度单位的遗漏。
    if len(pts):
        radius = max(radius, float(np.max(np.linalg.norm(pts - center, axis=1))))
    return center, radius


def coverage_tolerance(radius: float, n: int, scale_m: float, tol: GeometryTolerance = DEFAULT_TOL) -> float:
    """计算 D == 2R_min 的判定容差，其中包含外接多边形误差。"""
    h = float(radius) * (1.0 / math.cos(math.pi / int(n)) - 1.0)
    return max(4.0 * h, tol.abs_m) + tol.rel * max(1.0, abs(float(scale_m)))


def coverage_decision(
    diameter_m: float,
    mec_radius_m: float,
    radius: float = 1800.0,
    n: int = 3600,
    tol: GeometryTolerance = DEFAULT_TOL,
) -> dict[str, float | bool | str]:
    gap = float(diameter_m) - 2.0 * float(mec_radius_m)
    eps = coverage_tolerance(radius, n, max(abs(diameter_m), abs(2.0 * mec_radius_m)), tol)
    if abs(gap) <= eps:
        status = "numeric_boundary"
        possible = True
    elif gap > eps:
        status = "algorithm_inconsistency"
        possible = False
    else:
        status = "not_covered"
        possible = False
    return {"coverage_possible": possible, "coverage_gap_m": gap, "eps_cov_m": eps, "coverage_status": status}


def point_in_convex_polygon(point: Sequence[float], polygon: ArrayLike, eps: float = 1e-8) -> bool:
    """包含边界的凸多边形点包含判定，同时支持退化情形。"""
    p = np.asarray(point, dtype=float)
    poly = clean_polygon(polygon, eps_geo=eps, eps_col=eps)
    if len(poly) == 0:
        return False
    if len(poly) == 1:
        return bool(np.linalg.norm(p - poly[0]) <= eps)
    if len(poly) == 2:
        v = poly[1] - poly[0]
        return abs(cross(v, p - poly[0])) <= eps * max(1.0, float(np.linalg.norm(v))) and float(np.dot(p - poly[0], p - poly[1])) <= eps**2
    signs = [cross(poly[(i + 1) % len(poly)] - poly[i], p - poly[i]) for i in range(len(poly))]
    scale = max(1.0, float(np.linalg.norm(p)), float(np.max(np.linalg.norm(poly, axis=1))))
    threshold = eps * scale
    return min(signs) >= -threshold or max(signs) <= threshold


def solve_region(
    points: Sequence[Mapping[str, float] | Sequence[float]],
    radius: float = 1800.0,
    delta_deg: float = 1.0,
    n: int = 3600,
    tol: GeometryTolerance = DEFAULT_TOL,
) -> dict:
    """构造并测量圆盘裁剪后的主可行区域。"""
    parsed: list[tuple[float, float, float]] = []
    for item in points:
        if isinstance(item, Mapping):
            bearing = item.get("bearing_deg", item.get("svd_deg"))
            if bearing is None:
                raise ValueError("每个检测点都必须提供 bearing_deg")
            parsed.append((float(item["x"]), float(item["y"]), float(bearing)))
        else:
            if len(item) != 3:
                raise ValueError("检测点元组必须是 (x, y, bearing_deg)")
            parsed.append((float(item[0]), float(item[1]), float(item[2])))
    for x, y, bearing in parsed:
        if not all(math.isfinite(v) for v in (x, y, bearing)):
            raise ValueError("检测点坐标和示向度必须是有限数")

    flags: list[str] = []
    point_scale = max([radius] + [math.hypot(x, y) for x, y, _ in parsed])
    eps_clip = tol.length(point_scale)
    polygon = build_regular_ngon(radius, n)
    for i, (x, y, theta) in enumerate(parsed, start=1):
        sensor = np.array([x, y], dtype=float)
        f_minus, f_plus = bearing_halfplane_functions(sensor, theta, delta_deg)
        polygon = clip_halfplane(polygon, f_minus, eps_clip)
        polygon = clean_polygon(polygon, eps_geo=eps_clip, eps_col=tol.area(point_scale))
        polygon = clip_halfplane(polygon, f_plus, eps_clip)
        polygon = clean_polygon(polygon, eps_geo=eps_clip, eps_col=tol.area(point_scale))
        if len(polygon) == 0:
            flags.append("EMPTY_REGION_OBSERVATION_INCONSISTENT_OR_NUMERIC")
            break

    for i, first in enumerate(parsed):
        for second in parsed[i + 1 :]:
            if math.hypot(first[0] - second[0], first[1] - second[1]) <= eps_clip:
                flags.append("DUPLICATE_DETECTION_LOCATION")
                break

    polygon = clean_polygon(polygon, eps_geo=eps_clip, eps_col=tol.area(point_scale))
    if len(polygon) == 0:
        return {
            "region_type": "disk_clipped",
            "is_bounded": True,
            "num_vertices": 0,
            "vertices": [],
            "area_m2": None,
            "perimeter_m": None,
            "diameter_m": None,
            "diameter_pair": None,
            "mec_center": None,
            "mec_radius_m": None,
            "mec_diameter_m": None,
            "coverage_possible": False,
            "coverage_gap_m": None,
            "eps_cov_m": coverage_tolerance(radius, n, point_scale, tol),
            "coverage_status": "empty_region",
            "guaranteed_clearable": False,
            "ready_to_clear": False,
            "flags": flags,
        }

    diameter, pair = polygon_diameter(polygon)
    center, mec_radius = minimum_enclosing_circle(polygon, tol)
    decision = coverage_decision(diameter, mec_radius, radius, n, tol)
    if decision["coverage_status"] == "algorithm_inconsistency":
        flags.append("ALGORITHM_INCONSISTENCY_D_GREATER_THAN_MEC_DIAMETER")
    if decision["coverage_status"] == "numeric_boundary":
        flags.append("NUMERICAL_BOUNDARY_WITHIN_COVERAGE_TOLERANCE")
    if len(polygon) < 3:
        flags.append("DEGENERATE_POLYGON")
    result = {
        "region_type": "disk_clipped",
        "is_bounded": True,
        "num_vertices": int(len(polygon)),
        "vertices": polygon.tolist(),
        "area_m2": polygon_area(polygon),
        "perimeter_m": polygon_perimeter(polygon),
        "diameter_m": diameter,
        "diameter_pair": [pair[0].tolist(), pair[1].tolist()] if pair else None,
        "mec_center": center.tolist(),
        "mec_radius_m": mec_radius,
        "mec_diameter_m": 2.0 * mec_radius,
        **decision,
        "guaranteed_clearable": bool(mec_radius <= CLEAR_RADIUS_M),
        "ready_to_clear": bool(mec_radius <= SAFE_CLEAR_RADIUS_M),
        "flags": flags,
    }
    return result


def solve_case(case: Mapping) -> dict:
    """求解一个 JSON 案例，并返回可序列化为 JSON 的结果。"""
    case_id = str(case.get("case_id", "unnamed"))
    radius = float(case.get("R", 1800.0))
    delta_deg = float(case.get("delta_deg", 1.0))
    n = int(case.get("N", 3600))
    points = case.get("points", [])
    if not isinstance(points, list):
        raise ValueError("points 必须是列表")
    main = solve_region(points, radius, delta_deg, n)
    return {
        "case_id": case_id,
        "input_summary": {"R": radius, "delta_deg": delta_deg, "N": n, "num_points": len(points)},
        "main": main,
        "notes": [
            "主区域使用外接正 N 边形，因此是精确圆盘裁剪可行区域的保守外逼近。",
            "主输出有意不包含纯扇形交：有限包围盒若没有自适应或无界性分析，不能证明区域有界。",
        ],
    }
