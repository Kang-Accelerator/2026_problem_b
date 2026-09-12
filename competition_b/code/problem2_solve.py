"""问题 2 第一版命令行求解、候选区域整理和图像输出。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np

try:
    from .problem2_model import (
        Problem2Config,
        has_observation_axis_symmetry,
        observation_axis_basis,
        parse_case_and_geometry,
        parse_config,
        reflect_about_observation_axis,
        sample_validation_source_points,
        source_distance,
        symmetry_preserving_subsample,
        thales_reference_point,
    )
    from .problem2_score import certify_receive, certify_receive_cell, score_candidate
except ImportError:  # 支持直接执行 competition_b/code/problem2_solve.py。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from problem2_model import (  # type: ignore
        Problem2Config,
        has_observation_axis_symmetry,
        observation_axis_basis,
        parse_case_and_geometry,
        parse_config,
        reflect_about_observation_axis,
        sample_validation_source_points,
        source_distance,
        symmetry_preserving_subsample,
        thales_reference_point,
    )
    from problem2_score import certify_receive, certify_receive_cell, score_candidate  # type: ignore


class ChineseArgumentParser(argparse.ArgumentParser):
    """把 argparse 的帮助提示改成中文。"""

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法：")

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法：").replace("options:", "选项：")


def _candidate_grid(config: Problem2Config) -> tuple[list[dict[str, Any]], float]:
    """在首次示向局部坐标系生成关于中轴线成对的候选粗网格。"""

    raw_step = config.candidate_step_m
    extent = config.budget_b_m
    approx = (2.0 * extent / raw_step + 1.0) ** 2
    stride = max(1, int(math.ceil(math.sqrt(approx / config.max_candidates))))
    step = raw_step * stride
    candidates: list[dict[str, Any]] = []
    if has_observation_axis_symmetry(config):
        tangent, normal = observation_axis_basis(config)
        half_count = int(math.floor(extent / step + 1e-12))
        local_values = np.arange(-half_count, half_count + 1, dtype=float) * step
        raw_points = (
            (ix, iy, config.s1 + float(along) * tangent + float(across) * normal)
            for ix, along in enumerate(local_values)
            for iy, across in enumerate(local_values)
        )
    else:
        x0 = math.floor((config.s1[0] - extent) / step) * step
        x1 = math.ceil((config.s1[0] + extent) / step) * step
        y0 = math.floor((config.s1[1] - extent) / step) * step
        y1 = math.ceil((config.s1[1] + extent) / step) * step
        raw_points = (
            (ix, iy, np.array([float(x), float(y)], dtype=float))
            for ix, x in enumerate(np.arange(x0, x1 + step * 0.5, step))
            for iy, y in enumerate(np.arange(y0, y1 + step * 0.5, step))
        )
    for ix, iy, point in raw_points:
        distance = float(np.linalg.norm(point - config.s1))
        if distance <= config.budget_b_m + 1e-7:
            candidates.append({
                "point": point,
                "grid_i": ix,
                "grid_j": iy,
                "refined": False,
                "point_origin": "coarse_grid_formula",
            })
    if not any(float(np.linalg.norm(item["point"] - config.s1)) <= 1e-7 for item in candidates):
        candidates.append({"point": config.s1.copy(), "grid_i": 0, "grid_j": 0, "refined": False, "point_origin": "coarse_grid_formula"})
    return candidates, step


def _local_refinement(config: Problem2Config, coarse: list[dict[str, Any]], step: float) -> list[dict[str, Any]]:
    """围绕优先粗网格点成对生成局部十字细化点。"""

    if not coarse:
        return []
    point_map = {
        (round(float(item["point"][0]), 7), round(float(item["point"][1]), 7)): item
        for item in coarse
    }
    top: list[dict[str, Any]] = []
    top_keys: set[tuple[float, float]] = set()
    for item in coarse:
        orbit = [item]
        if has_observation_axis_symmetry(config):
            mirror = reflect_about_observation_axis(config, item["point"])
            mirror_item = point_map.get((round(float(mirror[0]), 7), round(float(mirror[1]), 7)))
            if mirror_item is not None and mirror_item is not item:
                orbit.append(mirror_item)
        new_items = [candidate for candidate in orbit if (round(float(candidate["point"][0]), 7), round(float(candidate["point"][1]), 7)) not in top_keys]
        if top and len(top) + len(new_items) > 6:
            continue
        for candidate in new_items:
            key = (round(float(candidate["point"][0]), 7), round(float(candidate["point"][1]), 7))
            top_keys.add(key)
            top.append(candidate)
        if len(top) >= min(6, len(coarse)):
            break
    radius = step / max(2.0, float(config.candidate_b_substeps) / 2.0)
    if has_observation_axis_symmetry(config):
        tangent, normal = observation_axis_basis(config)
        offsets = (np.zeros(2), radius * tangent, -radius * tangent, radius * normal, -radius * normal)
    else:
        offsets = (
            np.zeros(2), np.array([radius, 0.0]), np.array([-radius, 0.0]),
            np.array([0.0, radius]), np.array([0.0, -radius]),
        )
    refined: list[dict[str, Any]] = []
    # 粗网格中已有的坐标不重复写入“局部细化”记录。
    seen: set[tuple[int, int]] = {
        (round(float(item["point"][0]) * 1e6), round(float(item["point"][1]) * 1e6))
        for item in coarse
    }
    for item in top:
        base = item["point"]
        for offset in offsets:
            point = np.asarray(base, dtype=float) + offset
            if float(np.linalg.norm(point - config.s1)) > config.budget_b_m + 1e-7:
                continue
            key = (round(float(point[0]) * 1e6), round(float(point[1]) * 1e6))
            if key in seen:
                continue
            seen.add(key)
            refined.append({
                "point": point,
                "grid_i": item["grid_i"],
                "grid_j": item["grid_j"],
                "refined": True,
                "point_origin": "local_cross_formula",
            })
    return refined


def _symmetry_closed_prefix(config: Problem2Config, records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """截取待认证点时不拆开严格对称案例中的镜像点对。"""

    if not has_observation_axis_symmetry(config):
        return records[:limit]
    point_map = {
        (round(float(record["point"][0]), 7), round(float(record["point"][1]), 7)): record
        for record in records
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for record in records:
        orbit = [record]
        mirror = reflect_about_observation_axis(config, record["point"])
        mirror_record = point_map.get((round(float(mirror[0]), 7), round(float(mirror[1]), 7)))
        if mirror_record is not None and mirror_record is not record:
            orbit.append(mirror_record)
        new_records = [item for item in orbit if id(item) not in selected_ids]
        if selected and len(selected) + len(new_records) > limit:
            continue
        for item in new_records:
            selected_ids.add(id(item))
            selected.append(item)
        if len(selected) >= min(limit, len(records)):
            break
    return selected


def _symmetry_audit(config: Problem2Config, records: list[dict[str, Any]]) -> dict[str, Any]:
    """核验搜索记录的镜像闭合、认证状态和独立评分一致性。"""

    if not has_observation_axis_symmetry(config):
        return {"status": "not_applicable", "reason": "input geometry or error grid is not axis-symmetric"}
    lookup = {
        (round(float(record["point"][0]), 6), round(float(record["point"][1]), 6)): record
        for record in records
    }
    missing = 0
    certification_mismatches = 0
    score_differences: list[float] = []
    for record in records:
        mirror = reflect_about_observation_axis(config, record["point"])
        partner = lookup.get((round(float(mirror[0]), 6), round(float(mirror[1]), 6)))
        if partner is None:
            missing += 1
            continue
        if record.get("cert_status") != partner.get("cert_status"):
            certification_mismatches += 1
        left_j = float(record.get("j_lambda", float("inf")))
        right_j = float(partner.get("j_lambda", float("inf")))
        if math.isfinite(left_j) and math.isfinite(right_j):
            score_differences.append(abs(left_j - right_j))
    max_difference = max(score_differences, default=0.0)
    passed = missing == 0 and certification_mismatches == 0 and max_difference <= 1e-10
    return {
        "status": "passed" if passed else "failed",
        "point_count": len(records),
        "missing_mirror_count": missing,
        "certification_mismatch_count": certification_mismatches,
        "max_mirror_J_abs_difference": max_difference,
        "score_tolerance": 1e-10,
        "point_values_independently_evaluated": True,
    }


def _attach_certification(
    config: Problem2Config,
    outer_polygon: np.ndarray,
    source_points: np.ndarray,
    record: dict[str, Any],
) -> dict[str, Any]:
    cert = certify_receive(config, outer_polygon, record["point"], source_points)
    return {
        **record,
        "cert_status": cert["status"],
        "continuous_worst_case_certified": cert["continuous_worst_case_certified"],
        "cert_margin_m": cert["cert_margin_m"],
        "certification_budget_used": cert["budget_used"],
        "certification_evidence": cert["evidence"],
    }


def _record_public(record: dict[str, Any]) -> dict[str, Any]:
    """去掉内部 ndarray 后形成候选评分表行。"""

    public = {key: value for key, value in record.items() if key != "point"}
    public["x"] = float(record["point"][0])
    public["y"] = float(record["point"][1])
    return public


def _cell_stencil(bounds: Iterable[float]) -> np.ndarray:
    """矩形中心、四角和四条边中点，共 9 个评分检查点。"""

    x0, x1, y0, y1 = map(float, bounds)
    xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return np.asarray([
        [xm, ym],
        [x0, y0], [x1, y0], [x1, y1], [x0, y1],
        [xm, y0], [x1, ym], [xm, y1], [x0, ym],
    ], dtype=float)


def _region_topology(
    occupancy: set[tuple[int, int]], origin: np.ndarray, atom_size: float
) -> dict[str, Any]:
    """从统一细网格并集提取外边界，消除所有内部重复边。"""

    if not occupancy:
        return {"components": 0, "holes": 0, "has_holes": False, "boundary_segments": []}
    unseen = set(occupancy)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            i, j = stack.pop()
            for neighbor in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)

    min_i, max_i = min(i for i, _ in occupancy) - 1, max(i for i, _ in occupancy) + 1
    min_j, max_j = min(j for _, j in occupancy) - 1, max(j for _, j in occupancy) + 1
    empty = {(i, j) for i in range(min_i, max_i + 1) for j in range(min_j, max_j + 1) if (i, j) not in occupancy}
    outside: set[tuple[int, int]] = set()
    stack = [(min_i, min_j)]
    while stack:
        cell = stack.pop()
        if cell in outside or cell not in empty:
            continue
        outside.add(cell)
        i, j = cell
        for neighbor in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if min_i <= neighbor[0] <= max_i and min_j <= neighbor[1] <= max_j:
                stack.append(neighbor)
    enclosed = empty - outside
    holes = 0
    while enclosed:
        holes += 1
        stack = [enclosed.pop()]
        while stack:
            i, j = stack.pop()
            for neighbor in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if neighbor in enclosed:
                    enclosed.remove(neighbor)
                    stack.append(neighbor)

    def point(i: int, j: int) -> list[float]:
        return [float(origin[0] + i * atom_size), float(origin[1] + j * atom_size)]

    segments: list[list[list[float]]] = []
    for i, j in sorted(occupancy):
        if (i, j - 1) not in occupancy:
            segments.append([point(i, j), point(i + 1, j)])
        if (i + 1, j) not in occupancy:
            segments.append([point(i + 1, j), point(i + 1, j + 1)])
        if (i, j + 1) not in occupancy:
            segments.append([point(i + 1, j + 1), point(i, j + 1)])
        if (i - 1, j) not in occupancy:
            segments.append([point(i, j + 1), point(i, j)])
    return {"components": components, "holes": holes, "has_holes": holes > 0, "boundary_segments": segments}


def point_in_candidate_region(region: dict[str, Any], point: Iterable[float], tol: float = 1e-9) -> bool:
    """判断点是否位于已保留的自适应单元并集中。"""

    p = np.asarray(tuple(point), dtype=float)
    for cell in region.get("cells", []):
        vertices = cell.get("vertices")
        if vertices is not None:
            polygon = np.asarray(vertices, dtype=float)
            edges = np.roll(polygon, -1, axis=0) - polygon
            offsets = p[None, :] - polygon
            crosses = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
            if bool(np.all(crosses >= -tol) or np.all(crosses <= tol)):
                return True
            continue
        x, y = float(p[0]), float(p[1])
        bounds = cell["bounds"]
        if (
            float(bounds[0]) - tol <= x <= float(bounds[1]) + tol
            and float(bounds[2]) - tol <= y <= float(bounds[3]) + tol
        ):
            return True
    return False


def _adaptive_candidate_region(
    config: Problem2Config,
    outer_polygon: np.ndarray,
    source_points: np.ndarray,
    score_sources: np.ndarray,
    records: list[dict[str, Any]],
    best_j: float,
) -> dict[str, Any]:
    """用实际空间评分和整单元接收认证构造优先推荐区域。"""

    finite_best = math.isfinite(best_j)
    threshold = float(best_j + config.focus_epsilon_j) if finite_best else None
    base_size = float(config.region_base_cell_m)
    atom_size = float(config.region_min_cell_m)
    axis_symmetric = has_observation_axis_symmetry(config)
    if axis_symmetric:
        tangent, normal = observation_axis_basis(config)
        basis = np.column_stack((tangent, normal))
        half_cells = int(math.ceil(config.budget_b_m / base_size))
        origin = np.asarray([-(half_cells + 0.5) * base_size] * 2, dtype=float)

        def to_frame(point: np.ndarray) -> np.ndarray:
            return basis.T @ (np.asarray(point, dtype=float) - config.s1)

        def to_world(point: np.ndarray) -> np.ndarray:
            return config.s1 + basis @ np.asarray(point, dtype=float)
    else:
        origin = np.floor((config.s1 - config.budget_b_m) / base_size) * base_size - base_size / 2.0

        def to_frame(point: np.ndarray) -> np.ndarray:
            return np.asarray(point, dtype=float)

        def to_world(point: np.ndarray) -> np.ndarray:
            return np.asarray(point, dtype=float)
    point_cache: dict[tuple[float, float], dict[str, Any]] = {}
    accepted: list[dict[str, Any]] = []
    rejected_cells = 0
    subdivided_cells = 0
    cert_budget_used = 0

    for record in records:
        point = np.asarray(record["point"], dtype=float)
        key = (round(float(point[0]), 8), round(float(point[1]), 8))
        distance = float(np.linalg.norm(point - config.s1))
        margin = float(record.get("margin", float("inf")))
        j_value = float(record.get("j_lambda", float("inf")))
        if distance > config.budget_b_m + 1e-7:
            status = "outside_budget"
        elif margin > 1e-6:
            status = "receive_failed"
        elif math.isfinite(j_value):
            status = "evaluated"
        else:
            status = "score_unavailable"
        point_cache[key] = {
            "x": key[0], "y": key[1], "status": status,
            "point_origin": record.get("point_origin", "search_point_formula"),
            "j_lambda": j_value if math.isfinite(j_value) else None,
            "sample_worst_rmin_m": record.get("sample_worst_rmin_m"),
            "receive_margin_m": margin if math.isfinite(margin) else None,
            "score_passed": bool(finite_best and math.isfinite(j_value) and j_value <= float(threshold) + 1e-12),
        }

    def evaluate(point: np.ndarray) -> dict[str, Any]:
        key = (round(float(point[0]), 8), round(float(point[1]), 8))
        if key in point_cache:
            return point_cache[key]
        distance = float(np.linalg.norm(point - config.s1))
        if distance > config.budget_b_m + 1e-7:
            result = {"x": key[0], "y": key[1], "status": "outside_budget", "point_origin": "adaptive_cell_9_point_stencil", "j_lambda": None, "receive_margin_m": None, "score_passed": False}
        else:
            margin = float(np.nan_to_num(score_margin(config, point, score_sources), nan=float("inf")))
            if margin > 1e-6:
                result = {"x": key[0], "y": key[1], "status": "receive_failed", "point_origin": "adaptive_cell_9_point_stencil", "j_lambda": None, "receive_margin_m": margin, "score_passed": False}
            else:
                score = score_candidate(config, outer_polygon, point, score_sources)
                j_value = float(score["j_lambda"])
                finite = math.isfinite(j_value)
                result = {
                    "x": key[0], "y": key[1],
                    "status": "evaluated" if finite else "score_unavailable",
                    "point_origin": "adaptive_cell_9_point_stencil",
                    "j_lambda": j_value if finite else None,
                    "sample_worst_rmin_m": score.get("sample_worst_rmin_m"),
                    "receive_margin_m": margin,
                    "score_passed": bool(finite_best and finite and j_value <= float(threshold) + 1e-12),
                }
        point_cache[key] = result
        return result

    def inspect(bounds: list[float], depth: int, base_i: int, base_j: int) -> bool:
        nonlocal rejected_cells, subdivided_cells, cert_budget_used
        frame_stencil = _cell_stencil(bounds)
        world_stencil = np.asarray([to_world(point) for point in frame_stencil], dtype=float)
        checks = [evaluate(point) for point in world_stencil]
        passes = [bool(check["score_passed"]) for check in checks]
        all_score_pass = all(passes)
        any_score_pass = any(passes)
        if all_score_pass:
            x0, x1, y0, y1 = bounds
            frame_vertices = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
            world_vertices = np.asarray([to_world(point) for point in frame_vertices], dtype=float)
            world_bounds = [
                float(np.min(world_vertices[:, 0])), float(np.max(world_vertices[:, 0])),
                float(np.min(world_vertices[:, 1])), float(np.max(world_vertices[:, 1])),
            ]
            cert = certify_receive_cell(
                config,
                outer_polygon,
                world_bounds,
                source_points,
                corners=world_vertices if axis_symmetric else None,
            )
            cert_budget_used += int(cert.get("budget_used", 0))
            if cert["status"] == "passed":
                values = [float(check["j_lambda"]) for check in checks]
                world_center = to_world(np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0]))
                accepted.append({
                    "bounds": world_bounds,
                    "local_bounds": [float(v) for v in bounds],
                    "vertices": world_vertices.tolist(),
                    "center": {"x": float(world_center[0]), "y": float(world_center[1])},
                    "size_m": float(bounds[1] - bounds[0]),
                    "coordinate_frame": "observation_axis" if axis_symmetric else "world_xy",
                    "depth": depth,
                    "base_i": base_i,
                    "base_j": base_j,
                    "stencil_point_count": len(checks),
                    "j_min": min(values),
                    "j_max": max(values),
                    "score_threshold_J": threshold,
                    "receive_cell_status": "passed",
                    "receive_cert_margin_m": cert.get("cert_margin_m"),
                    "receive_certification_budget_used": cert.get("budget_used", 0),
                })
                return True
        if any_score_pass and depth < config.region_max_depth:
            subdivided_cells += 1
            x0, x1, y0, y1 = bounds
            xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            touched = False
            for child in ([x0, xm, y0, ym], [xm, x1, y0, ym], [xm, x1, ym, y1], [x0, xm, ym, y1]):
                touched = inspect(list(child), depth + 1, base_i, base_j) or touched
            return touched or any_score_pass
        rejected_cells += 1
        return any_score_pass

    seed_records = [
        record for record in records
        if record.get("cert_status") == "passed"
        and math.isfinite(record.get("j_lambda", float("inf")))
        and finite_best
        and float(record["j_lambda"]) <= best_j + config.epsilon_j + 1e-12
    ]
    queued: set[tuple[int, int]] = set()
    for record in seed_records:
        point = to_frame(np.asarray(record["point"], dtype=float))
        ux = float((point[0] - origin[0]) / base_size)
        uy = float((point[1] - origin[1]) / base_size)
        ix = {int(math.floor(ux))}
        iy = {int(math.floor(uy))}
        if abs(ux - round(ux)) <= 1e-9:
            ix.add(int(round(ux)) - 1)
        if abs(uy - round(uy)) <= 1e-9:
            iy.add(int(round(uy)) - 1)
        for i in sorted(ix):
            for j in sorted(iy):
                key = (i, j)
                queued.add(key)

    def mirror_key(key: tuple[int, int]) -> tuple[int, int]:
        i, j = key
        lower_v = float(origin[1] + j * base_size)
        mirrored_lower_v = -(lower_v + base_size)
        return i, int(round((mirrored_lower_v - origin[1]) / base_size))

    if axis_symmetric:
        queued.update(mirror_key(key) for key in tuple(queued))

    remaining = set(queued)
    pending_orbits: list[list[tuple[int, int]]] = []
    while remaining:
        key = min(remaining)
        mate = mirror_key(key) if axis_symmetric else key
        orbit = [key]
        remaining.remove(key)
        if mate != key and mate in remaining:
            orbit.append(mate)
            remaining.remove(mate)
        pending_orbits.append(orbit)
    examined: set[tuple[int, int]] = set()
    for orbit in pending_orbits:
        if len(examined) + len(orbit) > config.region_max_base_cells:
            break
        for i, j in orbit:
            examined.add((i, j))
            x0, y0 = origin + np.array([i * base_size, j * base_size])
            inspect([x0, x0 + base_size, y0, y0 + base_size], 0, i, j)

    occupancy: set[tuple[int, int]] = set()
    for cell in accepted:
        x0, x1, y0, y1 = cell["local_bounds"]
        i0 = int(round((x0 - origin[0]) / atom_size))
        i1 = int(round((x1 - origin[0]) / atom_size))
        j0 = int(round((y0 - origin[1]) / atom_size))
        j1 = int(round((y1 - origin[1]) / atom_size))
        occupancy.update((i, j) for i in range(i0, i1) for j in range(j0, j1))
    topology = _region_topology(occupancy, origin, atom_size)
    if axis_symmetric:
        topology["boundary_segments"] = [
            [to_world(np.asarray(start)).tolist(), to_world(np.asarray(end)).tolist()]
            for start, end in topology["boundary_segments"]
        ]
    evaluated = sorted(point_cache.values(), key=lambda item: (item["x"], item["y"]))
    return {
        "region_semantics": "adaptive_cell_union_discrete_score_approximation",
        "point_generation_policy": "every evaluated point is a coarse-grid, local-cross, or adaptive 9-point stencil coordinate and is independently scored; no placeholder or interpolated points",
        "recommendation_status": "available" if accepted else "empty",
        "J_best_passed_search_sample": float(best_j) if finite_best else None,
        "J_best_definition": "minimum J among passed coarse/local search samples, rescored with the same region score sources",
        "focus_epsilon_J": float(config.focus_epsilon_j),
        "score_threshold_J": threshold,
        "search_seed_epsilon_J": float(config.epsilon_j),
        "base_cell_size_m": base_size,
        "min_cell_size_m": atom_size,
        "coordinate_frame": "observation_axis" if axis_symmetric else "world_xy",
        "max_subdivision_depth": config.region_max_depth,
        "cells": accepted,
        "atomic_cell_count": len(occupancy),
        "components": topology["components"],
        "holes": topology["holes"],
        "has_holes": topology["has_holes"],
        "boundary_segments": topology["boundary_segments"],
        "evaluated_points": evaluated,
        "evaluated_point_count": len(evaluated),
        "region_score_source_count": int(len(score_sources)),
        "receive_failed_point_count": sum(item["status"] == "receive_failed" for item in evaluated),
        "outside_budget_point_count": sum(item["status"] == "outside_budget" for item in evaluated),
        "base_cells_examined": len(examined),
        "base_cell_limit_hit": len(queued) > config.region_max_base_cells,
        "subdivided_cell_count": subdivided_cells,
        "rejected_cell_count": rejected_cells,
        "receive_verification": {
            "status": "passed" if accepted else "not_available",
            "continuous_over_detector_cells": bool(accepted),
            "continuous_over_conservative_source_outer": bool(accepted),
            "certification_budget_used": cert_budget_used,
            "scope": "accepted rectangle union only",
        },
        "score_verification": {
            "status": "sampled_passed" if accepted else "not_available",
            "continuous_over_detector_cells": False,
            "stencil": "center + four corners + four edge midpoints",
            "points_per_cell": 9,
            "limitation": "单元内评分只按9点模板验证；红框是离散近似，不是连续评分证明。",
        },
        "verified_points": [item for item in evaluated if item.get("score_passed")],
    }


def _best_single_point(records: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    sampled = [record for record in records if math.isfinite(record.get("j_lambda", float("inf")))]
    sampled_best = min(sampled, key=lambda item: item["j_lambda"]) if sampled else None
    passed = [record for record in sampled if record.get("cert_status") == "passed"]
    if not passed:
        return None, sampled_best
    return min(passed, key=lambda item: item["j_lambda"]), sampled_best


def _best_output(record: dict[str, Any] | None, sampled_best: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {
            "position": None,
            "recommendation_status": "no_passed_candidate",
            "message": "离散搜索中没有完成 passed 可靠性认证的候选点；不把 unknown 点当作推荐点。",
            "best_sampled_point": None if sampled_best is None else _record_public(sampled_best),
        }
    result = _record_public(record)
    result["position"] = {"x": result["x"], "y": result["y"]}
    result["recommendation_status"] = "passed"
    return result


def solve_case(case: dict[str, Any], make_plots: bool = False, output_dir: Path | None = None) -> dict[str, Any]:
    """求解一个问题2案例，默认不生成图以便测试快速运行。"""

    started = time.perf_counter()
    config, source_points, outer_polygon, source_summary = parse_case_and_geometry(case)
    if len(source_points) == 0 or len(outer_polygon) == 0:
        result = _empty_result(config, source_summary, time.perf_counter() - started)
        if make_plots and output_dir is not None:
            _plot_case(case, result, output_dir)
        return result

    candidates, effective_step = _candidate_grid(config)
    source_for_search = symmetry_preserving_subsample(config, source_points, config.search_source_limit)
    all_records: list[dict[str, Any]] = []
    safe_count = 0
    for item in candidates:
        margin = float(np.nan_to_num(score_margin(config, item["point"], source_for_search), nan=float("inf")))
        if margin > 1e-6:
            all_records.append({**item, "margin": margin, "j_lambda": float("inf"), "cert_status": "violated", "continuous_worst_case_certified": False, "cert_margin_m": -margin, "certification_budget_used": 0, "certification_evidence": "样本源中存在保证接收违反点。"})
            continue
        safe_count += 1
        score = score_candidate(config, outer_polygon, item["point"], source_for_search)
        all_records.append({**item, **score})
    ranked = sorted((record for record in all_records if math.isfinite(record.get("j_lambda", float("inf")))), key=lambda item: item["j_lambda"])
    refined_items = _local_refinement(config, ranked, effective_step)
    for item in refined_items:
        margin = score_margin(config, item["point"], source_for_search)
        if margin > 1e-6:
            continue
        score = score_candidate(config, outer_polygon, item["point"], source_for_search)
        all_records.append({**item, **score})
    ranked = sorted((record for record in all_records if math.isfinite(record.get("j_lambda", float("inf")))), key=lambda item: item["j_lambda"])
    certification_records = _symmetry_closed_prefix(config, ranked, min(32, len(ranked)))
    for ranked_record in certification_records:
        updated = _attach_certification(config, outer_polygon, source_points, ranked_record)
        for pos, record in enumerate(all_records):
            if record is ranked_record:
                all_records[pos] = updated
                break
    ranked = sorted((record for record in all_records if math.isfinite(record.get("j_lambda", float("inf")))), key=lambda item: item["j_lambda"])
    best_record, sampled_best = _best_single_point(all_records)
    if best_record is None:
        reference_candidate = sampled_best
    else:
        reference_candidate = best_record
    validation_points = sample_validation_source_points(config, config.validation_source_count)
    if reference_candidate is not None and len(validation_points):
        validation_score = score_candidate(config, outer_polygon, reference_candidate["point"], validation_points, source_limit=len(validation_points))
    else:
        validation_score = {"sample_worst_rmin_m": None, "n_near": 0, "n_direction": 0, "n_no_signal": 0, "n_uncertain": 0, "scenario_count": 0, "source_count": 0, "error_count": len(config.error_grid_deg), "failed_scenarios": [], "mean_miss": None, "geometry_intersections": 0, "j_lambda": None}
    heuristic = thales_reference_point(config, source_points)
    if heuristic["thales_s2"] is not None:
        heuristic_scores = [
            _score_summary(score_candidate(config, outer_polygon, [point["x"], point["y"]], source_for_search))
            for point in heuristic.get("thales_s2_pair", [heuristic["thales_s2"]])
        ]
        heuristic["scores"] = heuristic_scores
        heuristic["score"] = heuristic_scores[0]
    else:
        heuristic["scores"] = []
        heuristic["score"] = None
    best_j = float(best_record["j_lambda"]) if best_record is not None else float("inf")
    region_sources = symmetry_preserving_subsample(config, source_for_search, config.region_source_limit)
    passed_search_records = [
        record for record in all_records
        if record.get("cert_status") == "passed"
        and math.isfinite(record.get("j_lambda", float("inf")))
    ]
    region_seed_records: list[dict[str, Any]] = []
    for seed in passed_search_records:
        region_score = score_candidate(config, outer_polygon, seed["point"], region_sources)
        region_seed_records.append({**seed, **region_score, "cert_status": seed["cert_status"]})
    region_best_j = min((float(record["j_lambda"]) for record in region_seed_records if math.isfinite(record.get("j_lambda", float("inf")))), default=float("inf"))
    region = _adaptive_candidate_region(
        config,
        outer_polygon,
        source_points,
        region_sources,
        region_seed_records,
        region_best_j,
    )
    branches = {"near": 0, "direction": 0, "no_signal": 0, "uncertain": 0}
    failures = 0
    intersections = 0
    for record in all_records:
        for branch in branches:
            key = f"n_{branch}"
            branches[branch] += int(record.get(key, 0) or 0)
        failures += len(record.get("failed_scenarios", []))
        intersections += int(record.get("geometry_intersections", 0) or 0)
    elapsed = time.perf_counter() - started
    symmetry_audit = _symmetry_audit(config, all_records)
    result = {
        "case_id": config.case_id,
        "input_summary": _input_summary(config),
        "possible_source_set": source_summary,
        "best_single_point": _best_output(best_record, sampled_best),
        "candidate_region": region,
        "heuristic_reference": heuristic,
        "observation_model": "conservative_lower_bound_rho=max(1000,d1); no probability prior",
        "sample_worst_rmin": _metric_block(best_record or sampled_best, "sample"),
        "validation_worst_rmin": _metric_block(validation_score, "validation"),
        "best_point_receive_certificate": None if best_record is None else {
            "status": best_record.get("cert_status", "unknown"),
            "scope": "single diagnostic point",
            "cert_margin_m": best_record.get("cert_margin_m"),
            "certification_budget_used": best_record.get("certification_budget_used", 0),
            "evidence": best_record.get("certification_evidence", ""),
            "outer_label": "conservative_outer",
        },
        "candidate_region_receive_verification": region["receive_verification"],
        "candidate_region_score_verification": region["score_verification"],
        "travel_distance_m": None if best_record is None else float(best_record["travel_distance_m"]),
        "travel_time_s": None if best_record is None else float(best_record["travel_distance_m"] / 5.0),
        "solver_wall_time_s": float(elapsed),
        "geometry_intersections": int(intersections),
        "mean_miss": None,
        "branch_statistics": branches,
        "failure_scenarios": failures,
        "diagnostics": {
            "candidate_count": len(candidates),
            "safe_sample_candidate_count": safe_count,
            "scored_candidate_count": len(all_records),
            "certified_candidate_count": sum(record.get("cert_status") == "passed" for record in all_records),
            "unknown_candidate_count": sum(record.get("cert_status") == "unknown" for record in all_records),
            "violated_candidate_count": sum(record.get("cert_status") == "violated" for record in all_records),
            "effective_grid_step_m": effective_step,
            "local_refinement_count": len(refined_items),
            "input_axis_symmetry": has_observation_axis_symmetry(config),
            "source_subsampling": "mirror_orbit_preserving" if has_observation_axis_symmetry(config) else "uniform_flat_index_non_symmetric_input",
            "plotted_point_policy": "all scatter markers come from formula-generated source/search/stencil records; no interpolation or placeholder points",
            "symmetry_audit": symmetry_audit,
            "best_point_continuous_receive_certified": bool(best_record is not None and best_record.get("continuous_worst_case_certified", False)),
            "notes": ["候选区域的接收条件按整矩形充分条件认证；评分按每单元9点模板验证，未完成连续评分或连续平面全局最优认证。"],
        },
        "candidate_score_table": [_record_public(record) for record in sorted(all_records, key=lambda item: item.get("j_lambda", float("inf")))],
    }
    if make_plots and output_dir is not None:
        _plot_case(case, result, output_dir)
    return result


def score_margin(config: Problem2Config, candidate: np.ndarray, source_points: np.ndarray) -> float:
    """轻量向量化保证接收筛选。"""

    if len(source_points) == 0:
        return float("nan")
    d1 = np.linalg.norm(source_points - config.s1[None, :], axis=1)
    r2 = np.linalg.norm(source_points - np.asarray(candidate)[None, :], axis=1)
    return float(np.max(r2 - np.maximum(config.rho_min, d1)))


def _input_summary(config: Problem2Config) -> dict[str, Any]:
    return {
        "S1": {"x": float(config.s1[0]), "y": float(config.s1[1])},
        "theta1_deg": config.theta1_deg,
        "delta_deg": config.delta_deg,
        "target_region": {"cx": config.target.cx, "cy": config.target.cy, "R": config.target.radius},
        "receive_range": [config.rho_min, config.rho_max],
        "near_threshold_m": config.near_threshold_m,
        "budget_B_m": config.budget_b_m,
        "source_grid": {"n_angle": config.source_n_angle, "n_distance": config.source_n_distance},
        "candidate_grid": {
            "step_m": config.candidate_step_m,
            "b_substeps": config.candidate_b_substeps,
            "max_candidates": config.max_candidates,
            "region_base_cell_m": config.region_base_cell_m,
            "region_min_cell_m": config.region_min_cell_m,
            "region_max_depth": config.region_max_depth,
            "region_max_base_cells": config.region_max_base_cells,
            "region_source_limit": config.region_source_limit,
        },
        "error_grid_deg": list(config.error_grid_deg),
        "ngon_edges": config.ngon_edges,
        "random_seed": config.random_seed,
        "objective": {"lambda": config.lambda_weight, "R_ref_m": config.r_ref_m, "B_ref_m": config.b_ref_m, "epsilon_J": config.epsilon_j, "focus_epsilon_J": config.focus_epsilon_j},
    }


def _score_summary(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_worst_rmin_m": score.get("sample_worst_rmin_m"),
        "margin": score.get("margin"),
        "j_lambda": score.get("j_lambda"),
        "n_near": score.get("n_near", 0),
        "n_direction": score.get("n_direction", 0),
        "n_no_signal": score.get("n_no_signal", 0),
        "n_uncertain": score.get("n_uncertain", 0),
    }


def _metric_block(record: dict[str, Any] | None, role: str) -> dict[str, Any]:
    if record is None:
        return {"value_m": None, "role": role, "source_count": 0, "error_count": 0, "scenario_count": 0, "mean_miss": None}
    return {
        "value_m": record.get("sample_worst_rmin_m"),
        "role": role,
        "source_count": record.get("source_count", 0),
        "error_count": record.get("error_count", 0),
        "scenario_count": record.get("scenario_count", 0),
        "mean_miss": None,
        "failed_scenarios": len(record.get("failed_scenarios", [])),
    }


def _empty_result(config: Problem2Config, source_summary: dict[str, Any], elapsed: float) -> dict[str, Any]:
    return {
        "case_id": config.case_id,
        "input_summary": _input_summary(config),
        "possible_source_set": source_summary,
        "best_single_point": {"position": None, "recommendation_status": "inconsistent_first_observation", "message": "首次观测没有可采样的精确源集合。"},
        "candidate_region": {
            "region_semantics": "adaptive_cell_union_discrete_score_approximation",
            "recommendation_status": "empty",
            "J_best_passed_search_sample": None,
            "J_best_definition": "minimum J among passed coarse/local search samples, rescored with the same region score sources",
            "focus_epsilon_J": config.focus_epsilon_j,
            "score_threshold_J": None,
            "base_cell_size_m": config.region_base_cell_m,
            "min_cell_size_m": config.region_min_cell_m,
            "cells": [],
            "atomic_cell_count": 0,
            "components": 0,
            "holes": 0,
            "has_holes": False,
            "boundary_segments": [],
            "evaluated_points": [],
            "evaluated_point_count": 0,
            "receive_verification": {"status": "not_available", "continuous_over_detector_cells": False, "continuous_over_conservative_source_outer": False},
            "score_verification": {"status": "not_available", "continuous_over_detector_cells": False, "points_per_cell": 9},
            "verified_points": [],
        },
        "heuristic_reference": {"note": "启发式参考不可用：首次精确源集合为空。", "thales_s2": None, "thales_s2_pair": [], "scores": [], "score": None},
        "observation_model": "conservative_lower_bound_rho=max(1000,d1); no probability prior",
        "sample_worst_rmin": _metric_block(None, "sample"),
        "validation_worst_rmin": _metric_block(None, "validation"),
        "best_point_receive_certificate": {"status": "unknown", "scope": "single diagnostic point", "cert_margin_m": None, "certification_budget_used": 0, "evidence": "首次可行集合为空。", "outer_label": "conservative_outer"},
        "candidate_region_receive_verification": {"status": "not_available", "continuous_over_detector_cells": False, "continuous_over_conservative_source_outer": False},
        "candidate_region_score_verification": {"status": "not_available", "continuous_over_detector_cells": False, "points_per_cell": 9},
        "travel_distance_m": None,
        "travel_time_s": None,
        "solver_wall_time_s": elapsed,
        "geometry_intersections": 0,
        "mean_miss": None,
        "branch_statistics": {"near": 0, "direction": 0, "no_signal": 0},
        "failure_scenarios": [],
        "diagnostics": {"candidate_count": 0, "safe_sample_candidate_count": 0, "scored_candidate_count": 0, "certified_candidate_count": 0, "unknown_candidate_count": 0, "violated_candidate_count": 0, "best_point_continuous_receive_certified": False, "notes": ["输出 inconsistent_first_observation；没有把中心射线无交误读为唯一判据。"]},
        "candidate_score_table": [],
    }


def _plot_point_groups(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """按搜索身份拆分绘图点，避免把非近优搜索点误画成区域评分点。"""

    region = result["candidate_region"]
    evaluated = [
        row for row in region.get("evaluated_points", [])
        if row.get("j_lambda") is not None and math.isfinite(float(row["j_lambda"]))
    ]
    passed_search_positions = {
        (round(float(row["x"]), 8), round(float(row["y"]), 8))
        for row in result.get("candidate_score_table", [])
        if row.get("cert_status") == "passed"
    }
    best_j = region.get("J_best_passed_search_sample")
    seed_epsilon = float(region.get("search_seed_epsilon_J", 0.0))
    seed_threshold = None if best_j is None else float(best_j) + seed_epsilon
    groups: dict[str, list[dict[str, Any]]] = {
        "region_stencil": [], "near_optimal_seed": [], "non_near_search": [],
    }
    for row in evaluated:
        key = (round(float(row["x"]), 8), round(float(row["y"]), 8))
        if key not in passed_search_positions:
            groups["region_stencil"].append(row)
        elif seed_threshold is not None and float(row["j_lambda"]) <= seed_threshold + 1e-12:
            groups["near_optimal_seed"].append(row)
        else:
            groups["non_near_search"].append(row)
    return groups


def _nice_tick_step(span: float, target_intervals: int = 7) -> float:
    """选择 1/2/2.5/5×10^k 的整齐刻度，并让同一面板两轴等间距。"""

    raw = max(float(span) / max(1, target_intervals), 1e-9)
    power = 10.0 ** math.floor(math.log10(raw))
    fraction = raw / power
    nice = next((value for value in (1.0, 2.0, 2.5, 5.0, 10.0) if fraction <= value), 10.0)
    return nice * power


def _plot_case(case: dict[str, Any], result: dict[str, Any], output_dir: Path) -> None:
    """绘制候选源集合、候选点、泰勒斯启发式参考与推荐区域，不把图当作连续认证。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Polygon, Rectangle
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    palette = {
        "target": "#1F3B57",
        "s1": "#D62728",
        "receive": "#9AA5B1",
        "outer_face": "#4C78A8",
        "outer_edge": "#123B63",
        "source": "#2F6B9A",
        "heat_cmap": "viridis",
        "non_near_face": "#E6E8EB",
        "non_near_edge": "#5A6672",
        "near_seed": "#F28E2B",
        "near_seed_edge": "#7A3E00",
        "failed": "#B71C1C",
        "outside": "#8A8F98",
        "cell_face": (0.78, 0.10, 0.10, 0.10),
        "boundary": "#C62828",
        "thales": "#D81B60",
        "info_face": "#F4F8FD",
        "info_edge": "#3E6B9A",
        "grid": "#B9C2CC",
    }

    config = parse_config(case)
    outer = np.asarray(result["possible_source_set"]["convex_outer_vertices"], dtype=float)
    sources = np.asarray(result["possible_source_set"].get("sample_points", []), dtype=float)
    region = result["candidate_region"]
    evaluated_rows = region.get("evaluated_points", [])
    point_groups = _plot_point_groups(result)
    stencil_rows = point_groups["region_stencil"]
    near_seed_rows = point_groups["near_optimal_seed"]
    non_near_rows = point_groups["non_near_search"]
    receive_failed_rows = [row for row in evaluated_rows if row.get("status") == "receive_failed"]
    outside_rows = [row for row in evaluated_rows if row.get("status") == "outside_budget"]
    # 左右面板使用完全相同的网格列；颜色条单独占一列，避免它挤压右图。
    fig = plt.figure(figsize=(15.0, 7.4), dpi=220)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.045],
        left=0.055,
        right=0.95,
        bottom=0.24,
        top=0.87,
        wspace=0.18,
    )
    ax = fig.add_subplot(grid[0, 0])
    ax_zoom = fig.add_subplot(grid[0, 1])
    colorbar_ax = fig.add_subplot(grid[0, 2])
    colorbar_ax.set_visible(False)
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    circle_x = config.target.cx + config.target.radius * np.cos(theta)
    circle_y = config.target.cy + config.target.radius * np.sin(theta)
    for panel in (ax, ax_zoom):
        panel.plot(circle_x, circle_y, color=palette["target"], lw=1.0, ls=(0, (6, 4)), alpha=0.85, label="目标圆域")
        panel.scatter([config.s1[0]], [config.s1[1]], marker="s", s=52, color=palette["s1"], edgecolors="white", linewidths=0.8, label=f"首次检测点 S1=({config.s1[0]:.1f},{config.s1[1]:.1f})米", zorder=6)
        panel.add_patch(Circle(config.s1, config.rho_max, fill=False, ls=":", color=palette["receive"], lw=0.9, label="1500米接收上界"))
        if len(outer):
            panel.add_patch(Polygon(outer, closed=True, facecolor=palette["outer_face"], alpha=0.22, edgecolor=palette["outer_edge"], lw=1.4, label="保守外包 U_bar"))
        if len(sources):
            panel.scatter(sources[:, 0], sources[:, 1], s=10, color=palette["source"], alpha=0.6, label="公式生成源样本 U")
    if stencil_rows:
        score_values = np.asarray([float(row["j_lambda"]) for row in stencil_rows], dtype=float)
        score_min = float(np.min(score_values))
        score_max = float(np.max(score_values))
        if score_max - score_min <= 1e-12:
            score_max = score_min + 1e-12
        candidate_x = np.asarray([float(row["x"]) for row in stencil_rows], dtype=float)
        candidate_y = np.asarray([float(row["y"]) for row in stencil_rows], dtype=float)
        heat = ax.scatter(candidate_x, candidate_y, c=score_values, cmap=palette["heat_cmap"], vmin=score_min, vmax=score_max, s=26, alpha=0.88, edgecolors="none", label="实际区域评分点（J越小越优）", zorder=3)
        ax_zoom.scatter(candidate_x, candidate_y, c=score_values, cmap=palette["heat_cmap"], vmin=score_min, vmax=score_max, s=32, alpha=0.88, edgecolors="none", zorder=3)
        colorbar_ax.set_visible(True)
        colorbar = fig.colorbar(heat, cax=colorbar_ax)
        colorbar.set_label("区域评分 J（越小越优）")
        colorbar.outline.set_edgecolor(palette["grid"])
        threshold = region.get("score_threshold_J")
        if threshold is not None and score_min <= float(threshold) <= score_max:
            colorbar.ax.axhline(float(threshold), color=palette["boundary"], lw=1.5, ls="--")
    if non_near_rows:
        xs = [float(row["x"]) for row in non_near_rows]
        ys = [float(row["y"]) for row in non_near_rows]
        for panel in (ax, ax_zoom):
            panel.scatter(xs, ys, marker="o", s=34, facecolors=palette["non_near_face"], edgecolors=palette["non_near_edge"], linewidths=1.0, label="接收认证通过但非近优的搜索点" if panel is ax else None, zorder=4)
    if near_seed_rows:
        xs = [float(row["x"]) for row in near_seed_rows]
        ys = [float(row["y"]) for row in near_seed_rows]
        for panel in (ax, ax_zoom):
            panel.scatter(xs, ys, marker="*", s=105, color=palette["near_seed"], edgecolors=palette["near_seed_edge"], linewidths=0.7, label="接收认证通过的近优种子" if panel is ax else None, zorder=7)
    if receive_failed_rows:
        failed_x = [float(row["x"]) for row in receive_failed_rows]
        failed_y = [float(row["y"]) for row in receive_failed_rows]
        for panel in (ax, ax_zoom):
            panel.scatter(failed_x, failed_y, marker="x", s=28, color=palette["failed"], linewidths=1.2, label="实际接收不合格检查点" if panel is ax else None, zorder=4)
    if outside_rows:
        outside_x = [float(row["x"]) for row in outside_rows]
        outside_y = [float(row["y"]) for row in outside_rows]
        for panel in (ax, ax_zoom):
            panel.scatter(outside_x, outside_y, marker="x", s=22, color=palette["outside"], linewidths=0.9, label="移动预算外检查点" if panel is ax else None, zorder=3)
    candidate_cells = region.get("cells", [])
    for cell in candidate_cells:
        for panel in (ax, ax_zoom):
            vertices = cell.get("vertices")
            if vertices is not None:
                panel.add_patch(Polygon(np.asarray(vertices, dtype=float), closed=True, facecolor=palette["cell_face"], edgecolor="none", zorder=4))
            else:
                bounds = cell["bounds"]
                panel.add_patch(Rectangle((bounds[0], bounds[2]), bounds[1] - bounds[0], bounds[3] - bounds[2], facecolor=palette["cell_face"], edgecolor="none", zorder=4))
    boundary_segments = np.asarray(region.get("boundary_segments", []), dtype=float)
    if len(boundary_segments):
        for panel in (ax, ax_zoom):
            panel.add_collection(LineCollection(boundary_segments, colors=palette["boundary"], linewidths=3.2, linestyles="--", zorder=6))
        ax.plot([], [], color=palette["boundary"], lw=3.2, ls="--", label="优先推荐区域（离散近似）")
    ax.set_xlim(config.target.cx - config.target.radius * 1.08, config.target.cx + config.target.radius * 1.08)
    ax.set_ylim(config.target.cy - config.target.radius * 1.08, config.target.cy + config.target.radius * 1.08)
    if len(boundary_segments):
        zoom_points = boundary_segments.reshape((-1, 2))
        low = zoom_points.min(axis=0)
        high = zoom_points.max(axis=0)
        center = (low + high) / 2.0
        span = max(4.0 * region.get("min_cell_size_m", 1.0), float(np.max(high - low))) * 1.45
        ax_zoom.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
        ax_zoom.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
    elif len(outer):
        low = outer.min(axis=0)
        high = outer.max(axis=0)
        center = (low + high) / 2.0
        span = max(40.0, float(np.max(high - low))) * 1.35
        ax_zoom.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
        ax_zoom.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
    else:
        ax_zoom.set_xlim(config.s1[0] - 100, config.s1[0] + 100)
        ax_zoom.set_ylim(config.s1[1] - 100, config.s1[1] + 100)
    heuristic = result.get("heuristic_reference") or {}
    thales_points = heuristic.get("thales_s2_pair") or ([heuristic["thales_s2"]] if heuristic.get("thales_s2") else [])
    for point_index, thales_point in enumerate(thales_points):
        thales_xy = np.asarray([float(thales_point["x"]), float(thales_point["y"])], dtype=float)
        for panel in (ax, ax_zoom):
            xlim = panel.get_xlim()
            ylim = panel.get_ylim()
            point_visible = xlim[0] <= thales_xy[0] <= xlim[1] and ylim[0] <= thales_xy[1] <= ylim[1]
            if not point_visible:
                continue
            panel.scatter(
                [thales_xy[0]], [thales_xy[1]], marker="D", s=58,
                facecolors="none", edgecolors=palette["thales"], linewidths=1.6, zorder=9,
                label="泰勒斯启发式参考点对（不参与推荐）" if panel is ax and point_index == 0 else None,
            )
    best_j_text = "不可用" if region.get("J_best_passed_search_sample") is None else f"{float(region['J_best_passed_search_sample']):.5f}"
    threshold_text = "不可用" if region.get("score_threshold_J") is None else f"{float(region['score_threshold_J']):.5f}"
    region_source_count = int(region.get("region_score_source_count", 0))
    search_source_count = int(result.get("sample_worst_rmin", {}).get("source_count", 0))
    search_point_count = len(result.get("candidate_score_table", []))
    plotted_passed_search_count = len(near_seed_rows) + len(non_near_rows)
    symmetry_status = result.get("diagnostics", {}).get("symmetry_audit", {}).get("status")
    symmetry_text = "\n镜像点对核验=通过" if symmetry_status == "passed" else ""
    heuristic_score = (result.get("heuristic_reference") or {}).get("score") or {}
    heuristic_j = heuristic_score.get("j_lambda")
    thales_text = "" if heuristic_j is None else f"\n泰勒斯参考点 J（仅诊断）={float(heuristic_j):.5f}"
    ax.text(0.02, 0.98, f"S1=({config.s1[0]:.1f},{config.s1[1]:.1f}) 米\n源样本：图示{len(sources)}；搜索评分{search_source_count}；区域评分{region_source_count}\nS2实际搜索={search_point_count}；图示接收认证通过={plotted_passed_search_count}\n其中近优种子={len(near_seed_rows)}，非近优={len(non_near_rows)}\nJ_best（通过检查的搜索样本）={best_j_text}\n红框阈值 J≤{threshold_text}\n推荐单元={len(candidate_cells)}；最小分辨率={region.get('min_cell_size_m', config.region_min_cell_m):.1f} 米\n单元模板实际评估点={region.get('evaluated_point_count', 0)}{symmetry_text}{thales_text}", transform=ax.transAxes, va="top", bbox={"boxstyle": "round,pad=0.35", "fc": palette["info_face"], "ec": palette["info_edge"], "alpha": 0.94})
    receive_status = region.get("receive_verification", {}).get("status", "not_available")
    score_status = region.get("score_verification", {}).get("status", "not_available")
    receive_text = "充分条件通过" if receive_status == "passed" else "不可用"
    score_text = "9点模板通过" if score_status == "sampled_passed" else "不可用"
    ax_zoom.text(0.02, 0.98, f"整单元接收认证={receive_text}\n单元评分验证={score_text}\n评分检查=中心+角点+边中点\n连续评分证明=否", transform=ax_zoom.transAxes, va="top", bbox={"boxstyle": "round,pad=0.35", "fc": palette["info_face"], "ec": palette["info_edge"], "alpha": 0.94})
    for panel in (ax, ax_zoom):
        panel.set_aspect("equal", adjustable="box")
        panel.set_anchor("C")
        x_span = abs(float(panel.get_xlim()[1] - panel.get_xlim()[0]))
        y_span = abs(float(panel.get_ylim()[1] - panel.get_ylim()[0]))
        tick_step = _nice_tick_step(max(x_span, y_span))
        panel.xaxis.set_major_locator(MultipleLocator(tick_step))
        panel.yaxis.set_major_locator(MultipleLocator(tick_step))
        panel.set_xlabel("x / 米")
        panel.set_ylabel("y / 米")
        panel.grid(color=palette["grid"], alpha=0.35, lw=0.6)
    case_name = str(case.get("display_name_zh", result["case_id"]))
    for forbidden_label in ("良好", "较差", "临界"):
        case_name = case_name.replace(forbidden_label, "")
    case_name = " ".join(case_name.split()) or result["case_id"]
    ax.set_title(f"问题2：{case_name}", color=palette["target"])
    ax_zoom.set_title("候选区域局部图", color=palette["target"])
    fig.suptitle("问题2第二检测点优先区域（离散近似）", fontsize=14, fontweight="bold", color=palette["target"])
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=8.2, framealpha=0.95, edgecolor="#C7D2DE", columnspacing=1.3, handletextpad=0.7)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"problem2_{result['case_id']}"
    fig.savefig(stem.with_suffix(".jpg"), dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95, "optimize": True})
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_four_case_summary(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    output_dir: Path,
    requested_case_ids: list[str] | None = None,
) -> None:
    """把四个代表案例画在同一张 2×2 图中，并共用一份图例。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Polygon, Rectangle
    from matplotlib.ticker import MultipleLocator

    palette = {
        "target": "#1F3B57", "s1": "#D62728", "receive": "#9AA5B1",
        "outer_face": "#4C78A8", "outer_edge": "#123B63", "source": "#2F6B9A",
        "stencil": "#2A9D8F", "non_near_face": "#F3F4F6", "non_near_edge": "#5A6672",
        "near_seed": "#F28E2B", "near_seed_edge": "#7A3E00", "failed": "#B71C1C",
        "cell_face": (0.78, 0.10, 0.10, 0.10), "boundary": "#C62828",
        "thales": "#D81B60", "grid": "#B9C2CC",
    }
    by_id = {str(result["case_id"]): (case, result) for case, result in zip(cases, results)}
    if requested_case_ids:
        if len(requested_case_ids) != 4:
            raise ValueError("四图汇总必须恰好指定 4 个案例编号")
        missing = [case_id for case_id in requested_case_ids if case_id not in by_id]
        if missing:
            raise ValueError(f"四图汇总找不到案例：{', '.join(missing)}")
        selected_ids = requested_case_ids
    else:
        preferred = ["p2_wrap_around", "p2_external_tangent", "p2_near_branch", "p2_multicomponent_probe"]
        selected_ids = [case_id for case_id in preferred if case_id in by_id]
        fallback = [
            case_id for case_id, (_, result) in by_id.items()
            if case_id not in selected_ids and not result.get("possible_source_set", {}).get("is_empty", False)
        ]
        selected_ids.extend(fallback[: 4 - len(selected_ids)])
    if len(selected_ids) < 4:
        return

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 12.2), dpi=220)
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    for panel, case_id in zip(axes.flat, selected_ids):
        case, result = by_id[case_id]
        config = parse_config(case)
        region = result["candidate_region"]
        outer = np.asarray(result["possible_source_set"].get("convex_outer_vertices", []), dtype=float)
        sources = np.asarray(result["possible_source_set"].get("sample_points", []), dtype=float)
        groups = _plot_point_groups(result)
        panel.plot(
            config.target.cx + config.target.radius * np.cos(theta),
            config.target.cy + config.target.radius * np.sin(theta),
            color=palette["target"], lw=1.1, ls=(0, (6, 4)), label="目标圆域",
        )
        panel.scatter([config.s1[0]], [config.s1[1]], marker="s", s=48, color=palette["s1"],
                      edgecolors="white", linewidths=0.7, label="首次检测点 S1", zorder=7)
        panel.add_patch(Circle(config.s1, config.rho_max, fill=False, ls=":", color=palette["receive"],
                               lw=0.9, label="1500米接收上界"))
        if len(outer):
            panel.add_patch(Polygon(outer, closed=True, facecolor=palette["outer_face"], alpha=0.20,
                                    edgecolor=palette["outer_edge"], lw=1.2, label="保守外包 U_bar"))
        if len(sources):
            panel.scatter(sources[:, 0], sources[:, 1], s=10, color=palette["source"], alpha=0.58,
                          label="公式生成源样本 U")
        stencil = groups["region_stencil"]
        if stencil:
            panel.scatter([float(row["x"]) for row in stencil], [float(row["y"]) for row in stencil],
                          s=24, color=palette["stencil"], alpha=0.78, edgecolors="none",
                          label="实际区域评分点")
        non_near = groups["non_near_search"]
        if non_near:
            panel.scatter([float(row["x"]) for row in non_near], [float(row["y"]) for row in non_near],
                          s=31, facecolors=palette["non_near_face"], edgecolors=palette["non_near_edge"],
                          linewidths=1.0, label="认证通过但非近优搜索点", zorder=5)
        near = groups["near_optimal_seed"]
        if near:
            panel.scatter([float(row["x"]) for row in near], [float(row["y"]) for row in near], marker="*",
                          s=92, color=palette["near_seed"], edgecolors=palette["near_seed_edge"], linewidths=0.7,
                          label="认证通过的近优种子", zorder=8)
        failed = [row for row in region.get("evaluated_points", []) if row.get("status") == "receive_failed"]
        if failed:
            panel.scatter([float(row["x"]) for row in failed], [float(row["y"]) for row in failed], marker="x",
                          s=27, color=palette["failed"], linewidths=1.1, label="接收不合格检查点", zorder=6)
        for cell in region.get("cells", []):
            vertices = cell.get("vertices")
            if vertices is not None:
                panel.add_patch(Polygon(np.asarray(vertices, dtype=float), closed=True,
                                        facecolor=palette["cell_face"], edgecolor="none", zorder=4))
            else:
                bounds = cell["bounds"]
                panel.add_patch(Rectangle((bounds[0], bounds[2]), bounds[1] - bounds[0], bounds[3] - bounds[2],
                                          facecolor=palette["cell_face"], edgecolor="none", zorder=4))
        segments = np.asarray(region.get("boundary_segments", []), dtype=float)
        if len(segments):
            panel.add_collection(LineCollection(segments, colors=palette["boundary"], linewidths=2.8,
                                                linestyles="--", label="优先推荐区域（离散近似）", zorder=7))
        heuristic = result.get("heuristic_reference") or {}
        thales_points = heuristic.get("thales_s2_pair") or ([heuristic["thales_s2"]] if heuristic.get("thales_s2") else [])
        if thales_points:
            panel.scatter([float(point["x"]) for point in thales_points],
                          [float(point["y"]) for point in thales_points], marker="D", s=58,
                          facecolors="none", edgecolors=palette["thales"], linewidths=1.6,
                          label="泰勒斯启发式参考点对", zorder=9)
        panel.set_xlim(config.target.cx - config.target.radius * 1.08, config.target.cx + config.target.radius * 1.08)
        panel.set_ylim(config.target.cy - config.target.radius * 1.08, config.target.cy + config.target.radius * 1.08)
        panel.set_aspect("equal", adjustable="box")
        panel.set_anchor("C")
        tick_step = _nice_tick_step(2.16 * config.target.radius, target_intervals=6)
        panel.xaxis.set_major_locator(MultipleLocator(tick_step))
        panel.yaxis.set_major_locator(MultipleLocator(tick_step))
        panel.set_xlabel("x / 米")
        panel.set_ylabel("y / 米")
        panel.grid(color=palette["grid"], alpha=0.32, lw=0.6)
        case_name = str(case.get("display_name_zh", case_id)).replace("问题2", "").strip()
        panel.set_title(case_name, color=palette["target"], fontsize=12)

    handles: list[Any] = []
    labels: list[str] = []
    for panel in axes.flat:
        panel_handles, panel_labels = panel.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.suptitle("问题2第二检测点优先区域：四案例对照", fontsize=16, fontweight="bold", color=palette["target"])
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=5, fontsize=9,
               framealpha=0.96, edgecolor="#C7D2DE", columnspacing=1.25, handletextpad=0.65)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.91, wspace=0.16, hspace=0.23)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "problem2_four_case_summary"
    fig.savefig(stem.with_suffix(".jpg"), dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95, "optimize": True})
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    best = result["best_single_point"]
    region = result["candidate_region"]
    return {
        "case_id": result["case_id"],
        "region_status": region.get("recommendation_status"),
        "J_best_passed_search_sample": region.get("J_best_passed_search_sample"),
        "J_best_definition": region.get("J_best_definition"),
        "focus_epsilon_J": region.get("focus_epsilon_J"),
        "score_threshold_J": region.get("score_threshold_J"),
        "region_receive_status": region.get("receive_verification", {}).get("status"),
        "region_receive_continuous_over_cells": region.get("receive_verification", {}).get("continuous_over_detector_cells", False),
        "region_receive_source_scope": region.get("receive_verification", {}).get("scope"),
        "region_score_status": region.get("score_verification", {}).get("status"),
        "region_score_continuous_certified": region.get("score_verification", {}).get("continuous_over_detector_cells", False),
        "score_stencil_points_per_cell": region.get("score_verification", {}).get("points_per_cell"),
        "candidate_components": region.get("components"),
        "candidate_holes": region.get("holes"),
        "candidate_cells": len(region.get("cells", [])),
        "atomic_cell_count": region.get("atomic_cell_count"),
        "base_cell_size_m": region.get("base_cell_size_m"),
        "min_cell_size_m": region.get("min_cell_size_m"),
        "max_subdivision_depth": region.get("max_subdivision_depth"),
        "region_score_source_count": region.get("region_score_source_count"),
        "evaluated_point_count": region.get("evaluated_point_count"),
        "receive_failed_point_count": region.get("receive_failed_point_count"),
        "best_point_status_diagnostic": best.get("recommendation_status"),
        "solver_wall_time_s": result.get("solver_wall_time_s"),
        "flags": ";".join(result.get("diagnostics", {}).get("notes", [])),
    }


def _json_safe(value: Any) -> Any:
    """把 NumPy 标量和非有限浮点数转成可移植的标准 JSON 值。"""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run(
    input_path: Path,
    out_dir: Path,
    make_plots: bool = True,
    summary_case_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    cases = payload if isinstance(payload, list) else [payload]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [solve_case(case, make_plots=make_plots, output_dir=out_dir) for case in cases]
    if make_plots:
        _plot_four_case_summary(cases, results, out_dir, summary_case_ids)
    (out_dir / "problem2_report.json").write_text(
        json.dumps(_json_safe(results), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    rows = [_csv_row(result) for result in results]
    with (out_dir / "problem2_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["case_id"])
        writer.writeheader()
        writer.writerows(rows)
    return results


def main() -> int:
    parser = ChineseArgumentParser(description="离线求解 B 题问题 2", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示此帮助信息并退出")
    parser.add_argument("--input", required=True, type=Path, help="问题2案例 JSON")
    parser.add_argument("--out", required=True, type=Path, help="输出目录")
    parser.add_argument("--no-plots", action="store_true", help="跳过图像生成")
    parser.add_argument("--summary-case-ids", nargs=4, metavar="CASE_ID", help="四图汇总使用的 4 个案例编号")
    args = parser.parse_args()
    results = run(args.input, args.out, make_plots=not args.no_plots, summary_case_ids=args.summary_case_ids)
    print(json.dumps({"cases": len(results), "output_dir": str(args.out.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
