"""问题 1 的命令行求解和绘图入口。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

try:
    from .problem1_geometry import solve_case
except ImportError:  # 支持执行 python competition_b/code/problem1_solve.py。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from problem1_geometry import solve_case


class ChineseArgumentParser(argparse.ArgumentParser):
    """将 argparse 自动生成的帮助标题显示为中文。"""

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法：")

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法：").replace("options:", "选项：").replace("optional arguments:", "可选参数：")


def _plot_case(case: dict, result: dict, png_path: Path, pdf_path: Path, jpg_path: Path | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    import numpy as np

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    radius = float(result["input_summary"]["R"])
    delta_deg = float(result["input_summary"]["delta_deg"])
    points = case.get("points", [])
    main = result["main"]
    vertices = np.asarray(main["vertices"], dtype=float) if main["vertices"] else np.empty((0, 2))
    pair = np.asarray(main["diameter_pair"], dtype=float) if main["diameter_pair"] else np.empty((0, 2))
    center = np.asarray(main["mec_center"], dtype=float) if main["mec_center"] is not None else None
    mec_radius = float(main["mec_radius_m"] or 0.0)
    sensor_xy = np.asarray([[float(item["x"]), float(item["y"])] for item in points], dtype=float) if points else np.empty((0, 2))

    coverage_text = "可覆盖" if main["coverage_possible"] else "不可覆盖"
    case_name = str(case.get("display_name_zh", result["case_id"]))
    # 中文案例名可能来自历史数据，绘图标题不再承载任何分档评价含义。
    for forbidden_label in ("良好", "较差", "临界"):
        case_name = case_name.replace(forbidden_label, "")
    case_name = " ".join(case_name.split()) or result["case_id"]
    diameter_value = main.get("diameter_m")
    diameter_text = "不可用（空区域）" if diameter_value is None else f"{float(diameter_value):.2f} 米"
    mec_text = "不可用（空区域）" if main.get("mec_radius_m") is None else f"{mec_radius:.2f} 米"
    metric_text = f"D={diameter_text}\nR_min={mec_text}\n覆盖：{coverage_text}"
    theta = np.linspace(0, 2 * np.pi, 720)

    def collect_feature_bounds() -> np.ndarray:
        chunks = []
        for values in (vertices, pair):
            if values.size:
                chunks.append(values)
        if center is not None:
            chunks.append(np.array([center - mec_radius, center + mec_radius], dtype=float))
        if not chunks:
            chunks.append(np.array([[-radius, -radius], [radius, radius]], dtype=float))
        return np.vstack(chunks)

    def limits(values: np.ndarray, padding: float = 0.08, minimum_span: float = 80.0) -> tuple[float, float, float, float]:
        low = values.min(axis=0)
        high = values.max(axis=0)
        span = np.maximum(high - low, minimum_span)
        margin = np.maximum(span * padding, 10.0)
        mid = (low + high) / 2.0
        half = span / 2.0 + margin
        return (float(mid[0] - half[0]), float(mid[0] + half[0]), float(mid[1] - half[1]), float(mid[1] + half[1]))

    feature_bounds = collect_feature_bounds()
    if sensor_xy.size:
        sensor_scale = float(np.max(np.linalg.norm(sensor_xy, axis=1)))
    else:
        sensor_scale = 0.0
    main_scale = max(radius, sensor_scale, 1.0)
    main_bounds = (-1.12 * main_scale, 1.12 * main_scale, -1.12 * main_scale, 1.12 * main_scale)
    zoom_bounds = limits(feature_bounds, padding=0.28, minimum_span=max(40.0, 2.0 * mec_radius))

    fig, (ax_main, ax_zoom) = plt.subplots(
        1, 2, figsize=(15.5, 7.8), dpi=300,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    def ray_distance_to_boundary(x: float, y: float, ux: float, uy: float, bounds) -> float:
        candidates = []
        if ux > 1e-12:
            candidates.append((bounds[1] - x) / ux)
        elif ux < -1e-12:
            candidates.append((bounds[0] - x) / ux)
        if uy > 1e-12:
            candidates.append((bounds[3] - y) / uy)
        elif uy < -1e-12:
            candidates.append((bounds[2] - y) / uy)
        positive = [value for value in candidates if value > 0]
        return min(positive) if positive else max(radius, 1.0)

    def draw_panel(ax, bounds, annotate_points: bool, compact: bool = False) -> None:
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), "k--", lw=0.9, label="精确圆盘边界")
        if vertices.size:
            ax.add_patch(Polygon(vertices, closed=True, facecolor="#4C78A8", alpha=0.42,
                                 edgecolor="#123B63", lw=1.8, label="可行区域", zorder=2))
            ax.scatter(vertices[:, 0], vertices[:, 1], s=12 if compact else 8,
                       color="#123B63", edgecolors="white", linewidths=0.25, zorder=6, label="区域顶点")
        line_scale = max(radius, sensor_scale, 1.0)
        ray_length = max(4.0 * line_scale, 100.0)
        for index, item in enumerate(points, start=1):
            x, y = float(item["x"]), float(item["y"])
            bearing = float(item.get("bearing_deg", item.get("svd_deg")))
            for side, angle in enumerate((bearing - delta_deg, bearing + delta_deg)):
                radians = np.deg2rad(angle)
                ux, uy = float(np.cos(radians)), float(np.sin(radians))
                length = ray_length if compact else ray_distance_to_boundary(x, y, ux, uy, bounds)
                ax.plot([x, x + length * ux], [y, y + length * uy],
                        color="#C23B3B", ls=(0, (4, 2)), lw=1.15, alpha=0.82, zorder=4,
                        label=f"示向误差边界 ±{delta_deg:g}°" if index == 1 and side == 0 else None)
            ax.scatter([x], [y], s=34 if not compact else 24, color="#D62728",
                       edgecolors="white", linewidths=0.7, zorder=5,
                       label="检测点" if index == 1 else None)
            if annotate_points:
                if len(points) >= 4:
                    offsets = [(-36, 24), (28, -28), (36, 24), (-28, -28), (22, 0), (-22, 0), (22, 18), (-22, 18)]
                    dx, dy = offsets[(index - 1) % len(offsets)]
                    label_text = f"S{index}"
                else:
                    offsets = [(12, 12), (12, -18), (-12, 12), (-12, -18), (22, 0), (-22, 0)]
                    dx, dy = offsets[(index - 1) % len(offsets)]
                    label_text = f"S{index}=({x:.2f}, {y:.2f}) 米"
                ax.annotate(
                    label_text,
                    xy=(x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=7.5 if not compact else 6.5,
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top",
                    color="#7A1F1F", zorder=7,
                    bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#D62728", "alpha": 0.88},
                    arrowprops={"arrowstyle": "-", "color": "#D62728", "lw": 0.55, "shrinkA": 2, "shrinkB": 2},
                )
        if pair.size:
            ax.plot(pair[:, 0], pair[:, 1], color="#E87500", lw=3.0, ls="-",
                    label=f"区域直径 D={main['diameter_m']:.2f} 米", zorder=4)
            ax.scatter(pair[:, 0], pair[:, 1], facecolors="white", edgecolors="#E87500",
                       marker="D", linewidths=1.6, s=58 if compact else 48,
                       zorder=9, label="直径端点")
        if center is not None:
            ax.add_patch(Circle(center, mec_radius, fill=False, color="#168A52", lw=2.6,
                                 label=f"最小包围圆 R_min={mec_radius:.2f} 米", zorder=4))
            ax.scatter([center[0]], [center[1]], facecolors="white", edgecolors="#168A52",
                       marker="X", linewidths=1.4, s=72 if compact else 58,
                       zorder=10, label="MEC 圆心")
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x / 米")
        ax.set_ylabel("y / 米")
        ax.grid(alpha=0.22)
        ax.text(0.02, 0.98, metric_text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9 if not compact else 8, color="#17365D",
                bbox={"boxstyle": "round,pad=0.35", "fc": "#F5F9FF", "ec": "#4C78A8", "alpha": 0.94},
                zorder=10)
    draw_panel(ax_main, main_bounds, annotate_points=True)
    ax_main.set_title(f"主图：{case_name}", fontsize=11)
    draw_panel(ax_zoom, zoom_bounds, annotate_points=False, compact=True)
    ax_zoom.set_title("定位区域局部放大图", fontsize=11)
    fig.suptitle("问题1定位结果（主图与局部放大）", fontsize=14, fontweight="bold")
    if len(points) >= 4:
        coordinate_lines = [f"S{i}=({float(item['x']):.2f}, {float(item['y']):.2f}) 米" for i, item in enumerate(points, start=1)]
        split = (len(coordinate_lines) + 1) // 2
        fig.text(0.27, 0.115, "\n".join(coordinate_lines[:split]), ha="center", va="center", fontsize=8, color="#7A1F1F")
        fig.text(0.73, 0.115, "\n".join(coordinate_lines[split:]), ha="center", va="center", fontsize=8, color="#7A1F1F")
    handles, labels = ax_main.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=5, fontsize=8, framealpha=0.94)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.22 if len(points) >= 4 else 0.16, top=0.88, wspace=0.14)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    if jpg_path is None:
        jpg_path = png_path.with_suffix(".jpg")
    fig.savefig(jpg_path, format="jpg", dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95, "optimize": True})
    plt.close(fig)


def _csv_row(result: dict) -> dict:
    main = result["main"]
    pair = main.get("diameter_pair") or [[None, None], [None, None]]
    center = main.get("mec_center") or [None, None]
    return {
        "case_id": result["case_id"],
        "num_points": result["input_summary"]["num_points"],
        "region": "main",
        "is_bounded": main["is_bounded"],
        "num_vertices": main["num_vertices"],
        "area_m2": main["area_m2"],
        "perimeter_m": main["perimeter_m"],
        "diameter_m": main["diameter_m"],
        "diameter_p1_x": pair[0][0],
        "diameter_p1_y": pair[0][1],
        "diameter_p2_x": pair[1][0],
        "diameter_p2_y": pair[1][1],
        "mec_cx": center[0],
        "mec_cy": center[1],
        "mec_radius_m": main["mec_radius_m"],
        "mec_diameter_m": main["mec_diameter_m"],
        "coverage_possible": main["coverage_possible"],
        "coverage_gap_m": main["coverage_gap_m"],
        "eps_cov_m": main["eps_cov_m"],
        "coverage_status": main["coverage_status"],
        "guaranteed_clearable": main["guaranteed_clearable"],
        "ready_to_clear": main["ready_to_clear"],
        "flags": ";".join(main["flags"]),
    }


def run(input_path: Path, out_dir: Path, make_plots: bool = True) -> list[dict]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    cases = payload if isinstance(payload, list) else [payload]
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for case in cases:
        result = solve_case(case)
        results.append(result)
        if make_plots:
            stem = f"problem1_{result['case_id']}"
            _plot_case(case, result, out_dir / f"{stem}.png", out_dir / f"{stem}.pdf", out_dir / f"{stem}.jpg")
    (out_dir / "problem1_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [_csv_row(result) for result in results]
    with (out_dir / "problem1_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["case_id"])
        writer.writeheader()
        writer.writerows(rows)
    return results


def main() -> int:
    parser = ChineseArgumentParser(description="离线求解 2026 年高教社杯 B 题问题 1", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示此帮助信息并退出")
    parser.add_argument("--input", required=True, type=Path, help="JSON 案例文件或案例数组")
    parser.add_argument("--out", required=True, type=Path, help="输出目录")
    parser.add_argument("--no-plots", action="store_true", help="跳过 PNG/PDF 图像生成")
    args = parser.parse_args()
    results = run(args.input, args.out, make_plots=not args.no_plots)
    print(json.dumps({"cases": len(results), "output_dir": str(args.out.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
