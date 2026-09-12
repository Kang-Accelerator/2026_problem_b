"""问题3自动求解器的官方模拟器运行入口。

模拟器负责生成隐藏案例和返回测量结果；本程序负责完整决策。只有同时提供
``--robot-id`` 和 ``--confirm-ready`` 才会调用 /enter。导入本模块或执行
``--preflight`` 均不会联网。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

try:
    from .action_logger import ActionLogger
    from .code.problem3_solver import (
        Config,
        Problem3Solver,
        SimulatorLike,
        coverage_report,
        guaranteed_second_measurement,
        load_config,
        write_result,
    )
    from .simulator_client import SimulatorClient
except ImportError:  # 支持 python competition_b/problem3_auto_run.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from competition_b.action_logger import ActionLogger
    from competition_b.code.problem3_solver import (
        Config,
        Problem3Solver,
        SimulatorLike,
        coverage_report,
        guaranteed_second_measurement,
        load_config,
        write_result,
    )
    from competition_b.simulator_client import SimulatorClient


def offline_preflight(cfg: Config) -> dict[str, Any]:
    """返回实时运行前必须通过的、完全不联网的几何检查。"""

    coverage = coverage_report(cfg)
    second = guaranteed_second_measurement(cfg)
    return {
        "passed": bool(coverage["certified"] and second["passed"]),
        "config": asdict(cfg),
        "coverage": coverage,
        "second_measurement_guarantee": second,
    }


def run_automatic_session(
    cfg: Config,
    simulator: SimulatorLike,
    logger: ActionLogger | None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """执行一次完整自动闭环；供正式入口和离线合成测试共用。"""

    result = Problem3Solver(cfg, simulator, logger).run()
    result["run_metadata"] = dict(metadata or {})
    result["control_mode"] = "automatic"
    result["operator_actions_required_after_enter"] = 0
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="问题3自动求解器官方模拟器入口")
    parser.add_argument(
        "--robot-id",
        default=os.getenv("CUMCM_ROBOT_ID"),
        help="当前模拟器登录的参赛队号；也可用 CUMCM_ROBOT_ID",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("CUMCM_BASE_URL", "http://127.0.0.1:2026"),
        help="官方模拟器本机接口地址",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("competition_b/data/problem3_config.json"),
        help="问题3参数文件",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("competition_b/results")
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("competition_b/logs")
    )
    parser.add_argument(
        "--case-id",
        default="",
        help="模拟器界面显示的案例编码；接口不返回，建议人工复制",
    )
    parser.add_argument(
        "--test-mode",
        choices=("practice", "formal", "unspecified"),
        default="unspecified",
        help="只写入本地报告；实际模式由模拟器界面决定",
    )
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--max-network-retries", type=int, default=3)
    parser.add_argument(
        "--confirm-ready",
        action="store_true",
        help="确认问题3测试界面倒计时结束且机器狗接口已就绪；提供后才允许 /enter",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="只做离线检查并退出，绝不连接模拟器",
    )
    return parser


def _write_preflight(report: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "problem3_auto_preflight.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"配置读取失败：{exc}", file=sys.stderr)
        return 2

    preflight = offline_preflight(cfg)
    preflight_path = _write_preflight(preflight, args.output_dir)
    print(
        "离线检查：{}；7点最坏距离={:.6f}m，余量={:.6f}m，第二点保证={}。".format(
            "通过" if preflight["passed"] else "失败",
            preflight["coverage"]["exact_worst_nearest_distance_m"],
            preflight["coverage"]["margin_m"],
            preflight["second_measurement_guarantee"]["passed"],
        )
    )
    print(f"离线检查文件：{preflight_path.resolve()}")
    if not preflight["passed"]:
        print("离线证书未通过，已禁止连接模拟器。", file=sys.stderr)
        return 2
    if args.preflight:
        print("仅执行 preflight，未连接模拟器。")
        return 0
    if not args.robot_id:
        print("缺少 --robot-id；已禁止连接模拟器。", file=sys.stderr)
        return 2
    if not args.confirm_ready:
        print(
            "缺少 --confirm-ready；请先在官方模拟器中启动问题3测试，"
            "等待5秒倒计时结束并确认接口已就绪。当前未调用 /enter。",
            file=sys.stderr,
        )
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = ActionLogger(args.log_dir / f"problem3_auto_actions_{stamp}.jsonl")
    client = SimulatorClient(
        base_url=args.base_url,
        robot_id=str(args.robot_id),
        logger=logger,
        timeout_s=float(args.timeout_s),
        max_network_retries=int(args.max_network_retries),
    )
    metadata = {
        "case_id": str(args.case_id),
        "test_mode_label": args.test_mode,
        "base_url": args.base_url,
        "config_path": str(args.config.resolve()),
        "started_local_time": datetime.now().astimezone().isoformat(),
        "note": "test_mode_label用于本地关联；真正模式由官方模拟器界面决定。",
    }

    print("开始自动闭环：从此处起无需输入任何 measure/clear 命令。")
    result = run_automatic_session(cfg, client, logger, metadata)
    json_path, csv_path = write_result(result, args.output_dir)
    print(
        json.dumps(
            {
                "run_status": result["run_status"],
                "completion_certificate": result["completion_certificate"],
                "completion_reason": result["completion_reason"],
                "cleared_count": result["cleared_count"],
                "total_time_s": result["total_time_s"],
                "report": str(json_path.resolve()),
                "summary": str(csv_path.resolve()),
                "action_log": str(logger.path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    if result["completion_certificate"]:
        print("自动算法已形成完成证书。请在官方模拟器中导出本局加密行为日志。")
        return 0
    print("本局未形成完成证书；请保留全部日志，不要把本局写成完整清除。", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
