"""不调用 /enter 且不消耗测试机会的离线检查。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    from . import config
except ImportError:  # 支持直接执行 python competition_b/preflight.py。
    import config


def main() -> int:
    failures: list[str] = []

    if not config.ROBOT_ID or "<" in config.ROBOT_ID or "你的" in config.ROBOT_ID:
        failures.append("CUMCM_ROBOT_ID 为空或仍是占位符")
    if not config.BASE_URL.startswith(("http://", "https://")):
        failures.append("CUMCM_BASE_URL 必须以 http:// 或 https:// 开头")

    for package in ("numpy", "pandas", "scipy"):
        if importlib.util.find_spec(package) is None:
            failures.append(f"缺少依赖包：{package}")

    for directory in (config.LOG_DIR, config.RESULT_DIR):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write_probe"
            probe.write_text("离线检查", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            failures.append(f"目录不可写：{directory}（{exc}）")

    if failures:
        print("[失败] 离线检查")
        for failure in failures:
            print(" -", failure)
        return 1

    print("[通过] 离线检查")
    print("接口地址：", config.BASE_URL)
    print("日志目录：", config.LOG_DIR)
    print("结果目录：", config.RESULT_DIR)
    print("未发送 /enter 请求。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
