"""供客户端和演练程序使用的简易 JSONL 动作日志记录器。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ActionLogger:
    """每行追加一个 JSON 对象，并保留简要的运行摘要。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started_at = time.perf_counter()
        self.measure_count = 0
        self.clear_attempt_count = 0
        self.cleared_channels: set[int] = set()

    def record(self, event: dict[str, Any]) -> None:
        """写入一个事件；日志写入失败时抛出错误并停止当前动作。"""
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise RuntimeError(f"无法写入动作日志：{self.path}") from exc

        action = event.get("action")
        if action == "measure":
            self.measure_count += 1
        elif action == "clear":
            self.clear_attempt_count += 1
            response = event.get("response") or {}
            if response.get("clear_result") == "success":
                channel = event.get("channel")
                if isinstance(channel, int):
                    self.cleared_channels.add(channel)

    def summary(self, *, virtual_time_s: float | None, end_reason: str) -> dict[str, Any]:
        """返回适合控制台输出或结果 JSON 文件的摘要值。"""
        cleared = len(self.cleared_channels)
        return {
            "cleared_count": cleared,
            "measure_count": self.measure_count,
            "clear_attempt_count": self.clear_attempt_count,
            "average_localization_clear_time_s": None if cleared == 0 else "策略指标尚未提供",
            "virtual_time_s": virtual_time_s,
            "program_runtime_s": round(time.perf_counter() - self.started_at, 6),
            "end_reason": end_reason,
        }
