"""官方 B 题模拟器的协议安全客户端。

本模块只实现 /enter、/measure、/clear 和 /exit，不包含搜索或定位策略。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .action_logger import ActionLogger
    from . import config
except ImportError:  # 支持从本目录直接执行脚本。
    from action_logger import ActionLogger
    import config


class SimulatorClientError(RuntimeError):
    """协议或传输失败的基础异常。"""


class RuntimeBudgetExceeded(SimulatorClientError):
    """现实时间预算耗尽时，在新动作执行前抛出。"""


@dataclass
class ClientState:
    position: tuple[float, float] = (0.0, 0.0)
    current_measure_channel: int = 1
    last_virtual_time_s: float = 0.0
    remaining_real_duration_s: float | None = None
    entered: bool = False
    cleared_channels: set[int] = field(default_factory=set)


class SimulatorClient:
    """面向官方 HTTP+JSON API 的串行幂等客户端。"""

    def __init__(
        self,
        *,
        base_url: str = config.BASE_URL,
        robot_id: str = config.ROBOT_ID,
        logger: ActionLogger | None = None,
        timeout_s: float = config.HTTP_TIMEOUT_S,
        max_network_retries: int = config.MAX_NETWORK_RETRIES,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.timeout_s = timeout_s
        self.max_network_retries = max(0, max_network_retries)
        self.opener = opener
        self.logger = logger
        self.state = ClientState()
        self._request_counter = 0
        self._real_deadline: float | None = None

    def _new_request_id(self, action: str) -> str:
        self._request_counter += 1
        return f"{action}-{self._request_counter}"

    def _base_payload(self, request_id: str) -> dict[str, Any]:
        if not self.robot_id:
            raise SimulatorClientError("robot_id 为空；请设置 CUMCM_ROBOT_ID 或传入 robot_id")
        return {
            "arena_id": "default",
            "robot_id": self.robot_id,
            "request_id": request_id,
        }

    @staticmethod
    def _validate_position(x: float, y: float) -> None:
        if not all(math.isfinite(value) and abs(value) <= 2_000_000 for value in (x, y)):
            raise ValueError("x 和 y 必须是有限数，且绝对值不超过 2000000")

    @staticmethod
    def _validate_channel(channel: int) -> None:
        if not isinstance(channel, int) or isinstance(channel, bool) or not 1 <= channel <= 20:
            raise ValueError("channel 必须是 [1, 20] 内的整数")

    def _ensure_budget(self) -> None:
        if self._real_deadline is not None and time.monotonic() >= self._real_deadline:
            raise RuntimeBudgetExceeded("剩余现实时间预算已耗尽")

    def _record(self, event: dict[str, Any]) -> None:
        if self.logger is not None:
            self.logger.record(event)

    def _post(self, path: str, payload: dict[str, Any], action: str, channel: int | None = None) -> dict[str, Any]:
        """发送一个动作；仅在传输失败后重试完全相同的动作。"""
        if path not in {"/enter", "/measure", "/clear", "/exit"}:
            raise ValueError(f"不支持的模拟器路径：{path}")

        last_error: Exception | None = None
        for attempt in range(self.max_network_retries + 1):
            started = time.perf_counter()
            response: dict[str, Any] | None = None
            http_status: int | None = None
            error_text: str | None = None
            try:
                request = Request(
                    self.base_url + path,
                    data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.opener(request, timeout=self.timeout_s) as http_response:
                    http_status = getattr(http_response, "status", None) or getattr(http_response, "code", None)
                    body = http_response.read().decode("utf-8")
                response = json.loads(body)
                if not isinstance(response, dict):
                    raise SimulatorClientError("模拟器响应不是 JSON 对象")
                if http_status is not None and not 200 <= http_status < 300:
                    raise SimulatorClientError(f"HTTP {http_status}：{response}")
                if response.get("accepted") is not True:
                    raise SimulatorClientError(f"模拟器拒绝了 {path}：{response}")
                self._record({
                    "real_elapsed_s": round(time.perf_counter() - started, 6),
                    "request_id": payload["request_id"],
                    "action": action,
                    "requested_position": payload.get("position"),
                    "channel": channel,
                    "http_status": http_status,
                    "attempt": attempt + 1,
                    "response": response,
                    "error": None,
                })
                return response
            except HTTPError as exc:
                http_status = exc.code
                try:
                    response = json.loads(exc.read().decode("utf-8"))
                except (JSONDecodeError, UnicodeDecodeError):
                    response = None
                error_text = f"HTTP {http_status}：{response or exc.reason}"
                self._record({
                    "real_elapsed_s": round(time.perf_counter() - started, 6),
                    "request_id": payload["request_id"],
                    "action": action,
                    "requested_position": payload.get("position"),
                    "channel": channel,
                    "http_status": http_status,
                    "attempt": attempt + 1,
                    "response": response,
                    "error": error_text,
                })
                # 明确的客户端拒绝必须立即报告。
                if http_status not in {429, 500, 502, 503, 504}:
                    last_error = SimulatorClientError(error_text)
                    break
                last_error = SimulatorClientError(error_text)
            except (URLError, TimeoutError, ConnectionError, OSError, JSONDecodeError, UnicodeDecodeError) as exc:
                last_error = exc
                error_text = f"传输异常：{exc!r}"
            except SimulatorClientError as exc:
                last_error = exc
                error_text = str(exc)
                if response is not None and response.get("accepted") is not True:
                    self._record({
                        "real_elapsed_s": round(time.perf_counter() - started, 6),
                        "request_id": payload["request_id"],
                        "action": action,
                        "requested_position": payload.get("position"),
                        "channel": channel,
                        "http_status": http_status,
                        "attempt": attempt + 1,
                        "response": response,
                        "error": error_text,
                    })
                    break

            self._record({
                "real_elapsed_s": round(time.perf_counter() - started, 6),
                "request_id": payload["request_id"],
                "action": action,
                "requested_position": payload.get("position"),
                "channel": channel,
                "http_status": http_status,
                "attempt": attempt + 1,
                "response": response,
                "error": error_text,
            })
            if attempt < self.max_network_retries:
                time.sleep(min(0.05 * (2**attempt), 0.2))

        raise SimulatorClientError(f"动作 {path} 在有限次重试后仍失败：{last_error}") from last_error

    def _update_virtual_time(self, response: dict[str, Any]) -> None:
        # accepted=false 响应使用 virtual_time_s=0，不会进入这里。
        value = response.get("virtual_time_s")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self.state.last_virtual_time_s = float(value)

    def enter(self) -> dict[str, Any]:
        if self.state.entered:
            raise SimulatorClientError("此客户端已经成功调用过 /enter")
        payload = self._base_payload(self._new_request_id("enter"))
        response = self._post("/enter", payload, "enter")
        self._update_virtual_time(response)
        remaining = response.get("remaining_real_duration_s")
        if not isinstance(remaining, (int, float)) or isinstance(remaining, bool):
            raise SimulatorClientError("/enter 响应缺少数值型 remaining_real_duration_s")
        self.state.remaining_real_duration_s = float(remaining)
        self._real_deadline = time.monotonic() + float(remaining)
        self.state.entered = True
        return response

    def measure(self, x: float, y: float, channel: int) -> dict[str, Any]:
        self._ensure_budget()
        if not self.state.entered:
            raise SimulatorClientError("请先调用 enter()，再调用 measure()")
        self._validate_position(x, y)
        self._validate_channel(channel)
        payload = self._base_payload(self._new_request_id("measure"))
        payload.update({"position": {"x": x, "y": y}, "channel": channel})
        response = self._post("/measure", payload, "measure", channel)
        self.state.position = (float(x), float(y))
        self.state.current_measure_channel = channel
        self._update_virtual_time(response)
        return response

    def clear(self, x: float, y: float, channel: int) -> dict[str, Any]:
        self._ensure_budget()
        if not self.state.entered:
            raise SimulatorClientError("请先调用 enter()，再调用 clear()")
        self._validate_position(x, y)
        self._validate_channel(channel)
        payload = self._base_payload(self._new_request_id("clear"))
        payload.update({"position": {"x": x, "y": y}, "channel": channel})
        response = self._post("/clear", payload, "clear", channel)
        self.state.position = (float(x), float(y))
        # /clear 不改变 current_measure_channel。
        if response.get("clear_result") == "success":
            self.state.cleared_channels.add(channel)
        self._update_virtual_time(response)
        return response

    def exit(self) -> dict[str, Any]:
        if not self.state.entered:
            raise SimulatorClientError("请先调用 enter()，再调用 exit()")
        payload = self._base_payload(self._new_request_id("exit"))
        response = self._post("/exit", payload, "exit")
        self._update_virtual_time(response)
        self.state.entered = False
        return response
