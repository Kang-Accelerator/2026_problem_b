"""官方四动作演练示例。

仅在模拟器中选择演练测试并等待接口就绪后运行。本示例不是搜索策略。
"""

from __future__ import annotations

import argparse
import time

try:
    from . import config
    from .action_logger import ActionLogger
    from .simulator_client import SimulatorClient, SimulatorClientError
except ImportError:  # 支持从项目根目录直接执行。
    import config
    from action_logger import ActionLogger
    from simulator_client import SimulatorClient, SimulatorClientError


class ChineseArgumentParser(argparse.ArgumentParser):
    """将 argparse 自动生成的帮助标题显示为中文。"""

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法：")

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法：").replace("options:", "选项：").replace("optional arguments:", "可选参数：")


def main() -> int:
    parser = ChineseArgumentParser(description="官方四动作通信演练示例", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示此帮助信息并退出")
    parser.add_argument("--robot-id", default=config.ROBOT_ID, help="队伍标识，默认读取 CUMCM_ROBOT_ID")
    parser.add_argument("--base-url", default=config.BASE_URL, help="模拟器接口地址，默认读取 CUMCM_BASE_URL")
    args = parser.parse_args()

    if not args.robot_id or "<" in args.robot_id:
        parser.error("请通过 --robot-id 或 CUMCM_ROBOT_ID 提供真实队伍标识")

    print("请确认模拟器当前选择的是演练测试，并且接口已经就绪。")
    log_path = config.LOG_DIR / f"practice_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    logger = ActionLogger(log_path)
    client = SimulatorClient(base_url=args.base_url, robot_id=args.robot_id, logger=logger)
    entered = False
    end_reason = "not_started"

    try:
        enter_response = client.enter()
        entered = True
        print("/enter:", enter_response)
        print("remaining_real_duration_s:", enter_response["remaining_real_duration_s"])

        actions = [
            ("/measure", lambda: client.measure(300, 400, 1)),
            ("/measure", lambda: client.measure(300, 400, 2)),
            ("/clear", lambda: client.clear(300, 0, 3)),
            ("/measure", lambda: client.measure(300, 0, 2)),
        ]
        for name, action in actions:
            response = action()
            print(name, response)

        exit_response = client.exit()
        entered = False
        print("/exit:", exit_response)
        end_reason = str(exit_response.get("exit_reason", "unknown"))
        return 0
    except (SimulatorClientError, KeyError, ValueError) as exc:
        print("演练示例失败：", exc)
        end_reason = f"错误：{exc}"
        return 1
    finally:
        if entered:
            print("程序未尝试自动调用 /exit；请根据模拟器状态手动结束演练。")
        print(logger.summary(virtual_time_s=client.state.last_virtual_time_s, end_reason=end_reason))


if __name__ == "__main__":
    raise SystemExit(main())
