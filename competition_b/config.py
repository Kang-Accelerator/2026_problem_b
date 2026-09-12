"""B 题通信框架的配置。

队伍标识保存在本文件对应的配置中，也可以通过
CUMCM_ROBOT_ID 环境变量提供。策略模块应导入这些常量，不要重复定义协议值。
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
BASE_URL = os.getenv("CUMCM_BASE_URL", "http://127.0.0.1:2026").rstrip("/")
ROBOT_ID = os.getenv("CUMCM_ROBOT_ID", "")

TARGET_RADIUS_M = 1800.0
MIN_RECEIVE_RADIUS_M = 1000.0
DOG_SPEED_MPS = 5.0
BEARING_ERROR_DEG = 1.0
NEAR_RADIUS_M = 5.0
CLEAR_RADIUS_M = 20.0
SAFE_CLEAR_RADIUS_M = 18.0

HTTP_TIMEOUT_S = 5.0
MAX_NETWORK_RETRIES = 3
LOG_DIR = PROJECT_DIR / "logs"
RESULT_DIR = PROJECT_DIR / "results"
