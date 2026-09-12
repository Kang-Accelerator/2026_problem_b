# B 题通信框架

本目录包含 2026 年高教社杯数学建模竞赛 B 题的可复用通信协议基础和离线测试基础。这里暂不选择问题 2 至问题 4 的数学算法。

问题 1 的离线实现只保留定位几何量、覆盖判定以及 `guaranteed_clearable`、`ready_to_clear` 两项清除判据，不设置分档评价字段；不连接官方模拟器，也不调用 `/enter`。

## 运行离线检查

在项目根目录执行：

```cmd
.venv\Scripts\python.exe -m unittest discover -s competition_b\tests -v
.venv\Scripts\python.exe competition_b\preflight.py
```

`preflight.py` 永远不会调用 `/enter`。

## 运行通信演练示例

先在官方模拟器中选择演练测试，并等待接口就绪，然后执行：

```cmd
.venv\Scripts\python.exe competition_b\demo_practice.py --robot-id YOUR_TEAM_ID
```

该示例使用官方给出的四个动作序列，仅用于说明通信，不是搜索策略。
