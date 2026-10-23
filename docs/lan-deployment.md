# 本机 JEV + 远端 VLA

采用相机与 VLA 同机、JEV 模型独立推理的方式。沿用 VLA 的现有权重、预处理、动作采样和控制队列。

```text
相机 / 夹爪反馈（VLA 主机）
  ├─ 三路原始图像 + state ────────────── VLA 原推理入口
  │                                   ↑ 仅前相机叠加红框
  └─ 最新前相机 → JEV HTTP 工作线程 → JEV 模型主机
                    ↓ 识别结果
              SEARCH → SHAPE → TRACK
                    ↓ 目标版本 / 框坐标
              VLA 推理、RTC 队列失效检查
```

三路图像留在 VLA 主机。JEV 请求只发送 SEARCH 所需的小图，或 SHAPE 的垫子裁剪图和参考图；回复为结构化判断。几何规则、释放检测和目标版本在 VLA 主机维护。整个抓取期间保持已确认的篮子，释放后清除，不要求 JEV 每轮跟随 VLA 重新识别。

## 1. 在 JEV 主机启动服务

使用模型原有的 Python 环境，从本仓库执行：

```bash
/path/to/jev/.venv/bin/python scripts/serve_jev.py \
  --model /path/to/StartLux-Decision-27B \
  --host 172.16.52.48 --port 18091
```

启动入口调用模型目录自带的 `startlux_decision.server`，只为 HTTP 配置连接复用和 `TCP_NODELAY`。模型计算、权重和原生参数由外部模型包负责，其他原生 CLI 参数可以继续传入。该入口应作为独立进程运行，不能在线程中嵌入另一个应用。

## 2. 在 VLA 主机接入

把本仓库安装在 VLA 的 Python 环境中，使用已安装相机与反馈线程的回调：

```python
from PIL import Image
from jevfabric.deployment import LivePolicyBridge
from jevfabric.live import LiveAnnotator
from jevfabric.replay import load_config

config = load_config("configs/live.lan.example.json")
with Image.open("/path/to/labelled_pieces.png") as reference:
    annotator = LiveAnnotator(reference, config).start()
bridge = LivePolicyBridge(annotator)

# 原相机线程：bridge.on_observation(obs)
# 原反馈线程：bridge.on_feedback(raw_state, observed_at_ns=timestamp_ns)
# 原 VLA 生产线程：candidate = bridge.predict(obs, existing_policy_infer)
# 原 RTC 消费路径：bridge.is_current(candidate)，失效则丢弃
```

完整异常处理、时间戳和 RTC 示例见根目录 README。相机和反馈时间戳必须来自 **VLA 主机进程的 monotonic 时钟域**，不要使用 JEV 主机时钟做过期判断。反馈为原始米制夹爪开度，不能传归一化 state 或动作指令。

JEV 结果到达后，红框只应用于它生效之后采集的画面。推理结束时重新检查目标版本和输入年龄。RTC 必须保留版本信息，并在选取动作时检查，包括已经开始消费的 chunk；只有推理结束时检查一次并不够。

## 3. 局域网配置示例

`configs/live.lan.example.json` 是本次链路的示例；修改主机地址、端口和相机布局后使用。

| 参数 | 示例值 | 含义 |
|---|---:|---|
| `request_interval_s` | 0.01 | 单次 JEV 请求完成后的最小间隔，仍为单请求串行 |
| `request_timeout_s` | 0.5 | JEV 请求截止时间 |
| `max_result_age_s` | 0.5 | JEV 原始帧到结果被接受的最大年龄 |
| `max_frame_age_s` | 0.25 | 输入新鲜度，亦用于拒绝迟到的 VLA 回复 |
| `max_feedback_age_s` | 0.25 | 夹爪反馈新鲜度 |
| `geometry_hz` | 15 | 几何规则的最高更新频率 |

这些是数据有效性和重试参数，**不是任务成功率开始下降的延迟临界值**。该临界值尚未测定。若现有 VLA 耗时超过输入年龄预算，应按其已验证的运行条件配置预算；不能仅为了让慢请求通过而无条件放宽。

`LiveAnnotator` 的两个工作线程之外，不再创建相机、硬件轮询或控制线程。网络请求、编码、几何计算与绘图不占用状态锁；最新帧槽容量为一。结束时先停止回调，再关闭 annotator，并检查线程已退出。

## 4. 离线回放与范围

```bash
python scripts/replay_live.py \
  --dataset /path/to/raw_lerobot_v2_1 --episode 105 \
  --front-key images.cam_front --state-key state \
  --reference /path/to/labelled_pieces.png \
  --config configs/live.lan.example.json --seconds 40 \
  --out output/lan-replay
```

`report.json` 与 `events.jsonl` 保存线程、延时、分类、画框、释放和过期丢弃信息。回放不打开真机设备，也不下发动作。

本仓库提供实时标注模块及部署接入接口。具体硬件运行器仍需安装上述回调与 RTC 失效检查，未在本次整理中自动修改或启动。临时 VLA 服务、CUDA Graph 实验、JPEG 对照和大量测试日志不纳入正式代码。
