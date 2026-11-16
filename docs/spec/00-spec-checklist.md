# PorryAgent 设计基线

> 本文件只承载「原则」与「范围」，是唯一权威基线。
> 具体阈值、工具签名、压缩参数、冲突权衡，在做到对应模块时讨论确定。
> 优先级：硬约束 > 即时口头指令 > 工程偏好。冲突时停下说明，不自行取舍。

## 一、介绍

PorryAgent 是从零构建的生产级 CodingAgent，Python 3.11+，用 uv 管理解释器与依赖。

结构内容：

- 单 Agent 主循环 + 子 Agent 工具（独立上下文、工具子集、只回摘要）。
- OpenAI 兼容 provider（base_url 可配）+ mock provider（离线测试）。
- 分层上下文压缩。
- 任务树与状态机、意图变更、持久化（JSON 快照 + JSONL 事件 + 文件锁）、中断后 resume。
- 结构化日志、trace、按 run 累计成本。

## 二、架构原则

子系统分组

### 消息与上下文

1. **块模型与配对**：内部消息用 provider-neutral 块模型（TextBlock / ToolUseBlock / ToolResultBlock），tool_call_id 显式关联；裁剪/压缩以「tool_use + 全部对应 result」为原子单元，同删同留，禁止只删结果。
2. **原始不可变 + 读时投影**：原始消息日志 append-only、不可变；发给模型的是投影生成的只读视图；压缩是提交记录，不是原地改写；投影幂等。
3. **前缀稳定**：静态内容（system、tool schema、项目指令）固定最前，动态消息按时间线，易变内容靠后；tool_use 先于对应 result。
4. **分层压缩**：便宜的先跑、贵的后跑；只改视图不改原始日志；系统/工具 schema、当前目标与任务、未解决错误、正在编辑的文件优先保留，压缩不得删除。

### 主循环与终止

1. **循环与终止**：think → act → observe；终止条件是「模型不再返回 tool call」，不是文本里说「完成了」；硬上限（轮次/单轮工具数/墙钟/成本）任一触发进入明确终态并记录原因。
2. **错误与重试**：错误分类（transient/permanent/tool/context_overflow/protocol）；瞬时退避重试、永久立即失败、重复失败熔断；context_overflow 先强制压缩再用新视图重试一次，仍失败则终态。

### 任务与状态

1. **任务树与状态机**：单根目标、依赖是有向无环图；任务只走合法状态迁移（pending → ready → in_progress → done/failed/cancelled/superseded），blocked 由依赖实时推导；认领是原子操作，父任务在子任务未终态前不能完成。
2. **意图变更**：检测显式纠正/新目标/与当前目标矛盾，生成意图修订记录；受影响的旧任务标 superseded/cancelled，不物理删除，历史可查；破坏性变更需用户确认。
3. **持久化与恢复**：JSON 快照 + JSONL 事件追加，原子写、带 schema_version，先写事件再写快照；SIGINT 只在安全点生效，保存 checkpoint 后退出；resume 从最后完成步骤继续，工具调用带幂等键，副作用未知时先只读探测、不盲目重放。

### 工具与权限

1. **工具契约**：工具统一注册描述（name/schema/handler/read_only 等），结果统一结构（ok/content/error_type/retryable）；异常转 observation，不把裸栈抛进主循环。
2. **安全编辑**：先读后写；old_string 唯一匹配，0 或多匹配拒绝并给候选行号；原子写（tmp + os.replace）；写后重读校验；失败回滚；外部 mtime/hash 变化强制重读。
3. **边界与沙箱声明**：路径解析到 workspace root，拒绝 `..` 与 symlink 逃逸；shell 有固定 cwd、超时、输出上限、非交互；明确声明「不是安全沙箱」。 
4. **子 Agent 隔离**：独立 messages、独立 system prompt、自包含任务简报、工具子集、可选模型路由、最多 N 轮；只回摘要，不回传中间工具历史，噪音不进主上下文。

### 观测、成本与密钥

1. **可观测与成本**：结构化日志 + 每请求/每压缩 trace + 成本按 run 累计可量化；provider 声明能力，不支持的走降级、不假装。
2. **密钥与配置**：密钥只从 env 读取，日志与异常脱敏，缺配置给「缺哪个变量、怎么设」的提示而非堆栈。

补充：四个存量模块（CC-Context、ContextManagerModel、Eval、MultiAgent）是参考实现，按模块参考设计思想。

## 四、权威与落地

- 历史决策与原文存档见 `docs/reference/`