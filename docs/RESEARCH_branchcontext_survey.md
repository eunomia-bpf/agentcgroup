# Branch Context Motivation 调研文档

本文档包含不需要新实验、通过调研/分析即可完成的 motivation 材料。

---

## 1. 现有 Agent 框架的隔离机制调研

论文声称 "current agent frameworks resort to ad hoc solutions"，以下是各框架的具体证据。

### 1.1 Claude Code — per-file checkpoint

**隔离方式**：Claude Code 使用 session-level checkpoint，在每次 file edit 工具调用前自动捕获文件状态。用户可通过 `Esc Esc` 或 `/rewind` 命令回退到任意 checkpoint。

**具体机制**：
- 每个用户 prompt 创建一个 checkpoint
- 支持 4 种恢复选项：恢复代码+对话、仅恢复对话、仅恢复代码、从此处摘要压缩
- Checkpoint 跨 session 持久化，30 天后自动清理

**关键局限**：
- **Bash 命令的文件修改不被追踪**：`rm`, `mv`, `cp`, `pip install`, `npm install` 等通过 bash 执行的文件变更**无法通过 rewind 回滚**。官方文档明确说明："Checkpointing does not track files modified by bash commands. These file modifications cannot be undone through rewind."
- **不追踪外部变更**：其他并发 session 的编辑和用户手动修改不被捕获
- **不支持并行分支**：没有并行探索机制。文档建议用 `--continue --fork-session` 创建 session fork，但这只是对话层面的分支，不是文件系统层面的
- **不追踪进程/内存/网络状态**：完全没有进程状态隔离

**来源**：[Claude Code Checkpointing 官方文档](https://code.claude.com/docs/en/checkpointing)；[GitHub Issue #6001](https://github.com/anthropics/claude-code/issues/6001)；[GitHub Issue #4472](https://github.com/anthropics/claude-code/issues/4472)

---

### 1.2 SWE-agent — Docker 容器 + git patch

**隔离方式**：SWE-agent 在 Docker 容器内运行 agent，通过 git diff 提取最终 patch。

**具体机制**：
- 每个任务在独立的 Docker 容器中执行
- 容器内将仓库 reset 到指定 commit（默认 HEAD）
- Agent 修改文件后，通过 `git diff` 生成 patch 作为输出
- 容器在任务结束后销毁

**关键局限**：
- **git add -A 的污染问题**：[mini-swe-agent Issue #528](https://github.com/SWE-agent/mini-swe-agent/issues/528) 报告，当使用 `git add -A` 生成 patch 时，环境相关的配置文件变更（如 `pyproject.toml`, `setup.cfg`）也被意外包含，导致 patch 不纯净
- **不支持并行探索**：SWE-agent 是串行执行的，每个任务一个容器，没有分支机制
- **容器级隔离过重**：每个任务需要启动一个完整的 Docker 容器（SWE-bench 镜像 2.9-17.3GB），无法做到亚毫秒级分支
- **无增量回滚**：容器内没有 checkpoint 机制，只能整个重启

**来源**：[SWE-agent GitHub](https://github.com/SWE-agent/SWE-agent)；[SWE-agent 文档](https://swe-agent.com/latest/reference/repo/)；[SWE-bench Issue #465](https://github.com/SWE-bench/SWE-bench/issues/465)

---

### 1.3 OpenHands — Docker sandbox + event-sourcing

**隔离方式**：OpenHands 使用 Docker 容器作为 sandbox，配合 event-sourcing 架构记录所有操作。

**具体机制**：
- 每个 agent session 运行在独立的 Docker 容器中（`DockerWorkspace`）
- 所有交互（命令、编辑、结果）记录为**不可变事件**追加到 event log
- Session 结束后容器销毁，确保文件系统完整性
- V1 架构重构为模块化 SDK，sandbox 成为 opt-in 组件

**关键局限**：
- **没有显式的 rollback 机制**：event-sourcing 支持重建历史状态，但不支持实时回滚到之前的文件系统状态
- **容器启动开销大**：[Issue #1637](https://github.com/OpenHands/OpenHands/issues/1637) 报告 sandbox timeout 问题，说明容器启动延迟是实际问题
- **不支持并行探索**：每个 session 一个容器，没有分支机制
- **不支持嵌套分支**：容器内无进一步的隔离手段

**来源**：[OpenHands ICLR 2025 论文](https://openreview.net/pdf/95990590797cff8b93c33af989ecf4ac58bde9bb.pdf)；[OpenHands SDK 论文](https://arxiv.org/html/2511.03690v1)；[Docker Sandbox 文档](https://docs.openhands.dev/sdk/guides/agent-server/docker-sandbox)

---

### 1.4 Aider — git auto-commit + /undo

**隔离方式**：Aider 在每次 LLM 编辑后自动创建 git commit，通过 `/undo` 命令回滚。

**具体机制**：
- 每次 LLM 修改文件后，自动 `git commit` 并生成描述性消息
- 如果文件有未提交的修改，先提交用户的修改，再提交 Aider 的修改（保持分离）
- `/undo` 回滚最近一个由 Aider 创建的 commit

**关键局限**：
- **仅追踪 git 管理的文件**：`npm install`, `pip install`, 构建产物等 git 不追踪的变更完全不可回滚
- **多次 commit 的复杂性**：单次运行可能产生多个 commit，回滚多步变得复杂
- **/undo 在 no-auto-commit 模式下无效**：[Issue #1528](https://github.com/Aider-AI/aider/issues/1528) 报告此问题
- **缺少 prompt 级别的 undo**：[Issue #1018](https://github.com/paul-gauthier/aider/issues/1018) 请求 `/undo-prompt` 功能
- **不支持并行探索**：纯串行执行

**来源**：[Aider Git 集成文档](https://aider.chat/docs/git.html)；[Issue #1528](https://github.com/Aider-AI/aider/issues/1528)；[Issue #76](https://github.com/paul-gauthier/aider/issues/76)

---

### 1.5 Windsurf — git worktree 并行

**隔离方式**：Windsurf Wave 13 引入了 git worktree 支持，每个 Cascade agent session 获得独立的工作目录。

**具体机制**：
- 每个 Cascade 对话可以在独立的 git worktree 中运行
- 最多 20 个 worktree per workspace，自动清理最久未访问的
- 支持多 agent 并行，在 pane/tab 中并排查看
- 提供 `post_setup_worktree` hook 用于初始化（复制 .env、安装依赖）

**关键局限**：
- **不包含未版本控制的文件**：文档明确说 worktree "does not include `.env` files or other packages that aren't version-controlled"
- **路径依赖可能破坏**：构建系统或工具如果依赖相对路径（如 `../shared-lib`），在 worktree 中可能失败
- **只能在 session 开始时选择 worktree 模式**：运行中无法切换
- **无进程隔离**：文档中完全没有提到进程级隔离，隔离仅限于文件系统（git 追踪的部分）
- **无 commit/abort 语义**：没有 first-commit-wins，需要手动 merge

**来源**：[Windsurf Worktrees 文档](https://docs.windsurf.com/windsurf/cascade/worktrees)；[Wave 13 博客](https://windsurf.com/blog/windsurf-wave-13)

---

### 1.6 Cursor — checkpoint + git worktree（2.0）

**隔离方式**：Cursor 在 Composer 发送 prompt 时自动创建 checkpoint，2.0 版本引入 git worktree 并行。

**具体机制**：
- 自动 checkpoint（记录文件 hash、git branch、变更文件列表）
- Cursor 2.0：支持最多 8 个 agent 并行，每个 agent 在独立的 git worktree 或 remote machine 中运行

**关键局限**：
- **Rollback 可靠性极差**：[论坛帖子](https://forum.cursor.com/t/rollback-fails-in-cursor-checkpoint-restore-doesn-t-work-either/122069) 报告 rollback 失败率 92%
- **undo 行为不可预测**：接受 AI 建议后 undo 行为不一致，关闭文件后 undo 历史丢失
- **VCS 才是真正的回滚机制**：官方建议依赖 git 而非 checkpoint
- **2.0 的 worktree 隔离**与 Windsurf 相同的局限：不包含未追踪文件，无进程隔离

**来源**：[Cursor Forum - Rollback Fails](https://forum.cursor.com/t/rollback-fails-in-cursor-checkpoint-restore-doesn-t-work-either/122069)；[Cursor Checkpoints 指南](https://stevekinney.com/courses/ai-development/cursor-checkpoints)

---

### 1.7 汇总对比表

| 框架 | 隔离机制 | 覆盖 git 外文件 | 进程隔离 | 并行探索 | commit/abort | 嵌套 |
|------|---------|---------------|---------|---------|-------------|------|
| **Claude Code** | per-file checkpoint | ✗ | ✗ | ✗ | ✗ | ✗ |
| **SWE-agent** | Docker 容器 | ✓（容器级） | ✓（容器级） | ✗ | ✗ | ✗ |
| **OpenHands** | Docker 容器 + event log | ✓（容器级） | ✓（容器级） | ✗ | ✗ | ✗ |
| **Aider** | git auto-commit | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Windsurf** | git worktree | ✗ | ✗ | ✓（worktree） | ✗ | ✗ |
| **Cursor 2.0** | checkpoint + worktree | ✗ | ✗ | ✓（worktree） | ✗ | ✗ |
| **Branch Context** | CoW FS + 进程组 | ✓ | ✓ | ✓ | ✓ | ✓ |

**关键发现**：
1. **所有 git 基方案**（Claude Code、Aider、Windsurf、Cursor）都**无法回滚 shell 命令的副作用**（npm install、pip install、build artifacts）
2. **Docker 基方案**（SWE-agent、OpenHands）提供完整隔离但**过重**（容器启动 160-170 秒，无亚毫秒级分支）
3. **没有任何框架**同时支持并行探索 + 完整状态覆盖 + commit/abort 语义 + 嵌套
4. **Windsurf 和 Cursor 2.0** 是最接近的——它们使用 git worktree 实现并行，但仍然只隔离 git 追踪的文件

---

## 2. 学术框架的状态处理调研

### 2.1 Tree-of-Thoughts (ToT) — 假设无副作用

**论文**：Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (NeurIPS 2023)

**状态处理**：ToT 的 BFS/DFS 搜索在纯文本空间进行，每个节点是一个"thought"（文本片段）。搜索过程中不涉及环境交互，因此**不存在副作用问题**。

**与 branch context 的关系**：ToT 本身不需要文件系统隔离，但当 ToT 模式应用于**有副作用的环境**（如 agent 执行 shell 命令）时，每个 thought 分支都需要独立的执行环境。当前没有框架支持这种"ToT + 环境隔离"的组合。

**来源**：[ToT Prompt Engineering Guide](https://www.promptingguide.ai/techniques/tot)；[Demystifying Chains, Trees, and Graphs of Thoughts](https://arxiv.org/html/2401.14295v3)

### 2.2 Graph-of-Thoughts (GoT) — 同样假设无副作用

**论文**：Besta et al., "Graph of Thoughts: Solving Elaborate Problems with Large Language Models" (AAAI 2024)

**状态处理**：GoT 扩展了 ToT，允许 thought 之间有任意依赖关系（而非仅树结构）。但同样在纯文本空间操作，不涉及环境副作用。

**与 branch context 的关系**：GoT 的任意图结构实际上比 branch context 的树结构更复杂。如果要在有副作用的环境中实现 GoT，甚至需要比 branch context 更强的隔离原语（如 merge 多个分支的状态）。

### 2.3 Reflexion — 假设环境可重置

**论文**：Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023)

**状态处理**：Reflexion 在每次 trial 失败后，agent 进行"自我反思"生成文字反馈，然后**重置环境**开始新的 trial。核心机制是 heuristic function 判断何时停止当前 trajectory（效率过低或产生幻觉），然后 reset。

**关键假设**："after each action, the agent computes a heuristic and optionally may decide to **reset the environment** to start a new trial"——Reflexion **假设环境可以被廉价地重置**，但没有说明如何实现这一点。

**与 branch context 的关系**：Reflexion 的 "reset environment" 正是 branch context 的 "abort" 操作。当前实现依赖于环境提供的 reset API（如 AlfWorld 的 `env.reset()`），但对于**软件工程 agent** 来说，"reset environment" 意味着回滚所有文件系统变更、终止所有进程——这正是 branch context 要解决的问题。

**来源**：[Reflexion 论文](https://arxiv.org/abs/2303.11366)；[Reflexion GitHub](https://github.com/noahshinn/reflexion)

### 2.4 LATS — 假设状态可重放

**论文**：Zhou et al., "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models" (ICML 2024)

**状态处理**：LATS 使用 Monte Carlo Tree Search (MCTS) 探索 action 空间。对于状态恢复，LATS 利用了 LM 任务的特殊性质——通过**重放对话历史**来重建状态：

> "we can conveniently revert to any state by setting the input to be the context and the corresponding previous output from the LM"

**关键局限**：论文明确承认：

> "LATS assumes the ability to revert to earlier states in decision-making environments, **which may not be universally applicable in all possible environments**."

这在文本任务（QA、web 导航）中有效，但在有永久副作用的环境中（文件系统修改、包安装）**完全不可行**。

**与 branch context 的关系**：LATS 的"重放历史重建状态"假设在 agent 执行 shell 命令时不成立。`pip install X` 不能通过重放来撤销。Branch context 提供了 LATS 缺少的**真正的状态隔离和回滚**。

**来源**：[LATS 论文](https://arxiv.org/abs/2310.04406)；[LATS GitHub](https://github.com/lapisrocks/LanguageAgentTreeSearch)

### 2.5 汇总：学术框架对状态的假设

| 框架 | 状态模型 | 副作用处理 | 回滚机制 | 适用于 SE agent? |
|------|---------|-----------|---------|-----------------|
| **ToT** | 纯文本 | 无副作用 | 不需要 | ✗（无环境交互） |
| **GoT** | 纯文本 | 无副作用 | 不需要 | ✗（无环境交互） |
| **Reflexion** | 假设可重置 | 依赖 env.reset() | 环境提供 | 部分（需要 reset 实现） |
| **LATS** | 重放历史 | 假设可重放 | 重放对话 | ✗（shell 命令不可重放） |

**核心结论**：所有搜索/探索类学术框架要么**假设无副作用**（ToT/GoT），要么**假设环境可廉价重置**（Reflexion/LATS），但**没有任何框架提供实际的状态隔离和回滚实现**。当这些框架应用于有副作用的软件工程场景时，状态隔离成为核心瓶颈。

---

## 3. 相关系统工作

### 3.1 DAXFS — 内存级分支文件系统

DAXFS 是一个基于 DAX（Direct Access）的内存原生文件系统，支持 copy-on-write 分支和 single-winner 语义。

**关键特性**：
- Read-only base image + CoW branches
- 支持任意深度嵌套
- Single-winner commit 语义
- 零拷贝读取（direct load/store，无 page cache）
- 用例之一：AI agent speculative execution

**局限**：
- 依赖 DAX-capable memory（persistent memory, CXL memory, DMA buffers）——**不可移植**
- 不适用于普通文件系统（ext4, XFS）

**与 branch context 的关系**：DAXFS 在文件系统层面提供了与 branch context 类似的语义（分支、commit、sibling invalidation），但仅限于特定硬件。BranchFS 通过 FUSE 实现了类似语义但可在任意文件系统上运行。

**来源**：[DAXFS GitHub](https://github.com/multikernel/daxfs)；[Phoronix 报道](https://www.phoronix.com/news/DAXFS-Linux-File-System)

### 3.2 AgentFS (Turso) — SQLite-backed agent 文件系统

AgentFS 是 Turso 开发的 agent 专用文件系统，提供 CoW overlay 和 sandbox。

**关键特性**：
- Linux: FUSE + user/mount namespace，bind-mount 可写路径，其余 read-only
- macOS: localhost NFS server + sandbox-exec
- SQLite-backed delta layer：所有写入存储在 SQLite 数据库中
- Whiteout 机制处理删除

**局限**：
- **无分支/commit/abort 语义**：只提供单层 overlay，没有 branch context 的 fork/explore/commit 生命周期
- **无进程状态管理**：仅隔离文件系统
- **无 sibling invalidation**：没有 first-commit-wins
- **无嵌套**

**与 branch context 的关系**：AgentFS 解决了"agent 不应修改宿主文件系统"的安全问题，但不解决"多条探索路径之间的隔离和选择性提交"问题。

**来源**：[AgentFS Overlay 博客](https://turso.tech/blog/agentfs-overlay)；[AgentFS GitHub](https://github.com/tursodatabase/agentfs)

### 3.3 IBM STRATUS — 事务性 undo-and-retry

STRATUS 是 IBM Research + UIUC 合作的多 agent 系统，用于云事件响应，具有事务性 undo 机制。

**关键特性**：
- 每个 agent action sequence 形成一个 "transaction"
- Transaction 失败后 abort 并 revert 到 checkpoint
- Transactional-no-regression (TNR)：仅允许可逆操作
- 不可逆操作（如删除数据库）在执行前被拒绝
- Write locks 防止多 agent 同时修改

**局限**：
- **不处理文件系统/进程级状态**：STRATUS 操作的是云基础设施（Kubernetes, VM），不是本地文件系统
- **核心假设是"每个 action 必须可撤销"**——但 agent 执行的 shell 命令（pip install, make build）不一定有对应的 undo operator
- **Write locks 阻止并行**：不支持并行探索

**与 branch context 的关系**：STRATUS 在应用层面实现了类似 branch context 的 abort 语义，但它的 undo 依赖于**每个操作有对应的逆操作**。Branch context 通过 CoW 在 OS 层面提供通用的回滚，不需要知道操作的语义。

**来源**：[IBM Research 博客](https://research.ibm.com/blog/undo-agent-for-cloud)；[STRATUS 论文](https://yinfangchen.github.io/assets/pdf/stratus_paper.pdf)（NeurIPS 2025）

---

## 4. 首次成功率 / 投机成功率分析

### 4.1 已有数据

从 AgentCgroup 项目已有分析：
- **85-97% 的任务包含重试循环**（Haiku 85%, GLM 97%）
- **GLM 平均每个任务 3.9 个重试组**，最大连续重试 56 次
- **重试时间占比**：Haiku 7.4%, GLM 20.5%
- 这意味着**绝大多数任务不是一次成功的**，并行探索有巨大价值空间

### 4.2 待分析（需编写脚本）

- 首次尝试成功率（第一次测试就通过的任务占比）
- 将重试分类为"同策略重试"vs"新策略探索"
- 每条失败路径的计算浪费（工具调用总时间）
- 理论加速比建模

---

## 5. 跨工作空间修改分析

### 5.1 待从 trace 数据扫描的命令模式

| 命令模式 | 影响范围 | git 是否追踪 |
|----------|---------|-------------|
| `pip install` | `site-packages/`, `~/.local/lib/` | ✗ |
| `pip install --user` | `~/.local/lib/python*/` | ✗ |
| `npm install` | `node_modules/` | ✗（.gitignore） |
| `npm install -g` | `/usr/local/lib/node_modules/` | ✗ |
| `apt-get install` | `/usr/lib/`, `/usr/bin/` | ✗ |
| `git config --global` | `~/.gitconfig` | ✗ |
| `make` / `python setup.py build` | `build/`, `dist/`, `*.o` | ✗ |
| `pytest` | `.pytest_cache/`, `__pycache__/` | ✗ |

---

## 6. 存储放大分析

### 6.1 已有数据

- SWE-bench 镜像大小：平均 4.1GB，中位数 3.5GB，最大 17.3GB
- 仅 GLM 111 个任务的镜像总量即达 456GB

### 6.2 估算

| 方案 | 3 路并行存储需求 | 说明 |
|------|----------------|------|
| `cp -r` | 3 × 4.1GB = 12.3GB | 完全复制 |
| Container clone | 3 × ~100MB (upper dir) | OverlayFS CoW，但需 root |
| git worktree | 3 × (tracked files only) | 不含 node_modules 等 |
| BranchFS | 仅修改增量 (~<100MB) | FUSE CoW，无需 root |

**存储放大比**：cp -r vs BranchFS ≈ 4.1GB / <100MB ≈ **40× 以上**

---

## 7. 不可逆外部副作用——能力边界讨论

### 7.1 不可逆操作分类

| 操作类型 | 示例 | 可回滚性 |
|----------|------|---------|
| 本地文件修改 | `echo "x" >> file.py` | ✓ CoW 回滚 |
| 本地包安装 | `pip install X` | ✓ CoW 回滚（整个 venv） |
| 本地进程 | `pytest`, 编译器 | ✓ 进程终止 |
| HTTP 读请求 | `curl GET ...` | ✓ 无副作用 |
| HTTP 写请求 | `curl POST ...` | ✗ 已发送 |
| git push | `git push origin main` | ✗ 已推送 |
| 数据库写入 | SQL INSERT/UPDATE | ✗ 已写入 |
| 发送消息 | Slack/email API | ✗ 已发送 |

### 7.2 设计选择

参考数据库事务和 STRATUS 的做法：
1. **阻止**：branch 内禁止外部写操作（最安全，但限制功能）
2. **延迟**：记录外部写操作，commit 时才执行（类似 write-ahead log）
3. **补偿**：为每个外部操作注册 compensating action（类似 STRATUS，但不通用）
4. **警告**：允许执行但标记为"不可回滚"，abort 时告知用户

---

## 8. 与数据库事务模型的类比

| ACID 属性 | 数据库事务 | Branch Context | 差异/挑战 |
|----------|-----------|---------------|----------|
| **Atomicity** | 全部提交或全部回滚 | commit 全部变更或 abort 全部丢弃 | 相似，但 branch context 还需终止活进程 |
| **Consistency** | schema + 约束检查 | agent 自定义（如"测试通过"） | 无通用约束，由 agent 判定 |
| **Isolation** | 读已提交 / 快照隔离 / 串行化 | CoW 快照隔离（snapshot isolation） | 非常相似，branch context ≈ snapshot isolation |
| **Durability** | WAL + fsync | commit 后写入父级文件系统 | 相似 |

### Branch context 的独特挑战（超越数据库事务）

1. **状态空间极大**：文件系统 + 进程 + 内存 + 网络 + IPC，远超数据库的行/页
2. **无 schema**：无法自动检测一致性违反
3. **活进程管理**：数据库事务不需要终止正在运行的进程
4. **不可逆外部操作**：数据库事务只管自己的数据，branch context 的 agent 可能调用外部 API
5. **First-commit-wins**：类似 optimistic concurrency control，但需要自动 invalidate 兄弟分支的进程

---

## 9. 组合竞态窗口分析

用户空间拼凑 OverlayFS + cgroup + PID namespace 时的竞态窗口：

```
时间线：
T0: mount -t overlay ...              # 文件系统隔离建立
    ← 竞态窗口1: 进程已在运行但还在旧 PID ns 中
T1: unshare --pid --fork ...          # 进程隔离建立
    ← 竞态窗口2: 进程在新 PID ns 但无内存限制
T2: cgcreate -g memory:/branch_x      # 资源隔离建立
    ← 竞态窗口3: cgroup 已创建但进程尚未加入
T3: echo $PID > /sys/fs/cgroup/.../cgroup.procs  # 进程加入 cgroup
```

**问题**：
1. T0-T1 间，新 fork 的进程可能未进入 overlay mount namespace
2. T1-T3 间，进程已在运行但无资源限制
3. 任何步骤失败需手动回滚之前的步骤（error-prone）
4. 多个分支并发创建时，步骤间存在 TOCTOU 竞态

**对比 branch context**：一个 `branch()` syscall 原子地完成所有隔离的建立。
