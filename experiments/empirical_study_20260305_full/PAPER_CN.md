# 面向 Agentic 软件工程任务的跨仓库容器开销实证研究

## 摘要
随着基于大语言模型（LLM）的代码代理在软件工程任务中的广泛应用，容器镜像的依赖结构与运行时行为成为影响实验成本和系统可扩展性的关键因素。本文针对 4 个 SWE-bench 仓库镜像（`starlette`、`diff_cover`、`azure_msrest`、`pytorch_ignite`）开展端到端实证研究。我们在统一运行配置下采集 eBPF 内核事件、工具调用、CPU/内存采样及镜像内部依赖信息，系统回答三个问题：跨仓库运行开销差异有多大；镜像依赖重叠度如何；可低风险回收的空间有多少。结果显示：不同仓库间运行行为高度异质，其中 `pytorch_ignite` 的写入量达到 `16062.33 MB`，约为最小样本的 `971.9x`；`testbed` 环境 pip 依赖重叠度较低（Jaccard `0.137~0.260`）；仅通过清理 `conda/pip` 缓存可回收约 `4533 MB`。此外，基于 5 次 `starlette` 重复实验得到 `0.21~0.25` 的变异系数（CV），表明在当前 agent 交互模式下运行波动不可忽略。本文给出了分层优化建议：短期做缓存清理，中期按 ML/非 ML 分族优化基础镜像，长期统一构建链路提升层复用。

**关键词**：LLM Agent；SWE-bench；eBPF；容器镜像；依赖重叠；运行时开销；实证研究

## 1. 引言
AgentCgroup 系统此前已具备工具调用轨迹与容器级资源监控能力，但对“跨仓库开销差异从何而来”这一问题，仍缺乏 syscall 级、依赖级和镜像级的联合证据。传统指标（总时长、平均 CPU、内存）不足以区分“算法行为差异”与“环境依赖差异”造成的性能分化。

本文的目标并非仅报告一次任务成功率，而是构建一套可解释的经验分析框架：

1. 在运行期，基于 eBPF 量化文件系统与进程行为；
2. 在静态镜像层面，量化依赖重叠与可回收空间；
3. 将两者联系起来，给出可执行的优化路径。

## 2. 研究问题
我们定义以下研究问题（Research Questions, RQs）：

- **RQ1**：不同仓库镜像在真实 agent 运行下的动态开销差异有多大？
- **RQ2**：不同仓库镜像在依赖集合与层结构上重叠度如何？
- **RQ3**：在不改变任务语义的前提下，能回收多少低风险空间？

补充问题：

- **RQ4（鲁棒性）**：单仓库重复运行的波动量级如何，是否会影响跨仓库结论解释？

## 3. 实验设计
### 3.1 被试对象与运行环境
本文覆盖 4 个 SWE-bench 仓库镜像：

1. `swerebench/sweb.eval.x86_64.encode_1776_starlette-1147`
2. `swerebench/sweb.eval.x86_64.bachmann1234_1776_diff_cover-210`
3. `swerebench/sweb.eval.x86_64.azure_1776_msrest-for-python-224`
4. `swerebench/sweb.eval.x86_64.pytorch_1776_ignite-1077`

统一运行器：`scripts/run_swebench_new.py`。统一参数：

- `--trace-all --trace-cgroup-children`
- `--trace-resources --resource-detail`
- `--sample-interval 100`（tracer 采样间隔）
- `--resource-monitor-interval 0.5`
- 模型：`haiku`

### 3.2 数据采集
每次运行采集以下产物：

- `run_manifest.json`：容器与 tracer 编排元数据
- `ebpf_trace.jsonl`：内核级事件（EXEC/EXIT/FILE_OPEN/SUMMARY 等）
- `swebench/results.json`：总时长、工具调用、资源摘要
- `swebench/resources.json`：CPU/内存时间序列
- `swebench/tool_calls.json`：agent 工具调用轨迹

### 3.3 指标定义
动态指标：

- `runtime_s`：任务总时长
- `event_total`：eBPF 事件总量
- `write_mb`：SUMMARY:WRITE 总字节（MB）
- `event_per_s`、`write_mb_per_s`：归一化压力指标
- `fs_summary_share`：FS 相关 SUMMARY 占比

静态指标：

- `image_size_gb`：镜像大小
- `testbed_pip_count`：`/opt/conda/envs/testbed` pip 包数量
- `pip overlap (Jaccard)`：跨仓库依赖集合重叠度
- `reclaim_mb`：`/opt/conda/pkgs + /root/.cache/pip` 可回收空间估计

## 4. 实验结果
### 4.1 RQ1：跨仓库动态开销
表 1 给出核心运行结果。

| 仓库 | runtime_s | tool_calls | event_total | write_mb | event_per_s | write_mb_per_s | fs_summary_share_% | mem_avg_mb | cpu_avg_% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| azure_msrest | 157.41 | 44 | 9349 | 16.53 | 59.39 | 0.10 | 95.61 | 334.35 | 11.35 |
| diffcover | 132.89 | 49 | 12637 | 29.33 | 95.09 | 0.22 | 95.42 | 335.69 | 19.12 |
| pytorch_ignite | 317.35 | 44 | 67000 | 16062.33 | 211.13 | 50.61 | 79.74 | 476.82 | 42.30 |
| starlette | 190.79 | 39 | 10116 | 18.01 | 53.02 | 0.09 | 94.73 | 333.18 | 11.21 |

主要发现：

1. `pytorch_ignite` 在总时长、事件强度和写入量上均显著高于其他仓库；
2. 跨仓库写入量差异极端（max/min ≈ `971.9x`），说明“平均开销”不具代表性；
3. 对非 ML 仓库，FS 相关 SUMMARY 占比接近 `95%`，体现典型“文件系统驱动”的 agent 执行形态。

图 1~图 3 展示镜像规模、运行时长/工具调用和写入量对比。

![图1 镜像大小对比](figures/fig1_image_sizes.png)

![图2 运行时长与工具调用对比](figures/fig2_runtime_toolcalls.png)

![图3 写入量对比（对数坐标）](figures/fig3_write_volume_log.png)

### 4.2 RQ2：依赖重叠与镜像热点
表 2 为静态依赖与空间热点统计。

| 仓库 | image_gb | layers | testbed_pip_count | /opt/conda MB | /opt/conda/pkgs MB | /root/.cache/pip MB | /opt/conda/envs/testbed MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| azure_msrest | 3.24 | 6 | 40 | 1970 | 923 | 115 | 273 |
| diffcover | 3.45 | 17 | 57 | 1916 | 865 | 113 | 260 |
| pytorch_ignite | 6.35 | 6 | 134 | 3375 | 955 | 478 | 1646 |
| starlette | 3.71 | 6 | 85 | 2053 | 912 | 172 | 367 |

`testbed` pip 集合重叠（Jaccard）仅为 `0.137~0.260`，说明仓库间依赖差异显著，难以依赖“单一统一依赖层”覆盖全部任务。

![图4 testbed pip 依赖重叠热图](figures/fig4_pip_overlap_heatmap.png)

同时，`/opt/conda/pkgs` 与 `/root/.cache/pip` 在所有镜像中都形成可观冗余。

![图5 镜像内空间热点](figures/fig5_space_hotspots.png)

### 4.3 RQ3：可优化空间与结构性建议
低风险可回收空间估计（`conda/pip` 缓存）合计约 `4533 MB`（4 镜像）。按镜像占比约 `22.0%~31.3%`。

![图8 缓存回收量与占比](figures/fig8_cache_reclaim.png)

结合动态行为，优化建议分三层：

1. **短期（立即可做）**：构建后清理 `conda/pip` 缓存；
2. **中期（结构优化）**：按 ML/非 ML 分族维护基础镜像；
3. **长期（流水线统一）**：统一构建链路与层划分，提升跨仓库层复用。

### 4.4 RQ4：重复性与稳定性（补充）
为估计波动性，本文额外纳入 `starlette` 的 5 次历史有效运行样本。结果显示：

- `runtime_s` CV = `0.22`
- `event_total` CV = `0.23`
- `write_mb` CV = `0.25`
- `tool_calls` CV = `0.21`

说明在当前 agent 交互机制下，单次观测存在中等波动，跨仓库比较应优先采用归一化指标并补充重复实验。

![图10 Starlette 重复实验 CV](figures/fig10_starlette_repeat_cv.png)

## 5. 本研究相对原 AgentCgroup 论文数据的增量价值
相较于原有以工具轨迹与资源采样为主的数据形态，本文提供以下新增能力：

1. **syscall 级行为可解释性**：可直接量化 FS/NET/MMAP/PROC 维度；
2. **跨仓库写放大诊断**：识别 `pytorch_ignite` 的极端写入异常；
3. **依赖重叠可视化**：用 Jaccard 热图量化“统一依赖层”的可行边界；
4. **镜像冗余可估算**：路径级空间热点支持立刻落地的瘦身策略。

## 6. 有效性威胁
1. 跨仓库主实验当前每仓库 1 次运行，仍受 agent 随机性影响；
2. 运行行为同时受仓库复杂度与模型决策共同驱动，无法完全解耦；
3. 存储指标是镜像与路径级代理变量，未下钻到块级去重（dedupe）层面；
4. `pytorch_ignite` 为重型 outlier，虽能揭示上限风险，但会影响总体均值解释。

## 7. 结论
本文通过 4 个 SWE-bench 仓库镜像的统一实证实验表明：

1. 跨仓库开销差异远超直觉，尤其体现在写入放大和事件强度；
2. 镜像依赖重叠不足以支撑“单一统一依赖层”策略；
3. 缓存清理可带来立刻可验证的空间回收（约 4.5GB）；
4. 若要将本结论提升到更强统计显著性，应对每仓库进行多次重复并引入无 tracing/低频 tracing 对照组。

总体而言，当前结果已经足以支撑“跨仓库异质性显著且可优化空间明确”这一核心结论，并为 AgentCgroup 后续系统优化提供可执行路线。

## 附录：图表索引
- 图1：`figures/fig1_image_sizes.png`
- 图2：`figures/fig2_runtime_toolcalls.png`
- 图3：`figures/fig3_write_volume_log.png`
- 图4：`figures/fig4_pip_overlap_heatmap.png`
- 图5：`figures/fig5_space_hotspots.png`
- 图6：`figures/fig6_event_mix_stacked.png`
- 图7：`figures/fig7_normalized_pressure.png`
- 图8：`figures/fig8_cache_reclaim.png`
- 图9：`figures/fig9_summary_heatmap.png`
- 图10：`figures/fig10_starlette_repeat_cv.png`
